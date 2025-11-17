import argparse
import csv
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

try:
    from transformers import Sam2VideoModel, Sam2VideoProcessor
except ModuleNotFoundError:
    print(
        "Missing dependency: transformers. Please install the SAM-2 requirements "
        "(transformers>=4.43.0 huggingface_hub opencv-python pillow numpy pandas tqdm)."
    )
    raise

from sam2_smoke import (
    earliest_label_json,
    load_labelme_polys,
    make_clicks_with_negatives,
    overlay_mask,
    prob_to_mask,
    read_resize_frames,
    roi_from_polys_scaled,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM-2 pilot runner")
    parser.add_argument("--clips", nargs="+", help="List of MP4 clips to process")
    parser.add_argument("--gt-root", "--gt_root", dest="gt_root", default="data/gt_frames")
    parser.add_argument("--weights", default="models/sam2.1-hiera-tiny")
    parser.add_argument("--target-width", "--target_width", dest="target_width", type=int, default=640)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--roi-pad", "--roi_pad", dest="roi_pad", type=int, default=16)
    parser.add_argument("--outdir", default="outputs/sam2_pilot")
    parser.add_argument(
        "--max-frames",
        "--max_frames",
        dest="max_frames",
        type=int,
        default=None,
        help="Optional limit on number of frames to decode per clip",
    )
    return parser.parse_args()


def polys_to_mask(
    polys: Sequence[Tuple[int, np.ndarray]],
    src_w: int,
    src_h: int,
    tgt_w: int,
    tgt_h: int,
) -> np.ndarray:
    if not polys:
        return np.zeros((tgt_h, tgt_w), dtype=np.uint8)

    sx = tgt_w / max(1, src_w)
    sy = tgt_h / max(1, src_h)

    grouped: Dict[int, List[np.ndarray]] = {}
    for gid, pts in polys:
        pts = np.asarray(pts, dtype=np.float32)
        scaled = np.column_stack((pts[:, 0] * sx, pts[:, 1] * sy))
        grouped.setdefault(int(gid), []).append(scaled)

    mask = np.zeros((tgt_h, tgt_w), dtype=np.uint8)
    for segments in grouped.values():
        polygons = [np.round(seg).astype(np.int32) for seg in segments if seg.shape[0] >= 3]
        if polygons:
            cv2.fillPoly(mask, polygons, 255)

    return mask


def band_mask(mask_bool: np.ndarray, dilate_px: int = 3) -> np.ndarray:
    if not mask_bool.any():
        return np.zeros_like(mask_bool, dtype=bool)

    mask = mask_bool.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grad = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        grad = cv2.dilate(grad, dilate_kernel)
    return grad > 0


def compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def centroid_from_mask(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return math.nan, math.nan
    return float(xs.mean()), float(ys.mean())


def write_per_frame_csv(path: Path, rows: List[Dict[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            serialised = {}
            for key, value in row.items():
                if isinstance(value, float) and math.isnan(value):
                    serialised[key] = ""
                else:
                    serialised[key] = value
            writer.writerow(serialised)


def nan_percentiles(data: Iterable[float], q: Sequence[float]) -> List[float]:
    arr = np.array([d for d in data if not math.isnan(d)], dtype=np.float32)
    if arr.size == 0:
        return [math.nan for _ in q]
    return np.percentile(arr, q=q).tolist()


def nan_mean(data: Iterable[float]) -> float:
    arr = np.array([d for d in data if not math.isnan(d)], dtype=np.float32)
    if arr.size == 0:
        return math.nan
    return float(arr.mean())


def run_clip(
    clip_path: Path,
    args: argparse.Namespace,
    output_root: Path,
) -> Dict[str, object]:
    clip_id = clip_path.stem

    frames, (tgt_w, tgt_h), fps = read_resize_frames(str(clip_path), args.target_width)
    if args.max_frames is not None:
        max_f = max(0, int(args.max_frames))
        frames = frames[:max_f]
        if not frames:
            print("No frames remaining after applying --max-frames limit")
            raise SystemExit(1)

    gt_dir = Path(args.gt_root) / clip_id
    seed_idx, seed_json = earliest_label_json(gt_dir)
    if seed_idx >= len(frames):
        print(
            f"Seed frame {seed_idx} exceeds available frames ({len(frames)}). "
            "Increase --max-frames or target-width to decode more frames."
        )
        raise SystemExit(1)

    src_w, src_h, seed_polys = load_labelme_polys(seed_json)
    if not seed_polys:
        print(f"No 'billboard' polygons in {seed_json.name}")
        raise SystemExit(1)

    pad = int(args.roi_pad)
    clicks = (None, None)
    roi = (0, 0, tgt_w, tgt_h)
    for attempt in range(2):
        roi = roi_from_polys_scaled(seed_polys, src_w, src_h, tgt_w, tgt_h, pad)
        clicks = make_clicks_with_negatives(
            seed_polys,
            src_w,
            src_h,
            tgt_w,
            tgt_h,
            roi,
            pos_per_obj=12,
            neg_stride=30,
            neg_offset=6,
        )
        if clicks[0] is not None:
            break
        print("Could not sample interior points; increasing ROI pad by +8 and retrying...")
        pad += 8
    if clicks[0] is None:
        x0, y0, x1, y1 = roi
        print(
            f"Failed to sample interior/negative points from {seed_json.name} within ROI"
            f" ({x0},{y0},{x1},{y1})."
        )
        raise SystemExit(1)

    x0, y0, x1, y1 = roi
    roi_w = x1 - x0
    roi_h = y1 - y0

    print(f"Processing {clip_id}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float32
    print(f"  Device: {device}")
    print(f"  Seeding from {seed_json.name}")
    print(f"  ROI: ({x0},{y0},{x1},{y1}) size={roi_w}x{roi_h}")

    frames_roi = [frame.crop((x0, y0, x1, y1)) for frame in frames]

    model = Sam2VideoModel.from_pretrained(args.weights).to(device=device, dtype=dtype)
    processor = Sam2VideoProcessor.from_pretrained(args.weights)
    session = processor.init_video_session(
        video=frames_roi,
        inference_device=device,
        processing_device=device,
        video_storage_device=device,
        dtype=dtype,
    )

    processor.add_inputs_to_inference_session(
        inference_session=session,
        frame_idx=seed_idx,
        obj_ids=1,
        input_points=clicks[0],
        input_labels=clicks[1],
    )

    total_frames = len(frames)
    masks_full: List[Optional[np.ndarray]] = [None] * total_frames
    runtime_ms: Dict[int, float] = {}
    roi_shape = [[roi_h, roi_w]]

    seed_start = time.perf_counter()
    seed_output = model(inference_session=session, frame_idx=seed_idx)
    seed_time = time.perf_counter() - seed_start
    seed_post = processor.post_process_masks(
        [seed_output.pred_masks], original_sizes=roi_shape, binarize=False
    )[0]
    if isinstance(seed_post, torch.Tensor):
        seed_post = seed_post.detach().cpu().numpy()
    seed_prob = np.max(seed_post, axis=0) if seed_post.ndim == 3 else seed_post
    seed_roi = prob_to_mask(seed_prob, shape=(roi_h, roi_w))
    seed_full = np.zeros((tgt_h, tgt_w), dtype=np.uint8)
    seed_full[y0:y1, x0:x1] = seed_roi[:roi_h, :roi_w]
    masks_full[seed_idx] = seed_full
    runtime_ms[seed_idx] = seed_time * 1000.0

    stride = max(1, int(args.stride))
    target_indices = set(range(seed_idx, total_frames, stride))
    remaining_targets = sorted(idx for idx in target_indices if idx > seed_idx)

    processed = 0
    propagation_elapsed = 0.0
    if remaining_targets:
        start_time = time.perf_counter()
        last_time = start_time
        for output in model.propagate_in_video_iterator(
            inference_session=session, start_frame_idx=seed_idx + 1
        ):
            idx = output.frame_idx
            now = time.perf_counter()
            frame_runtime = (now - last_time) * 1000.0
            last_time = now

            if idx >= total_frames:
                continue
            if idx not in target_indices:
                continue

            post = processor.post_process_masks(
                [output.pred_masks], original_sizes=roi_shape, binarize=False
            )[0]
            if isinstance(post, torch.Tensor):
                post = post.detach().cpu().numpy()
            prob = np.max(post, axis=0) if post.ndim == 3 else post
            mask_roi = prob_to_mask(prob, shape=(roi_h, roi_w))
            full_mask = np.zeros((tgt_h, tgt_w), dtype=np.uint8)
            full_mask[y0:y1, x0:x1] = mask_roi[:roi_h, :roi_w]
            masks_full[idx] = full_mask
            runtime_ms[idx] = frame_runtime

            processed += 1
            if processed == len(remaining_targets):
                break
        propagation_elapsed = time.perf_counter() - start_time

    total_processed = len(target_indices)
    total_time = seed_time + propagation_elapsed
    fps_measured = total_processed / max(1e-6, total_time)

    last_mask = np.zeros((tgt_h, tgt_w), dtype=np.uint8)
    for i in range(total_frames):
        if masks_full[i] is None:
            masks_full[i] = last_mask.copy()
        else:
            last_mask = masks_full[i]

    overlay_dir = output_root / clip_id / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = overlay_dir / f"{clip_id}_sam2.mp4"
    writer = cv2.VideoWriter(
        str(overlay_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (tgt_w, tgt_h),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for frame, mask in zip(frames, masks_full):
        rgb = np.array(frame)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        bgr = overlay_mask(rgb, cleaned)
        writer.write(bgr)
    writer.release()
    if overlay_path.exists() and overlay_path.stat().st_size == 0:
        print(f"Overlay failed to write: {overlay_path}")
        print("If codecs are missing, try: brew install ffmpeg")
        raise SystemExit(1)

    label_paths = sorted(gt_dir.glob("frame_*.json"))
    frame_data: List[Dict[str, object]] = []
    prev_centroid = (math.nan, math.nan)

    theoretical_fps = fps / stride

    for label_path in label_paths:
        frame_idx = int(label_path.stem.split("_")[-1])
        if frame_idx == seed_idx or frame_idx >= len(masks_full):
            continue
        if frame_idx < seed_idx:
            continue

        src_w_i, src_h_i, polys = load_labelme_polys(label_path)
        gt_mask_full = polys_to_mask(polys, src_w_i, src_h_i, tgt_w, tgt_h)

        pred_mask_full = masks_full[frame_idx] > 0
        gt_mask_bool = gt_mask_full > 0

        pred_roi = pred_mask_full[y0:y1, x0:x1]
        gt_roi = gt_mask_bool[y0:y1, x0:x1]

        iou = compute_iou(pred_roi, gt_roi)

        pred_band = band_mask(pred_roi)
        gt_band = band_mask(gt_roi)
        biou = compute_iou(pred_band, gt_band)

        area = float(pred_roi.sum())
        cx, cy = centroid_from_mask(pred_roi)

        is_empty = 1 if math.isnan(cx) else 0

        shift_px = math.nan
        shift_norm = math.nan
        if frame_idx in runtime_ms and not math.isnan(cx) and not math.isnan(prev_centroid[0]):
            dx = cx - prev_centroid[0]
            dy = cy - prev_centroid[1]
            shift_px = math.hypot(dx, dy)
            shift_norm = (shift_px / max(1, roi_w)) * 100.0
        if frame_idx in runtime_ms and math.isnan(cx):
            prev_centroid = (math.nan, math.nan)
        elif frame_idx in runtime_ms and not math.isnan(cx):
            prev_centroid = (cx, cy)

        runtime_val = runtime_ms.get(frame_idx, math.nan)

        row = {
            "clip_id": clip_id,
            "frame_no": frame_idx,
            "iou": float(iou),
            "biou": float(biou),
            "area_px": float(area),
            "centroid_x": cx,
            "centroid_y": cy,
            "centroid_shift_px": shift_px,
            "centroid_shift_norm": shift_norm,
            "is_empty": is_empty,
            "runtime_ms": runtime_val,
            "roi_w": roi_w,
            "roi_h": roi_h,
        }
        frame_data.append(row)

    per_frame_path = output_root / clip_id / "per_frame.csv"
    columns = [
        "clip_id",
        "frame_no",
        "iou",
        "biou",
        "area_px",
        "centroid_x",
        "centroid_y",
        "centroid_shift_px",
        "centroid_shift_norm",
        "is_empty",
        "runtime_ms",
        "roi_w",
        "roi_h",
    ]
    write_per_frame_csv(per_frame_path, frame_data, columns)

    ious = [row["iou"] for row in frame_data]
    biou_vals = [row["biou"] for row in frame_data]
    jitter_vals = [
        val
        for val in (row["centroid_shift_norm"] for row in frame_data)
        if isinstance(val, float) and not math.isnan(val)
    ]
    area_vals = [row["area_px"] for row in frame_data]
    empty_vals = [row["is_empty"] for row in frame_data]

    iou_p25, iou_med, iou_p75 = nan_percentiles(ious, [25, 50, 75])
    biou_p25, biou_med, biou_p75 = nan_percentiles(biou_vals, [25, 50, 75])
    jitter_med, jitter_p95 = nan_percentiles(jitter_vals, [50, 95]) if jitter_vals else (math.nan, math.nan)

    area_mean = nan_mean(area_vals)
    area_std = math.nan
    if not math.isnan(area_mean):
        arr = np.array(area_vals, dtype=np.float32)
        area_std = float(np.std(arr))
    area_cv = math.nan if math.isnan(area_mean) or area_mean == 0 else (area_std / area_mean) * 100.0

    empty_pct = (sum(empty_vals) / len(empty_vals) * 100.0) if empty_vals else math.nan

    summary = {
        "clip_id": clip_id,
        "iou_median": iou_med,
        "iou_p25": iou_p25,
        "iou_p75": iou_p75,
        "biou_median": biou_med,
        "biou_p25": biou_p25,
        "biou_p75": biou_p75,
        "jitter_norm_median_pct": jitter_med,
        "jitter_norm_p95_pct": jitter_p95,
        "area_cv_pct": area_cv,
        "empty_frames_pct": empty_pct,
        "proc_fps_measured": fps_measured,
        "proc_fps_theoretical": theoretical_fps,
        "roi_w": roi_w,
        "roi_h": roi_h,
        "target_W": tgt_w,
        "target_H": tgt_h,
        "stride": stride,
    }

    print(
        "  Processed {} frames at ~{:.2f} fps (theoretical {:.2f})".format(
            total_processed, fps_measured, theoretical_fps
        )
    )

    return summary


def main() -> None:
    args = parse_args()
    if not args.clips:
        raise SystemExit("No clips provided")

    output_root = Path(args.outdir)
    summaries: List[Dict[str, object]] = []

    for clip in args.clips:
        summary = run_clip(Path(clip), args, output_root)
        summaries.append(summary)

    summary_path = output_root / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "clip_id",
        "iou_median",
        "iou_p25",
        "iou_p75",
        "biou_median",
        "biou_p25",
        "biou_p75",
        "jitter_norm_median_pct",
        "jitter_norm_p95_pct",
        "area_cv_pct",
        "empty_frames_pct",
        "proc_fps_measured",
        "proc_fps_theoretical",
        "roi_w",
        "roi_h",
        "target_W",
        "target_H",
        "stride",
    ]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)

    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
