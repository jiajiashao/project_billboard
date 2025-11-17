import argparse
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

try:
    from transformers import Sam2VideoModel, Sam2VideoProcessor
except ModuleNotFoundError:
    print(
        "Missing dependency: transformers. Please install the SAM-2 requirements "
        "(transformers>=4.43.0 huggingface_hub opencv-python pillow numpy pandas tqdm)."
    )
    raise


RNG = np.random.default_rng(0)


def earliest_label_json(gt_dir: Path) -> Tuple[int, Path]:
    if not gt_dir.exists():
        parent = gt_dir.parent
        candidates = []
        if parent.exists():
            candidates = sorted(p.name for p in parent.iterdir() if p.is_dir())
        print(f"Did not find GT folder: {gt_dir}")
        if candidates:
            print("Available GT folders:")
            for name in candidates:
                print(f"  - {name}")
        raise SystemExit(1)

    frame_re = re.compile(r"frame_(\d+)\.json$")
    best: Optional[Tuple[int, Path]] = None
    for path in sorted(gt_dir.glob("frame_*.json")):
        match = frame_re.match(path.name)
        if not match:
            continue
        idx = int(match.group(1))
        if best is None or idx < best[0]:
            best = (idx, path)

    if best is None:
        print(f"Earliest label not found under {gt_dir}")
        raise SystemExit(2)

    return best


def load_labelme_polys(json_path: Path) -> Tuple[int, int, List[Tuple[int, np.ndarray]]]:
    with json_path.open("r") as f:
        data = json.load(f)

    width = int(data["imageWidth"])
    height = int(data["imageHeight"])
    polys: List[Tuple[int, np.ndarray]] = []

    fallback_gid = 0
    for shape in data.get("shapes", []):
        if shape.get("label") != "billboard":
            continue
        pts = np.array(shape.get("points", []), dtype=np.float32)
        if pts.shape[0] < 3:
            continue
        gid = shape.get("group_id")
        if gid is None:
            gid = fallback_gid
            fallback_gid += 1
        polys.append((int(gid), pts))

    return width, height, polys


def read_resize_frames(path: str, target_width: int) -> Tuple[List[Image.Image], Tuple[int, int], float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Could not open video: {path}")
        print("Try: ls data/clips to verify the filename.")
        raise SystemExit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or target_width
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or target_width
    scale = target_width / max(1, src_w)
    tgt_w = int(round(src_w * scale))
    tgt_h = int(round(src_h * scale))

    frames: List[Image.Image] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)
        frames.append(Image.fromarray(frame))
    cap.release()

    if not frames:
        print(f"No frames decoded from video: {path}")
        raise SystemExit(1)

    return frames, (tgt_w, tgt_h), float(fps)


def roi_from_polys_scaled(
    polys: Sequence[Tuple[int, np.ndarray]],
    src_w: int,
    src_h: int,
    tgt_w: int,
    tgt_h: int,
    pad: int,
) -> Tuple[int, int, int, int]:
    if not polys:
        raise ValueError("No polygons provided for ROI computation")

    sx = tgt_w / max(1, src_w)
    sy = tgt_h / max(1, src_h)
    scaled_pts: List[np.ndarray] = []
    for _, pts in polys:
        pts = np.asarray(pts, dtype=np.float32)
        scaled = np.column_stack((pts[:, 0] * sx, pts[:, 1] * sy))
        scaled_pts.append(scaled)

    all_pts = np.vstack(scaled_pts)
    x0 = math.floor(float(all_pts[:, 0].min()) - pad)
    y0 = math.floor(float(all_pts[:, 1].min()) - pad)
    x1 = math.ceil(float(all_pts[:, 0].max()) + pad)
    y1 = math.ceil(float(all_pts[:, 1].max()) + pad)

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(tgt_w, x1)
    y1 = min(tgt_h, y1)

    if x1 <= x0:
        x1 = min(tgt_w, x0 + 1)
    if y1 <= y0:
        y1 = min(tgt_h, y0 + 1)

    return x0, y0, x1, y1


def bbox_from_polys_scaled(
    polys: Sequence[Tuple[int, np.ndarray]],
    src_w: int,
    src_h: int,
    tgt_w: int,
    tgt_h: int,
    pad: int,
    min_width: int = 2,
    min_height: int = 2,
) -> Tuple[float, float, float, float]:
    """Compute a padded bounding box around the provided polygons in the resized space."""

    x0, y0, x1, y1 = roi_from_polys_scaled(polys, src_w, src_h, tgt_w, tgt_h, pad)

    if x1 - x0 < min_width:
        x1 = min(tgt_w, x0 + max(min_width, 1))
    if y1 - y0 < min_height:
        y1 = min(tgt_h, y0 + max(min_height, 1))

    # Convert to inclusive coordinates expected by SAM-2 box prompts.
    x_max = max(float(x0 + 1), float(x1 - 1))
    y_max = max(float(y0 + 1), float(y1 - 1))
    return float(x0), float(y0), float(x_max), float(y_max)


def sample_points_roi(
    polys: Sequence[Tuple[int, np.ndarray]],
    src_w: int,
    src_h: int,
    tgt_w: int,
    tgt_h: int,
    roi: Tuple[int, int, int, int],
    max_points: int = 12,
) -> List[np.ndarray]:
    x0, y0, x1, y1 = roi
    roi_w = max(1, x1 - x0)
    roi_h = max(1, y1 - y0)

    sx = tgt_w / max(1, src_w)
    sy = tgt_h / max(1, src_h)

    grouped: Dict[int, List[np.ndarray]] = defaultdict(list)
    for gid, pts in polys:
        pts = np.asarray(pts, dtype=np.float32)
        scaled = np.column_stack((pts[:, 0] * sx, pts[:, 1] * sy))
        shifted = scaled - np.array([x0, y0], dtype=np.float32)
        grouped[int(gid)].append(shifted)

    positives: List[np.ndarray] = []
    for segments in grouped.values():
        mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        for seg in segments:
            if seg.shape[0] < 3:
                continue
            cv2.fillPoly(mask, [np.round(seg).astype(np.int32)], 255)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        take = min(max_points, len(xs))
        idx = RNG.choice(len(xs), size=take, replace=False)
        pts = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
        positives.append(pts)

    return positives


def make_clicks_with_negatives(
    polys: Sequence[Tuple[int, np.ndarray]],
    src_w: int,
    src_h: int,
    tgt_w: int,
    tgt_h: int,
    roi: Tuple[int, int, int, int],
    pos_per_obj: int = 12,
    neg_stride: int = 30,
    neg_offset: int = 6,
) -> Tuple[Optional[List[List[List[List[float]]]]], Optional[List[List[List[int]]]]]:
    positives = sample_points_roi(polys, src_w, src_h, tgt_w, tgt_h, roi, max_points=pos_per_obj)
    if not positives:
        return None, None

    pos_array = np.concatenate(positives, axis=0)
    if pos_array.size == 0:
        return None, None

    pos_points = pos_array.tolist()

    x0, y0, x1, y1 = roi
    sx = tgt_w / max(1, src_w)
    sy = tgt_h / max(1, src_h)

    all_scaled = []
    for _, pts in polys:
        arr = np.asarray(pts, dtype=np.float32)
        arr = np.column_stack((arr[:, 0] * sx, arr[:, 1] * sy))
        all_scaled.append(arr)

    if not all_scaled:
        return None, None

    merged = np.vstack(all_scaled)
    left = int(np.floor(merged[:, 0].min()))
    right = int(np.ceil(merged[:, 0].max()))
    top = int(np.floor(merged[:, 1].min()))
    bottom = int(np.ceil(merged[:, 1].max()))

    neg_offset = max(0, int(neg_offset))
    top_line = max(y0, top - neg_offset)
    bottom_line = min(y1 - 1, bottom + neg_offset)

    span = max(1, right - left)
    samples = min(8, span)  # up to 8 pairs of negatives
    xs = np.linspace(left, right, num=samples, dtype=np.int32)
    xs = np.unique(xs)

    negs: List[List[float]] = []
    for x in xs:
        x_clamped = min(max(int(x), x0), x1 - 1)
        negs.append([float(x_clamped - x0), float(top_line - y0)])
        negs.append([float(x_clamped - x0), float(bottom_line - y0)])

    points = [[pos_points + negs]]
    labels = [[[1] * len(pos_points) + [0] * len(negs)]]
    return points, labels


def make_box_with_edge_negatives(
    polys: Sequence[Tuple[int, np.ndarray]],
    src_w: int,
    src_h: int,
    tgt_w: int,
    tgt_h: int,
    pad: int,
    neg_count: int = 4,
    neg_offset: int = 6,
    add_center_positive: bool = False,
) -> Tuple[
    Tuple[float, float, float, float],
    Optional[List[List[List[List[float]]]]],
    Optional[List[List[List[int]]]],
]:
    """Return a padded bounding box prompt and optional edge-fence negatives."""

    if not polys:
        return (0.0, 0.0, 1.0, 1.0), None, None

    x0, y0, x1, y1 = bbox_from_polys_scaled(polys, src_w, src_h, tgt_w, tgt_h, pad)
    box = (x0, y0, x1, y1)

    points: List[List[List[List[float]]]] = []
    labels: List[List[List[int]]] = []

    per_object_points: List[List[List[float]]] = [[]]
    per_object_labels: List[List[int]] = [[]]

    if add_center_positive:
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        per_object_points[0].append([cx, cy])
        per_object_labels[0].append(1)

    if neg_count > 0:
        width = float(tgt_w)
        height = float(tgt_h)
        offset = max(0.0, float(neg_offset))
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0

        candidates = []
        if neg_count >= 1:
            candidates.append([cx, max(0.0, y0 - offset)])
        if neg_count >= 2:
            candidates.append([cx, min(height - 1.0, y1 + offset)])
        if neg_count >= 3:
            candidates.append([max(0.0, x0 - offset), cy])
        if neg_count >= 4:
            candidates.append([min(width - 1.0, x1 + offset), cy])

        seen = set()
        for pt in candidates:
            key = (round(pt[0], 3), round(pt[1], 3))
            if key in seen:
                continue
            seen.add(key)
            per_object_points[0].append(pt)
            per_object_labels[0].append(0)

    if per_object_points[0]:
        points = [per_object_points]
        labels = [per_object_labels]
        return box, points, labels

    return box, None, None


def largest_component(mask_bool: np.ndarray) -> np.ndarray:
    if mask_bool.dtype != np.bool_:
        mask_bool = mask_bool.astype(bool)
    if not mask_bool.any():
        return mask_bool
    num_labels, labels = cv2.connectedComponents(mask_bool.astype(np.uint8))
    if num_labels <= 1:
        return mask_bool
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    best = int(np.argmax(areas))
    return labels == best


def fill_small_holes(mask_bool: np.ndarray, max_frac: float) -> np.ndarray:
    if mask_bool.dtype != np.bool_:
        mask_bool = mask_bool.astype(bool)
    total = mask_bool.sum()
    if total == 0:
        return mask_bool
    inv = (~mask_bool).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(inv)
    h, w = mask_bool.shape
    for label in range(1, num_labels):
        component = labels == label
        if not component.any():
            continue
        ys, xs = np.nonzero(component)
        if (ys == 0).any() or (ys == h - 1).any() or (xs == 0).any() or (xs == w - 1).any():
            continue
        area = component.sum()
        if area / float(total) <= max_frac:
            mask_bool[component] = True
    return mask_bool


def prob_to_mask(
    prob: np.ndarray,
    threshold: float = 0.65,
    shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    arr = np.asarray(prob, dtype=np.float32)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[-2], arr.shape[-1])
    elif arr.ndim == 1 and shape is not None:
        arr = arr.reshape(shape)
    elif arr.ndim == 0 and shape is not None:
        arr = np.full(shape, float(arr), dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    mask = (arr > threshold).astype(np.uint8) * 255
    if mask.shape[0] >= 3 and mask.shape[1] >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def overlay_mask(rgb: np.ndarray, mask_u8: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = bgr.copy()
    mask = mask_u8 > 0
    if np.any(mask):
        purple_bgr = np.array([255, 0, 170], dtype=np.uint8)
        overlay[mask] = purple_bgr
    return cv2.addWeighted(overlay, alpha, bgr, 1 - alpha, 0)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SAM-2 smoke test runner")
    ap.add_argument("--clip", required=True, help="Path to the MP4 clip")
    ap.add_argument("--gt-root", "--gt_root", dest="gt_root", default="data/gt_frames")
    ap.add_argument("--weights", default="models/sam2.1-hiera-tiny")
    ap.add_argument("--target-width", "--target_width", dest="target_width", type=int, default=640)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--roi-pad", "--roi_pad", dest="roi_pad", type=int, default=16)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument(
        "--max-frames",
        "--max_frames",
        dest="max_frames",
        type=int,
        default=None,
        help="Optional limit on number of frames to decode",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    clip_path = Path(args.clip)
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

    src_w, src_h, polys = load_labelme_polys(seed_json)
    if not polys:
        print(f"No 'billboard' polygons in {seed_json.name}")
        raise SystemExit(1)

    pad = int(args.roi_pad)
    clicks = (None, None)
    roi = (0, 0, tgt_w, tgt_h)
    for attempt in range(2):
        roi = roi_from_polys_scaled(polys, src_w, src_h, tgt_w, tgt_h, pad)
        clicks = make_clicks_with_negatives(
            polys,
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

    print(f"Device: {('mps' if torch.backends.mps.is_available() else 'cpu')}")
    print(f"Seeding from {seed_json.name}")
    print(f"ROI: ({x0},{y0},{x1},{y1}) size={roi_w}x{roi_h}")

    frames_roi = [im.crop((x0, y0, x1, y1)) for im in frames]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float32

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

    stride = max(1, int(args.stride))
    target_indices = set(range(seed_idx, total_frames, stride))
    remaining_targets = sorted(idx for idx in target_indices if idx > seed_idx)

    processed = 0
    propagation_elapsed = 0.0
    if remaining_targets:
        start = time.perf_counter()
        for output in model.propagate_in_video_iterator(
            inference_session=session, start_frame_idx=seed_idx + 1
        ):
            idx = output.frame_idx
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
            processed += 1
            if processed == len(remaining_targets):
                break
        propagation_elapsed = time.perf_counter() - start

    total_processed = 1 + processed
    total_time = seed_time + propagation_elapsed
    fps_measured = total_processed / max(1e-6, total_time)
    print(
        f"Processed {total_processed} frames at ~{fps_measured:.2f} fps "
        f"(stride={args.stride}, target_width={args.target_width})"
    )

    last_mask = np.zeros((tgt_h, tgt_w), dtype=np.uint8)
    for i in range(total_frames):
        if masks_full[i] is None:
            masks_full[i] = last_mask.copy()
        else:
            last_mask = masks_full[i]

    out_dir = Path(args.outdir) / clip_id / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip_id}_sam2_smoke.mp4"

    writer = cv2.VideoWriter(
        str(out_path),
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
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"Overlay saved: {out_path}")
    else:
        print(f"Overlay failed to write: {out_path}")
        print("If codecs are missing, try: brew install ffmpeg")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
