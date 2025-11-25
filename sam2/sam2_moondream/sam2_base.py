import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
    bbox_from_polys_scaled,
    earliest_label_json,
    load_labelme_polys,
    make_box_with_edge_negatives,
    overlay_mask,
    prob_to_mask,
    read_resize_frames,
)
from sam2_pilot import (
    band_mask,
    centroid_from_mask,
    compute_iou,
    nan_mean,
    nan_percentiles,
    polys_to_mask,
    write_per_frame_csv,
)


RUN_SPEC = {
    "run_id": "gt",
    "model": {
        "name": "sam2",
        "random_seed": 42,
        "deterministic": True,
    },
    # "clips": [
    #     {
    #         "id": "clip_gentle",
    #         "input_width": 1024,
    #         "stride": 2,
    #         "full_frame": True,
    #         "seed": {
    #             "mode": "first_labeled_frame",
    #             "from_gt_bbox": True,
    #             "bbox_pad_px": 6,
    #             "negatives": {
    #                 "mode": "edge_fence",
    #                 "count": 4,
    #                 "offset_px": 6,
    #             },
    #         },
    #         "reseed": {
    #             "enabled": False,
    #             "triggers": {
    #                 "centroid_jump_pct_of_width": 5.0,
    #                 "area_change_pct": 30.0,
    #                 "empty_mask_consecutive": 2,
    #             },
    #             "action": "reseed_with_box_plus_neg",
    #             "cooldown_frames": 3,
    #             "max_events": 0,
    #         },
    #     },
    #     {
    #         "id": "clip_fast",
    #         "input_width": 1280,
    #         "stride": 1,
    #         "full_frame": True,
    #         "seed": {
    #             "mode": "first_labeled_frame",
    #             "from_gt_bbox": True,
    #             "bbox_pad_px": 6,
    #             "negatives": {
    #                 "mode": "edge_fence",
    #                 "count": 4,
    #                 "offset_px": 6,
    #             },
    #         },
    #         "reseed": {
    #             "enabled": True,
    #             "triggers": {
    #                 "centroid_jump_pct_of_width": 5.0,
    #                 "area_change_pct": 30.0,
    #                 "empty_mask_consecutive": 2,
    #             },
    #             "action": "reseed_with_box_plus_neg",
    #             "cooldown_frames": 3,
    #             "max_events": 0,
    #         },
    #     },
    # ],
    "metrics": {
        "compute": ["iou", "biou", "jitter_norm_pct"],
        "scoring_policy": {
            "skip_frames_with_empty_gt": True,
            "include_seed_frame": True,
        },
        # "pass_fail": {
        #     "gentle": {
        #         "iou_median_min": 0.60,
        #         "jitter_norm_median_pct_max": 1.0,
        #     },
        #     "fast": {
        #         "iou_median_min": 0.50,
        #     },
        # },
    },
    "logging": {
        "write_params_json": True,
        "write_script_hash": True,
        "capture_git_commit": True,
        "echo_config_to_log": True,
    },
    "safety": {
        "if_out_dir_exists": "create_new",
    },
}

AREA_EPS = 1.0
MASK_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM-2 pilot runner for Fix-2")
    parser.add_argument("--data-root", dest="data_root", default="..\sam2\data")
    parser.add_argument("--weights", default="facebook/sam2.1-hiera-tiny")
    parser.add_argument("--runs-root", dest="runs_root", default="runs")
    parser.add_argument("--clips", nargs="*", help="Optional subset of clip IDs to process")
    parser.add_argument("--device", choices=["cpu", "mps"], default=None)
    return parser.parse_args()


def select_device(preferred: Optional[str]) -> str:
    if preferred:
        if preferred == "mps" and not torch.backends.mps.is_available():
            print("Requested mps device but it is unavailable; falling back to cpu")
            return "cpu"
        return preferred
    return "mps" if torch.backends.mps.is_available() else "cpu"


def set_random_state(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except RuntimeError:
        pass


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def prepare_run_dir(root: Path, run_id: str) -> Tuple[Path, str]:
    ensure_dir(root)
    candidate = root / run_id
    if not candidate.exists():
        candidate.mkdir()
        return candidate, candidate.name
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / f"{run_id}_{timestamp}"
    candidate.mkdir()
    return candidate, candidate.name


def get_git_commit(root: Path) -> str:
    if not (root / ".git").exists():
        return "N/A"
    import subprocess

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        return commit or "N/A"
    except Exception:
        return "N/A"


def clone_nested(data: Optional[Sequence]) -> Optional[List]:
    if data is None:
        return None
    return json.loads(json.dumps(data))


def bbox_from_mask(
    mask_bool: np.ndarray,
    pad_px: int,
    width: int,
    height: int,
) -> Optional[Tuple[float, float, float, float]]:
    mask = np.asarray(mask_bool, dtype=bool)
    if not mask.any():
        return None

    ys, xs = np.nonzero(mask)
    x0 = max(0, math.floor(float(xs.min()) - pad_px))
    y0 = max(0, math.floor(float(ys.min()) - pad_px))
    x1 = min(width - 1, math.ceil(float(xs.max()) + pad_px))
    y1 = min(height - 1, math.ceil(float(ys.max()) + pad_px))

    if x1 <= x0:
        x1 = min(width - 1, x0 + 1)
    if y1 <= y0:
        y1 = min(height - 1, y0 + 1)

    x_max = max(float(x0 + 1), float(x1))
    y_max = max(float(y0 + 1), float(y1))
    return float(x0), float(y0), x_max, y_max


def infer_frame(
    model: Sam2VideoModel,
    processor: Sam2VideoProcessor,
    session,
    frame_idx: int,
    tgt_size: Tuple[int, int],
) -> Tuple[np.ndarray, float]:
    start = time.perf_counter()
    output = model(inference_session=session, frame_idx=frame_idx)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    post = processor.post_process_masks(
        [output.pred_masks],
        original_sizes=[[tgt_size[1], tgt_size[0]]],
        binarize=False,
    )[0]
    if isinstance(post, torch.Tensor):
        post = post.detach().cpu().numpy()
    prob = np.max(post, axis=0) if post.ndim == 3 else post
    mask = prob_to_mask(prob, threshold=MASK_THRESHOLD, shape=tgt_size[::-1])
    return mask, runtime_ms


def create_logger() -> Tuple[List[str], Callable[[str], None]]:
    lines: List[str] = []

    def _log(msg: str) -> None:
        print(msg)
        lines.append(msg)

    return lines, _log


def write_json(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def write_reprompt_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    fields = [
        "event_idx",
        "frame_idx",
        "reasons",
        "centroid_jump_px",
        "centroid_jump_pct",
        "area_change_pct",
        "empty_streak",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _iter_labelme_polys(gt_dir: Path) -> Sequence[Tuple[int, Path, int, int, List[Tuple[int, np.ndarray]]]]:
    frame_paths = sorted(gt_dir.glob("frame_*.json"))
    for path in frame_paths:
        frame_idx = int(path.stem.split("_")[-1])
        src_w, src_h, polys = load_labelme_polys(path)
        yield frame_idx, path, src_w, src_h, polys


def load_frames_and_seed(
    clip_cfg: Dict,
    clip_path: Path,
    gt_dir: Path,
    target_width: int,
) -> Tuple[
    List,
    Tuple[int, int],
    float,
    int,
    Path,
    Tuple[float, float, float, float],
    Optional[List],
    Optional[List],
    Dict[int, Tuple[float, float, float, float]],
]:
    frames, (tgt_w, tgt_h), fps = read_resize_frames(str(clip_path), target_width)
    seed_idx, seed_json = earliest_label_json(gt_dir)
    seed_src_w = seed_src_h = None
    seed_polys: Optional[List[Tuple[int, np.ndarray]]] = None
    pad = int(clip_cfg["seed"]["bbox_pad_px"])
    gt_boxes: Dict[int, Tuple[float, float, float, float]] = {}
    for idx, json_path, src_w, src_h, polys in _iter_labelme_polys(gt_dir):
        if polys:
            gt_boxes[idx] = bbox_from_polys_scaled(polys, src_w, src_h, tgt_w, tgt_h, pad)
            seed_idx = idx
            seed_json = json_path
            seed_src_w = src_w
            seed_src_h = src_h
            seed_polys = polys
            break
    if seed_polys is None:
        raise SystemExit(f"No 'billboard' polygons found under {gt_dir}")
    if seed_idx >= len(frames):
        raise SystemExit(
            f"Seed frame {seed_idx} exceeds available frames ({len(frames)}). "
            "Increase input width or verify the clip."
        )
    box = bbox_from_polys_scaled(seed_polys, seed_src_w, seed_src_h, tgt_w, tgt_h, pad)
    points = None
    labels = None
    neg_cfg = clip_cfg["seed"].get("negatives", {})
    if neg_cfg.get("mode") == "edge_fence":
        _, points, labels = make_box_with_edge_negatives(
            seed_polys,
            seed_src_w,
            seed_src_h,
            tgt_w,
            tgt_h,
            pad,
            neg_count=int(neg_cfg.get("count", 0)),
            neg_offset=int(neg_cfg.get("offset_px", 0)),
        )
    return frames, (tgt_w, tgt_h), fps, seed_idx, seed_json, box, points, labels, gt_boxes


def process_clip(
    clip_cfg: Dict,
    context: Dict,
    model: Sam2VideoModel,
    processor: Sam2VideoProcessor,
    run_dir: Path,
    git_commit: str,
    script_hash: str,
) -> Dict[str, object]:
    clip_id = clip_cfg["id"]
    clip_dir = run_dir / clip_id
    ensure_dir(clip_dir)
    log_lines, log = create_logger()

    data_root: Path = context["data_root"]
    clip_path = data_root / "clips" / f"{clip_id}.mp4"
    gt_dir = data_root / "gt_frames" / clip_id
    if not clip_path.exists():
        raise SystemExit(
            f"Clip not found: {clip_path}. Try: ls {clip_path.parent} to verify the filename."
        )

    (
        frames,
        (tgt_w, tgt_h),
        fps,
        seed_idx,
        seed_json,
        seed_box,
        seed_points,
        seed_labels,
        gt_boxes,
    ) = load_frames_and_seed(
        clip_cfg,
        clip_path,
        gt_dir,
        clip_cfg["input_width"],
    )

    device = context["device"]
    log(f"Processing {clip_id}")
    log(f"  Device: {device}")
    log(f"  Frames decoded: {len(frames)} @ {fps:.2f} fps -> {tgt_w}x{tgt_h}")
    log(f"  Seeding from {seed_json.name} (frame {seed_idx})")
    log(
        "  Seed box (padded): ({:.1f}, {:.1f}, {:.1f}, {:.1f})".format(
            seed_box[0], seed_box[1], seed_box[2], seed_box[3]
        )
    )

    session = processor.init_video_session(
        video=frames,
        inference_device=device,
        processing_device=device,
        video_storage_device=device,
        dtype=context["dtype"],
    )

    bbox_pad_px = int(clip_cfg.get("seed", {}).get("bbox_pad_px", 0))
    seed_prompt_boxes = [[[seed_box[0], seed_box[1], seed_box[2], seed_box[3]]]]
    seed_prompt_points = clone_nested(seed_points)
    seed_prompt_labels = clone_nested(seed_labels)

    processor.add_inputs_to_inference_session(
        inference_session=session,
        frame_idx=seed_idx,
        obj_ids=1,
        input_boxes=seed_prompt_boxes,
        input_points=seed_prompt_points,
        input_labels=seed_prompt_labels,
    )

    total_frames = len(frames)
    masks_full: List[np.ndarray] = [np.zeros((tgt_h, tgt_w), dtype=np.uint8) for _ in range(total_frames)]
    frame_runtimes: Dict[int, float] = {}
    stride = max(1, int(clip_cfg["stride"]))
    target_indices = set(range(seed_idx, total_frames, stride))

    prev_centroid: Optional[Tuple[float, float]] = None
    prev_area: Optional[float] = None
    empty_streak = 0
    cooldown_remaining = 0
    reseed_cfg = clip_cfg.get("reseed", {})
    reseed_enabled = reseed_cfg.get("enabled", False)
    triggers = reseed_cfg.get("triggers", {})
    cooldown_frames = int(reseed_cfg.get("cooldown_frames", 0))
    max_events = int(reseed_cfg.get("max_events", 0))
    reseed_events: List[Dict[str, object]] = []
    reseed_count = 0

    with torch.inference_mode():
        for idx in range(seed_idx, total_frames):
            mask, runtime_ms = infer_frame(model, processor, session, idx, (tgt_w, tgt_h))
            masks_full[idx] = mask
            if idx in target_indices:
                frame_runtimes[idx] = frame_runtimes.get(idx, 0.0) + runtime_ms

                mask_bool = mask > 0
                area = float(mask_bool.sum())
                is_empty = area <= AREA_EPS
                centroid = centroid_from_mask(mask_bool) if not is_empty else (math.nan, math.nan)

                if is_empty:
                    empty_streak += 1
                else:
                    empty_streak = 0

                jump_px = math.nan
                jump_pct = math.nan
                if not is_empty and prev_centroid is not None and not math.isnan(prev_centroid[0]):
                    dx = centroid[0] - prev_centroid[0]
                    dy = centroid[1] - prev_centroid[1]
                    jump_px = math.hypot(dx, dy)
                    jump_pct = (jump_px / max(1.0, tgt_w)) * 100.0

                area_change_pct = math.nan
                if prev_area is not None and prev_area > AREA_EPS and area > AREA_EPS:
                    area_change_pct = abs(area - prev_area) / prev_area * 100.0

                triggers_allowed = reseed_enabled and cooldown_remaining == 0 and reseed_count < max_events
                reasons: List[str] = []
                if triggers_allowed:
                    if not math.isnan(jump_pct) and jump_pct > float(triggers.get("centroid_jump_pct_of_width", math.inf)):
                        reasons.append("centroid_jump")
                    if not math.isnan(area_change_pct) and area_change_pct > float(
                        triggers.get("area_change_pct", math.inf)
                    ):
                        reasons.append("area_change")
                    if empty_streak >= int(triggers.get("empty_mask_consecutive", math.inf)):
                        reasons.append("empty_mask_consecutive")

                if reasons:
                    reseed_count += 1
                    cooldown_remaining = cooldown_frames
                    event_record = {
                        "event_idx": reseed_count,
                        "frame_idx": idx,
                        "reasons": ",".join(reasons),
                        "centroid_jump_px": float(jump_px) if not math.isnan(jump_px) else "",
                        "centroid_jump_pct": float(jump_pct) if not math.isnan(jump_pct) else "",
                        "area_change_pct": float(area_change_pct) if not math.isnan(area_change_pct) else "",
                        "empty_streak": empty_streak,
                    }
                    reseed_events.append(event_record)
                    log(
                        f"  Reseed #{reseed_count} at frame {idx} due to {event_record['reasons']}"
                    )

                    dynamic_box = bbox_from_mask(mask_bool, bbox_pad_px, tgt_w, tgt_h)
                    if dynamic_box is None:
                        if gt_boxes:
                            nearest_idx = min(gt_boxes.keys(), key=lambda frame_idx: abs(frame_idx - idx))
                            dynamic_box = gt_boxes[nearest_idx]
                        else:
                            dynamic_box = (
                                seed_prompt_boxes[0][0][0],
                                seed_prompt_boxes[0][0][1],
                                seed_prompt_boxes[0][0][2],
                                seed_prompt_boxes[0][0][3],
                            )
                    reseed_boxes = [[[dynamic_box[0], dynamic_box[1], dynamic_box[2], dynamic_box[3]]]]

                    processor.add_inputs_to_inference_session(
                        inference_session=session,
                        frame_idx=idx,
                        obj_ids=1,
                        input_boxes=reseed_boxes,
                        input_points=None,
                        input_labels=None,
                    )
                    mask, runtime_extra = infer_frame(model, processor, session, idx, (tgt_w, tgt_h))
                    masks_full[idx] = mask
                    frame_runtimes[idx] = frame_runtimes.get(idx, 0.0) + runtime_extra
                    mask_bool = mask > 0
                    area = float(mask_bool.sum())
                    is_empty = area <= AREA_EPS
                    centroid = centroid_from_mask(mask_bool) if not is_empty else (math.nan, math.nan)
                    if is_empty:
                        empty_streak = min(empty_streak, int(triggers.get("empty_mask_consecutive", 0)))
                    else:
                        empty_streak = 0
                elif cooldown_remaining > 0:
                    cooldown_remaining -= 1

                prev_area = area if area > AREA_EPS else prev_area
                prev_centroid = centroid if not math.isnan(centroid[0]) else prev_centroid

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    overlay_path = clip_dir / f"overlay_{clip_id}_{RUN_SPEC['run_id']}.mp4"
    writer = cv2.VideoWriter(
        str(overlay_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (tgt_w, tgt_h),
    )
    for frame, mask in zip(frames, masks_full):
        rgb = np.array(frame)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        bgr = overlay_mask(rgb, cleaned)
        writer.write(bgr)
    writer.release()
    if overlay_path.exists() and overlay_path.stat().st_size == 0:
        log(f"Overlay failed to write: {overlay_path}")
        log("If codecs are missing, try: brew install ffmpeg")
        raise SystemExit(1)

    label_paths = sorted(gt_dir.glob("frame_*.json"))
    frame_rows: List[Dict[str, object]] = []
    prev_metric_centroid = (math.nan, math.nan)

    for label_path in label_paths:
        frame_idx = int(label_path.stem.split("_")[-1])
        if frame_idx >= len(masks_full):
            continue
        src_w, src_h, polys = load_labelme_polys(label_path)
        if RUN_SPEC["metrics"]["scoring_policy"]["skip_frames_with_empty_gt"] and not polys:
            continue
        gt_mask = polys_to_mask(polys, src_w, src_h, tgt_w, tgt_h)
        if RUN_SPEC["metrics"]["scoring_policy"]["skip_frames_with_empty_gt"] and not gt_mask.any():
            continue

        pred_mask = masks_full[frame_idx] > 0
        gt_mask_bool = gt_mask > 0

        iou = compute_iou(pred_mask, gt_mask_bool)
        biou = compute_iou(band_mask(pred_mask), band_mask(gt_mask_bool))
        area = float(pred_mask.sum())
        cx, cy = centroid_from_mask(pred_mask)
        is_empty = 1 if area <= AREA_EPS else 0

        shift_px = math.nan
        shift_norm = math.nan
        if frame_idx in frame_runtimes and not math.isnan(cx) and not math.isnan(prev_metric_centroid[0]):
            dx = cx - prev_metric_centroid[0]
            dy = cy - prev_metric_centroid[1]
            shift_px = math.hypot(dx, dy)
            shift_norm = (shift_px / max(1.0, tgt_w)) * 100.0
        if frame_idx in frame_runtimes and not math.isnan(cx):
            prev_metric_centroid = (cx, cy)
        elif frame_idx in frame_runtimes and math.isnan(cx):
            prev_metric_centroid = (math.nan, math.nan)

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
            "runtime_ms": frame_runtimes.get(frame_idx, math.nan),
            "roi_w": tgt_w,
            "roi_h": tgt_h,
        }
        frame_rows.append(row)

    per_frame_path = clip_dir / f"per_frame_{clip_id}_{RUN_SPEC['run_id']}.csv"
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
    write_per_frame_csv(per_frame_path, frame_rows, columns)

    ious = [row["iou"] for row in frame_rows]
    biou_vals = [row["biou"] for row in frame_rows]
    jitter_vals = [
        val
        for val in (row["centroid_shift_norm"] for row in frame_rows)
        if isinstance(val, float) and not math.isnan(val)
    ]
    area_vals = [row["area_px"] for row in frame_rows]
    empty_vals = [row["is_empty"] for row in frame_rows]

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

    processed_count = len(target_indices)
    total_time_s = sum(frame_runtimes.get(idx, 0.0) for idx in target_indices) / 1000.0
    fps_measured = processed_count / total_time_s if total_time_s > 0 else 0.0
    fps_theoretical = fps / stride

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
        "proc_fps_theoretical": fps_theoretical,
        "roi_w": tgt_w,
        "roi_h": tgt_h,
        "target_W": tgt_w,
        "target_H": tgt_h,
        "stride": stride,
    }

    log(
        "  Processed {} frames at ~{:.2f} fps (theoretical {:.2f})".format(
            processed_count, fps_measured, fps_theoretical
        )
    )
    log(
        "  Metrics: IoU_med={:.3f}, BIoU_med={:.3f}, jitter_med={:.3f}%/frame".format(
            iou_med if not math.isnan(iou_med) else float("nan"),
            biou_med if not math.isnan(biou_med) else float("nan"),
            jitter_med if not math.isnan(jitter_med) else float("nan"),
        )
    )

    if RUN_SPEC["logging"].get("write_params_json", False):
        params_path = clip_dir / f"params_{clip_id}_{RUN_SPEC['run_id']}.json"
        params_payload = {
            "run_id": RUN_SPEC["run_id"],
            "clip_id": clip_id,
            "clip_path": str(clip_path),
            "gt_dir": str(gt_dir),
            "input_width": clip_cfg["input_width"],
            "stride": stride,
            "full_frame": clip_cfg.get("full_frame", False),
            "seed": {
                "frame_index": seed_idx,
                "json": str(seed_json),
                "bbox_pad_px": clip_cfg["seed"]["bbox_pad_px"],
                "box_xyxy": list(seed_box),
                "negatives": clip_cfg["seed"].get("negatives", {}),
            },
            "reseed": clip_cfg.get("reseed", {}),
            "model": {
                "weights": str(context["weights"]),
                "device": device,
                "dtype": str(context["dtype"]),
            },
            "script_hash": script_hash if RUN_SPEC["logging"].get("write_script_hash", False) else "N/A",
            "git_commit": git_commit if RUN_SPEC["logging"].get("capture_git_commit", False) else "N/A",
        }
        write_json(params_path, params_payload)

    re_prompt_path = clip_dir / f"re_prompts_{clip_id}_{RUN_SPEC['run_id']}.csv"
    write_reprompt_csv(re_prompt_path, reseed_events)

    log_path = clip_dir / f"pilot_{clip_id}_{RUN_SPEC['run_id']}.log"
    ensure_dir(log_path.parent)
    if RUN_SPEC["logging"].get("echo_config_to_log", False):
        log("  Config snapshot: ")
        cfg_snapshot = {
            "clip": clip_cfg,
            "seed_frame": seed_idx,
            "seed_box": list(seed_box),
            "device": device,
            "dtype": str(context["dtype"]),
            "weights": str(context["weights"]),
        }
        log(json.dumps(cfg_snapshot, indent=2))
    with log_path.open("w") as f:
        for line in log_lines:
            f.write(f"{line}\n")

    return {
        "clip_id": clip_id,
        "summary": summary,
        "per_frame_path": per_frame_path,
        "per_frame_rows": len(frame_rows),
        "overlay_path": overlay_path,
        "re_prompts_path": re_prompt_path,
        "log_path": log_path,
        "params_path": clip_dir / f"params_{clip_id}_{RUN_SPEC['run_id']}.json",
        "reseed_events": reseed_events,
        "fps_measured": fps_measured,
        "iou_median": iou_med,
        "biou_median": biou_med,
        "jitter_med": jitter_med,
        "empty_pct": empty_pct,
    }


def write_summary(run_dir: Path, summaries: List[Dict[str, object]]) -> Path:
    summary_path = run_dir / f"summary_{RUN_SPEC['run_id']}.csv"
    fields = [
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
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)
    return summary_path


def format_pass_fail(value: bool) -> str:
    return "PASS" if value else "FAIL"


def write_run_notes(
    outputs_root: Path,
    run_folder_name: str,
    clip_results: Dict[str, Dict[str, object]],
) -> None:
    ensure_dir(outputs_root)
    notes_path = outputs_root / "RUN_NOTES.md"
    now_iso = dt.datetime.now().isoformat(timespec="seconds")
    host = platform.node()
    python_version = platform.python_version()

    gentle = clip_results.get("clip_gentle", {})
    fast = clip_results.get("clip_fast", {})

    gentle_iou = gentle.get("iou_median", math.nan)
    gentle_jitter = gentle.get("jitter_med", math.nan)
    fast_iou = fast.get("iou_median", math.nan)

    gentle_pass = (
        not math.isnan(gentle_iou)
        and gentle_iou >= RUN_SPEC["metrics"]["pass_fail"]["gentle"]["iou_median_min"]
        and not math.isnan(gentle_jitter)
        and gentle_jitter <= RUN_SPEC["metrics"]["pass_fail"]["gentle"]["jitter_norm_median_pct_max"]
    )
    fast_pass = (
        not math.isnan(fast_iou)
        and fast_iou >= RUN_SPEC["metrics"]["pass_fail"]["fast"]["iou_median_min"]
    )

    notes_lines = [
        "# Run Notes — SAM-2 Pilot (Fix-2)",
        f"Date/Time: {now_iso} | Machine: {host} | Python: {python_version}",
        f"Run ID: {RUN_SPEC['run_id']} | Output: {outputs_root}/{run_folder_name}",
        f"Model: {RUN_SPEC['model']['name']} (seed={RUN_SPEC['model']['random_seed']} | deterministic={RUN_SPEC['model']['deterministic']})",
        "Clips: clip_gentle, clip_fast (processed=2)",
        "Input width/stride: gentle=1024/2, fast=1280/1",
        "Seeding: earliest “billboard” frame; pad=6px; negatives=4 at 6px",
        "Auto re-prompt: centroid>5% | area>30% | empty×2; cooldown=3; max=20/30",
        "Artifacts: per_frame.csv, overlays/overlay.mp4, re_prompts.csv, params.json; run summary.csv",
        (
            "Result: gentle IoU_med={:.3f}, jitter_med={:.3f}%/frame | fast IoU_med={:.3f} "
            "(Pass/Fail gentle={}, fast={})"
        ).format(
            gentle_iou if not math.isnan(gentle_iou) else float("nan"),
            gentle_jitter if not math.isnan(gentle_jitter) else float("nan"),
            fast_iou if not math.isnan(fast_iou) else float("nan"),
            format_pass_fail(gentle_pass),
            format_pass_fail(fast_pass),
        ),
    ]

    with notes_path.open("w") as f:
        for line in notes_lines:
            f.write(f"{line}\n")


def main() -> None:
    args = parse_args()

    run_id = RUN_SPEC["run_id"]
    data_root = Path(args.data_root)
    runs_root = Path(args.runs_root)
    run_dir, run_folder_name = prepare_run_dir(runs_root, run_id)

    device = select_device(args.device)
    dtype = torch.float32

    set_random_state(RUN_SPEC["model"]["random_seed"])

    model = Sam2VideoModel.from_pretrained(args.weights).to(device=device, dtype=dtype)
    model.eval()
    processor = Sam2VideoProcessor.from_pretrained(args.weights)

    context = {
        "data_root": data_root,
        "device": device,
        "dtype": dtype,
        "weights": Path(args.weights),
    }

    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    git_commit = get_git_commit(Path.cwd()) if RUN_SPEC["logging"].get("capture_git_commit", False) else "N/A"

    selected_clips = set(args.clips) if args.clips else None
    clip_results: Dict[str, Dict[str, object]] = {}
    summaries: List[Dict[str, object]] = []

    for clip_cfg in RUN_SPEC["clips"]:
        if selected_clips and clip_cfg["id"] not in selected_clips:
            continue
        result = process_clip(clip_cfg, context, model, processor, run_dir, git_commit, script_hash)
        clip_results[clip_cfg["id"]] = result
        summaries.append(result["summary"])

    if not summaries:
        raise SystemExit("No clips were processed")

    summary_path = write_summary(run_dir, summaries)
    print(f"Summary saved to {summary_path}")

    # outputs_root = Path("outputs/sam2_pilot_fix2")
    # write_run_notes(outputs_root, run_folder_name, clip_results)

    for clip_id, result in clip_results.items():
        status_line = (
            f"✔ {clip_id}: IoU_med={result['iou_median']:.3f}, "
            f"BIoU_med={result['biou_median']:.3f}, "
            f"jitter_med={result['jitter_med']:.3f}%/frame, "
            f"empty={result['empty_pct']:.2f}%, "
            f"FPS={result['fps_measured']:.2f}"
        )
        print(status_line)


if __name__ == "__main__":
    main()
