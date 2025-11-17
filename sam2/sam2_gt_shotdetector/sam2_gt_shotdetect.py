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
from typing import Dict, List, Optional, Sequence, Tuple

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

import sys
from pathlib import Path as _PathForSys

# Ensure repo root (parent of sam2_gt) is importable for sibling modules
_THIS = _PathForSys(__file__).resolve()
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from shot_detection import detect_shots
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
    "run_id": "shotgt",
    "model": {
        "name": "sam2",
        "random_seed": 42,
        "deterministic": True,
    },
    "clips": [
        {
            "id": "clip_gentle",
            "input_width": 1024,
            "stride": 2,
            "full_frame": True,
            "seed": {
                "bbox_pad_px": 6,
            },
        },
        {
            "id": "clip_fast",
            "input_width": 1280,
            "stride": 1,
            "full_frame": True,
            "seed": {
                "bbox_pad_px": 6,
            },
        },
    ],
    "metrics": {
        "compute": ["iou", "biou", "jitter_norm_pct"],
        "scoring_policy": {
            "skip_frames_with_empty_gt": True,
            "include_seed_frame": True,
        },
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
    parser = argparse.ArgumentParser(
        description="SAM-2 pilot (GT-shot reseed): reseed at each shot start using nearest GT bbox"
    )
    parser.add_argument("--data-root", dest="data_root", default="data")
    parser.add_argument("--weights", default="models/sam2.1-hiera-tiny")
    parser.add_argument("--runs-root", dest="runs_root", default="runs")
    parser.add_argument("--clips", nargs="*", help="Optional subset of clip IDs to process")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cuda")
    # Shot detection controls
    parser.add_argument("--shot-method", choices=["adaptive", "content"], default="adaptive")
    parser.add_argument("--shot-min-seconds", type=float, default=1.0)
    return parser.parse_args()


def select_device(preferred: Optional[str]) -> str:
    if preferred:
        if preferred == "cuda" and not torch.cuda.is_available():
            print("Requested mps device but it is unavailable; falling back to cpu")
            return "cpu"
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


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


def build_gt_boxes_map(gt_dir: Path, tgt_w: int, tgt_h: int, pad: int) -> Dict[int, Tuple[float, float, float, float]]:
    boxes: Dict[int, Tuple[float, float, float, float]] = {}
    for idx, _, src_w, src_h, polys in _iter_labelme_polys(gt_dir):
        if polys:
            boxes[idx] = bbox_from_polys_scaled(polys, src_w, src_h, tgt_w, tgt_h, pad)
    return boxes


def nearest_gt_box(gt_boxes: Dict[int, Tuple[float, float, float, float]], target_idx: int,
                   fallback_box: Optional[Tuple[float, float, float, float]]) -> Optional[Tuple[float, float, float, float]]:
    if gt_boxes:
        nearest = min(gt_boxes.keys(), key=lambda j: abs(j - target_idx))
        return gt_boxes[nearest]
    return fallback_box


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

    data_root: Path = context["data_root"]
    clip_path = data_root / "clips" / f"{clip_id}.mp4"
    gt_dir = data_root / "gt_frames" / clip_id
    if not clip_path.exists():
        raise SystemExit(
            f"Clip not found: {clip_path}. Try: ls {clip_path.parent} to verify the filename."
        )

    frames, (tgt_w, tgt_h), fps = read_resize_frames(str(clip_path), clip_cfg["input_width"])
    device = context["device"]

    # Build GT boxes map and a simple fallback seed from earliest GT
    pad = int(clip_cfg.get("seed", {}).get("bbox_pad_px", 0))
    gt_boxes = build_gt_boxes_map(gt_dir, tgt_w, tgt_h, pad)
    seed_idx0, seed_json0 = earliest_label_json(gt_dir)
    fallback_box: Optional[Tuple[float, float, float, float]] = None
    if seed_json0 is not None and Path(seed_json0).exists():
        try:
            src_w, src_h, polys0 = load_labelme_polys(seed_json0)
            if polys0:
                fallback_box = bbox_from_polys_scaled(polys0, src_w, src_h, tgt_w, tgt_h, pad)
        except Exception:
            fallback_box = None

    # Detect shots (always on for this runner)
    try:
        shots = detect_shots(clip_path, total_frames=len(frames), fps=fps, method=context["shot_method"], min_shot_len_s=context["shot_min_seconds"])
        shot_bounds: List[Tuple[int, int]] = [(int(s.start), int(s.end)) for s in shots]
    except Exception:
        shot_bounds = [(0, len(frames))]
    shot_starts = sorted({max(0, int(s)) for s, _ in shot_bounds})

    # Init SAM-2 session
    session = processor.init_video_session(
        video=frames,
        inference_device=device,
        processing_device=device,
        video_storage_device=device,
        dtype=context["dtype"],
    )

    # Metrics bookkeeping
    total_frames = len(frames)
    masks_full: List[np.ndarray] = [np.zeros((tgt_h, tgt_w), dtype=np.uint8) for _ in range(total_frames)]
    frame_runtimes: Dict[int, float] = {}
    stride = max(1, int(clip_cfg["stride"]))
    target_indices = set(range(0, total_frames, stride))

    # Logging
    log_lines: List[str] = []
    def log(msg: str) -> None:
        print(msg)
        log_lines.append(msg)

    log(f"Processing {clip_id}")
    log(f"  Device: {device}")
    log(f"  Frames decoded: {len(frames)} @ {fps:.2f} fps -> {tgt_w}x{tgt_h}")
    log(f"  Shots detected: {len(shot_bounds)}")

    # Reseed events CSV
    reseed_events: List[Dict[str, object]] = []
    event_count = 0

    # Main loop with shot-start reseeding using nearest GT bbox
    with torch.inference_mode():
        for idx in range(0, total_frames):
            if idx in shot_starts:
                # Find nearest GT bbox and add prompt at frame idx
                b = nearest_gt_box(gt_boxes, idx, fallback_box)
                if b is not None:
                    processor.add_inputs_to_inference_session(
                        inference_session=session,
                        frame_idx=idx,
                        obj_ids=1,
                        input_boxes=[[[float(b[0]), float(b[1]), float(b[2]), float(b[3])]]],
                        input_points=None,
                        input_labels=None,
                    )
                    event_count += 1
                    reseed_events.append({
                        "event_idx": event_count,
                        "frame_idx": idx,
                        "reasons": "shot_start",
                        "centroid_jump_px": "",
                        "centroid_jump_pct": "",
                        "area_change_pct": "",
                        "empty_streak": 0,
                    })
                    log(f"  Reseed at shot start {idx} using nearest GT bbox")

            # Infer current frame and record runtime
            mask, runtime_ms = infer_frame(model, processor, session, idx, (tgt_w, tgt_h))
            masks_full[idx] = mask
            if idx in target_indices:
                frame_runtimes[idx] = frame_runtimes.get(idx, 0.0) + runtime_ms

    # Write overlay
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

    # Metrics against GT
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
    total_time_s = sum(frame_runtimes.get(i, 0.0) for i in target_indices) / 1000.0
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

    # Params JSON
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
                "bbox_pad_px": clip_cfg["seed"].get("bbox_pad_px", 0),
            },
            "model": {
                "weights": str(context["weights"]),
                "device": device,
                "dtype": str(context["dtype"]),
            },
            "script_hash": script_hash if RUN_SPEC["logging"].get("write_script_hash", False) else "N/A",
            "git_commit": git_commit if RUN_SPEC["logging"].get("capture_git_commit", False) else "N/A",
            "shots": shot_bounds,
        }
        write_json(params_path, params_payload)

    # Re-seed events CSV
    re_prompt_path = clip_dir / f"re_prompts_{clip_id}_{RUN_SPEC['run_id']}.csv"
    write_reprompt_csv(re_prompt_path, reseed_events)

    # Log file
    log_path = clip_dir / f"pilot_{clip_id}_{RUN_SPEC['run_id']}.log"
    ensure_dir(log_path.parent)
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

    notes_lines = [
        "# Run Notes – SAM-2 Pilot (Shot-GT reseed)",
        f"Date/Time: {now_iso} | Machine: {host} | Python: {python_version}",
        f"Run ID: {RUN_SPEC['run_id']} | Output: {outputs_root}/{run_folder_name}",
        f"Model: {RUN_SPEC['model']['name']} (seed={RUN_SPEC['model']['random_seed']} | deterministic={RUN_SPEC['model']['deterministic']})",
        "Clips: clip_gentle, clip_fast (processed=2)",
        "Seeding: nearest GT bbox at each shot start; pad=6px",
        "Artifacts: per_frame.csv, overlays/overlay.mp4, re_prompts.csv, params.json; run summary.csv",
        (
            "Result snapshot: gentle IoU_med={:.3f}, jitter_med={:.3f}%/frame | fast IoU_med={:.3f}"
        ).format(
            gentle_iou if not math.isnan(gentle_iou) else float("nan"),
            gentle_jitter if not math.isnan(gentle_jitter) else float("nan"),
            fast_iou if not math.isnan(fast_iou) else float("nan"),
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
        "shot_method": args.shot_method,
        "shot_min_seconds": float(args.shot_min_seconds),
    }

    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    git_commit = get_git_commit(Path.cwd()) if RUN_SPEC["logging"].get("capture_git_commit", False) else "N/A"

        # Build list of clip configs to process (supports custom IDs via --clips)
    clips_to_process: List[Dict[str, object]] = []
    cfg_by_id = {cfg["id"]: cfg for cfg in RUN_SPEC["clips"]}
    if args.clips:
        for cid in args.clips:
            if cid in cfg_by_id:
                clips_to_process.append(cfg_by_id[cid])
            else:
                clips_to_process.append({
                    "id": cid,
                    "input_width": 1280,
                    "stride": 1,
                    "full_frame": True,
                    "seed": {"bbox_pad_px": 6},
                })
    else:
        clips_to_process = list(RUN_SPEC["clips"])
    clip_results: Dict[str, Dict[str, object]] = {}
    summaries: List[Dict[str, object]] = []

    for clip_cfg in clips_to_process:
        result = process_clip(clip_cfg, context, model, processor, run_dir, git_commit, script_hash)
        clip_results[clip_cfg["id"]] = result
        summaries.append(result["summary"])

    if not summaries:
        raise SystemExit("No clips were processed")

    summary_path = write_summary(run_dir, summaries)
    print(f"Summary saved to {summary_path}")

    outputs_root = Path("outputs/sam2_gt_shotdetect")
    write_run_notes(outputs_root, run_folder_name, clip_results)

    for clip_id, result in clip_results.items():
        status_line = (
            f"{clip_id}: IoU_med={result['iou_median']:.3f}, "
            f"BIoU_med={result['biou_median']:.3f}, "
            f"jitter_med={result['jitter_med']:.3f}%/frame, "
            f"empty={result['empty_pct']:.2f}%, "
            f"FPS={result['fps_measured']:.2f}"
        )
        print(status_line)


if __name__ == "__main__":
    main()





