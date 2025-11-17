import sys
import json
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Ensure project root is importable so sam2_smoke/sam2_pilot resolve
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sam2_pilot_fix2 as base
from shot_detection import detect_shots
from autoprompt_yolo import YoloBoxPromptor


def parse_args() -> base.argparse.Namespace:
    parser = base.argparse.ArgumentParser(description="SAM-2 with YOLO autoprompt (per-shot sessions)")
    parser.add_argument("--data-root", dest="data_root", default="data")
    parser.add_argument("--weights", default="models/sam2.1-hiera-tiny")
    parser.add_argument("--runs-root", dest="runs_root", default="runs")
    parser.add_argument("--clips", nargs="*", help="Optional subset of clip IDs to process")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    # YOLO autoprompt
    parser.add_argument("--auto-prompt-yolo", action="store_true", default=True)
    parser.add_argument("--yolo-model", default="runs/detect/train7/weights/best.pt")
    parser.add_argument("--yolo-conf-thr", type=float, default=0.20)
    parser.add_argument("--yolo-max-objects", type=int, default=3)
    parser.add_argument("--yolo-device", default="0")
    # Optional GT fallback (none|gt)
    parser.add_argument("--autoprompt-fallback", choices=["none", "gt"], default="none")
    return parser.parse_args()


def select_device(preferred: Optional[str]) -> str:
    if preferred:
        if preferred == "cuda" and not base.torch.cuda.is_available():
            print("Requested cuda but unavailable; falling back to cpu")
            return "cpu"
        if preferred == "mps" and not base.torch.backends.mps.is_available():
            print("Requested mps but unavailable; falling back to cpu")
            return "cpu"
        return preferred
    if base.torch.cuda.is_available():
        return "cuda"
    return "mps" if base.torch.backends.mps.is_available() else "cpu"


# Shared state for CSV/JPG augmentation
_STATE = {"shot_rows": [], "events": [], "last_frames": [], "log": None}


def _ensure_requested_in_runspec(requested: List[str], data_root: Path) -> None:
    if not requested:
        return
    existing = {cfg.get("id") for cfg in base.RUN_SPEC.get("clips", [])}
    for cid in requested:
        if cid in existing:
            continue
        if (data_root / "clips" / f"{cid}.mp4").exists():
            base.RUN_SPEC["clips"].append({
                "id": cid,
                "input_width": 1280,
                "stride": 1,
                "full_frame": True,
                "seed": {"mode": "none", "from_gt_bbox": False, "negatives": None, "bbox_pad_px": 6},
                "reseed": {"enabled": False, "triggers": {}, "action": "reseed_with_box_plus_neg", "cooldown_frames": 0, "max_events": 0},
            })


def _to_bgr(frame) -> Optional["base.np.ndarray"]:
    try:
        import numpy as np
        import cv2
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return None
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        try:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception:
            return arr
    except Exception:
        return None


def _first_gt_box_in_segment(gt_dir: Path, s: int, e: int, tgt_w: int, tgt_h: int, pad: int = 6) -> Optional[Tuple[int, Tuple[float, float, float, float]]]:
    try:
        files = sorted(gt_dir.glob("frame_*.json"))
        for path in files:
            try:
                idx = int(path.stem.split("_")[-1])
            except Exception:
                continue
            if idx < s or idx >= e:
                continue
            src_w, src_h, polys = base.load_labelme_polys(path)
            if polys:
                box = base.bbox_from_polys_scaled(polys, src_w, src_h, tgt_w, tgt_h, pad)
                return idx, box
    except Exception:
        return None
    return None


def _infer_frame_multi(model, processor, session, frame_idx: int, tgt_size: tuple[int, int]):
    import numpy as np
    import time
    H, W = tgt_size[1], tgt_size[0]
    start = time.perf_counter()
    output = model(inference_session=session, frame_idx=frame_idx)
    post = processor.post_process_masks(
        [output.pred_masks],
        original_sizes=[[H, W]],
        binarize=False,
    )[0]
    if hasattr(post, 'detach'):
        try:
            post = post.detach().cpu().numpy()
        except Exception:
            post = np.array(post)
    prob = None
    arr = post
    if arr is None:
        prob = np.zeros((H, W), dtype=np.float32)
    else:
        if arr.ndim == 2:
            prob = arr
        elif arr.ndim == 3:
            if arr.shape[-2:] == (H, W):
                prob = np.max(arr, axis=0)
            elif arr.shape[0:2] == (H, W):
                prob = np.max(arr, axis=2)
            else:
                prob = np.max(arr, axis=0)
        else:
            while arr.ndim > 2:
                arr = np.max(arr, axis=0)
            prob = arr
    runtime_ms = (time.perf_counter() - start) * 1000.0
    mask = base.prob_to_mask(prob, threshold=base.MASK_THRESHOLD, shape=(H, W))
    return mask, runtime_ms


def process_clip_yolo(
    clip_cfg: Dict,
    context: Dict,
    model: "base.Sam2VideoModel",
    processor: "base.Sam2VideoProcessor",
    run_dir: Path,
    git_commit: str,
    script_hash: str,
) -> Dict[str, object]:
    import cv2
    import numpy as np

    clip_id = clip_cfg["id"]
    clip_dir = run_dir / clip_id
    base.ensure_dir(clip_dir)
    log_lines, log = base.create_logger()
    _STATE["log"] = log

    data_root: Path = context["data_root"]
    clip_path = data_root / "clips" / f"{clip_id}.mp4"
    gt_dir = data_root / "gt_frames" / clip_id
    if not clip_path.exists():
        raise SystemExit(f"Clip not found: {clip_path}")

    frames, (tgt_w, tgt_h), fps = base.read_resize_frames(str(clip_path), clip_cfg["input_width"])
    _STATE["last_frames"] = frames

    clip_cfg.setdefault("reseed", {})["enabled"] = False
    clip_cfg["reseed"]["max_events"] = 0
    clip_cfg["reseed"]["triggers"] = {}
    clip_cfg.setdefault("seed", {})
    clip_cfg["seed"].update({"mode": "none", "from_gt_bbox": False, "negatives": None, "bbox_pad_px": int(clip_cfg["seed"].get("bbox_pad_px", 6))})

    device = context["device"]
    log(f"Processing {clip_id}")
    log(f"  Device: {device}")
    log(f"  Frames decoded: {len(frames)} @ {fps:.2f} fps -> {tgt_w}x{tgt_h}")

    try:
        shots = detect_shots(
            clip_path,
            total_frames=len(frames),
            fps=fps,
            method="adaptive",
            min_shot_len_s=1.0,
            adaptive_sensitivity=3,
        )
        shot_bounds = [(int(s.start), int(s.end)) for s in shots]
    except Exception:
        shot_bounds = [(0, int(len(frames)))]
    clip_cfg["shot_bounds"] = shot_bounds

    args = parse_args()
    yolo = None
    try:
        yolo = YoloBoxPromptor(model_path=args.yolo_model, device=args.yolo_device, conf=float(args.yolo_conf_thr), imgsz=1280)
    except Exception:
        yolo = None
    log(f"Autoprompt[YOLO] model={args.yolo_model} conf={args.yolo_conf_thr} device={args.yolo_device}")
    log(f"Shots detected: {len(shot_bounds)}")

    total_frames = len(frames)
    masks_full: List[np.ndarray] = [np.zeros((tgt_h, tgt_w), dtype=np.uint8) for _ in range(total_frames)]
    frame_runtimes: Dict[int, float] = {}
    stride = max(1, int(clip_cfg.get("stride", 1)))

    with base.torch.inference_mode():
        for i, (s, e) in enumerate(shot_bounds, start=1):
            s = int(max(0, min(s, total_frames - 1))) if total_frames > 0 else 0
            e = int(max(s + 1, min(e, total_frames)))
            local_len = e - s
            if local_len <= 0:
                continue

            shot_frames = frames[s:e]
            preds = []
            if yolo is not None and shot_frames:
                bgr0 = _to_bgr(frames[s])
                if bgr0 is not None:
                    try:
                        preds = yolo(bgr0, max_det=int(getattr(args, 'yolo_max_objects', 3))) or []
                    except Exception:
                        preds = []

            fb_box = None
            if (not preds) and args.autoprompt_fallback == "gt":
                pad = int(clip_cfg.get("seed", {}).get("bbox_pad_px", 6))
                fb = _first_gt_box_in_segment(gt_dir, s, e, tgt_w, tgt_h, pad)
                if fb is not None:
                    _, fb_box = fb

            if (not preds) and fb_box is None:
                log(f"Shot {i}/{len(shot_bounds)}: yolo NONE - skip - frames {s}-{e-1}")
                _STATE["events"].append({"shot_index": i, "start": s, "end": e, "mode": "fallback-skip", "score": "", "box": [], "label": ""})
                _STATE["shot_rows"].append({"shot_idx": i, "start": s, "end": e, "mode": "fallback-skip", "score": "", "box_json": "[]"})
                continue

            boxes: List[List[int]] = []
            labels: List[str] = []
            scores: List[float] = []
            if preds:
                for p in preds[: int(getattr(args, 'yolo_max_objects', 3))]:
                    try:
                        x0, y0, x1, y1 = p.as_int_tuple(width=tgt_w, height=tgt_h)
                    except Exception:
                        continue
                    boxes.append([int(x0), int(y0), int(x1), int(y1)])
                    labels.append(str(getattr(p, "label", "billboard")))
                    try:
                        scores.append(float(getattr(p, "score", 0.0)))
                    except Exception:
                        scores.append(0.0)
                mode = "yolo"
            else:
                x0, y0, x1, y1 = int(fb_box[0]), int(fb_box[1]), int(fb_box[2]), int(fb_box[3])
                boxes.append([x0, y0, x1, y1])
                labels.append("1")
                scores.append(0.0)
                mode = "fallback-gt"

            session = processor.init_video_session(
                video=shot_frames,
                inference_device=device,
                processing_device=device,
                video_storage_device=device,
                dtype=context["dtype"],
            )

            seed_boxes = [boxes]
            obj_ids = list(range(1, len(boxes) + 1))
            processor.add_inputs_to_inference_session(
                inference_session=session,
                frame_idx=0,
                obj_ids=obj_ids,
                input_boxes=seed_boxes,
                input_points=None,
                input_labels=None,
            )

            log(f"Shot {i}/{len(shot_bounds)}: {mode} ok (n={len(boxes)}) - seeding @ {s}; scores=" + ",".join([f"{x:.2f}" for x in scores]))
            try:
                _STATE["events"].append({
                    "shot_index": i,
                    "start": int(s),
                    "end": int(e),
                    "mode": mode,
                    "score": (scores[0] if scores else ""),
                    "box": (boxes[0] if boxes else []),
                    "boxes": boxes,
                    "label": (labels[0] if labels else ""),
                    "labels": labels,
                    "scores": scores,
                })
                _STATE["shot_rows"].append({
                    "shot_idx": i,
                    "start": int(s),
                    "end": int(e),
                    "mode": mode,
                    "score": (scores[0] if scores else ""),
                    "box_json": json.dumps(boxes),
                })
            except Exception:
                pass

            for local_idx in range(local_len):
                mask, runtime_ms = _infer_frame_multi(model, processor, session, local_idx, (tgt_w, tgt_h))
                global_idx = s + local_idx
                masks_full[global_idx] = mask
                frame_runtimes[global_idx] = frame_runtimes.get(global_idx, 0.0) + runtime_ms

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    overlay_path = clip_dir / f"overlay_{clip_id}_{base.RUN_SPEC['run_id']}.mp4"
    writer = cv2.VideoWriter(
        str(overlay_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (tgt_w, tgt_h),
    )
    for frame, mask in zip(frames, masks_full):
        rgb = np.array(frame)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        bgr = base.overlay_mask(rgb, cleaned)
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
        src_w, src_h, polys = base.load_labelme_polys(label_path)
        if base.RUN_SPEC["metrics"]["scoring_policy"]["skip_frames_with_empty_gt"] and not polys:
            continue
        gt_mask = base.polys_to_mask(polys, src_w, src_h, tgt_w, tgt_h)
        if base.RUN_SPEC["metrics"]["scoring_policy"]["skip_frames_with_empty_gt"] and not gt_mask.any():
            continue
        pred_mask = masks_full[frame_idx] > 0
        gt_mask_bool = gt_mask > 0
        iou = base.compute_iou(pred_mask, gt_mask_bool)
        biou = base.compute_iou(base.band_mask(pred_mask), base.band_mask(gt_mask_bool))
        area = float(pred_mask.sum())
        cx, cy = base.centroid_from_mask(pred_mask)
        is_empty = 1 if area <= base.AREA_EPS else 0
        shift_px = math.nan
        shift_norm = math.nan
        if (frame_idx in frame_runtimes) and not math.isnan(cx) and not math.isnan(prev_metric_centroid[0]):
            dx = cx - prev_metric_centroid[0]
            dy = cy - prev_metric_centroid[1]
            shift_px = math.hypot(dx, dy)
            shift_norm = (shift_px / max(1.0, tgt_w)) * 100.0
        if frame_idx in frame_runtimes and not math.isnan(cx):
            prev_metric_centroid = (cx, cy)
        elif frame_idx in frame_runtimes and math.isnan(cx):
            prev_metric_centroid = (math.nan, math.nan)
        frame_rows.append({
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
        })

    per_frame_path = clip_dir / f"per_frame_{clip_id}_{base.RUN_SPEC['run_id']}.csv"
    base.write_per_frame_csv(per_frame_path, frame_rows, [
        "clip_id","frame_no","iou","biou","area_px","centroid_x","centroid_y","centroid_shift_px","centroid_shift_norm","is_empty","runtime_ms","roi_w","roi_h",
    ])

    ious = [row["iou"] for row in frame_rows]
    biou_vals = [row["biou"] for row in frame_rows]
    jitter_vals = [val for val in (row["centroid_shift_norm"] for row in frame_rows) if isinstance(val, float) and not math.isnan(val)]
    iou_p25, iou_med, iou_p75 = base.nan_percentiles(ious, [25, 50, 75])
    biou_p25, biou_med, biou_p75 = base.nan_percentiles(biou_vals, [25, 50, 75])
    jitter_med = base.nan_percentiles(jitter_vals, [50])[0] if jitter_vals else math.nan
    empty_pct = (sum(1 for r in frame_rows if r["is_empty"] == 1) / len(frame_rows) * 100.0) if frame_rows else math.nan

    processed_count = sum(1 for k in frame_runtimes.keys())
    total_time_s = sum(frame_runtimes.values()) / 1000.0
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
        "jitter_norm_p95_pct": base.nan_percentiles(jitter_vals, [95])[0] if jitter_vals else math.nan,
        "area_cv_pct": math.nan,
        "empty_frames_pct": empty_pct,
        "proc_fps_measured": fps_measured,
        "proc_fps_theoretical": fps_theoretical,
        "roi_w": tgt_w,
        "roi_h": tgt_h,
        "target_W": tgt_w,
        "target_H": tgt_h,
        "stride": stride,
    }

    log("  Processed {} frames at ~{:.2f} fps (theoretical {:.2f})".format(processed_count, fps_measured, fps_theoretical))
    log("  Metrics: IoU_med={:.3f}, BIoU_med={:.3f}, jitter_med={:.3f}%/frame".format(
        iou_med if not math.isnan(iou_med) else float("nan"),
        biou_med if not math.isnan(biou_med) else float("nan"),
        (base.nan_percentiles(jitter_vals, [50])[0] if jitter_vals else float("nan")),
    ))

    if base.RUN_SPEC["logging"].get("write_params_json", False):
        params_path = clip_dir / f"params_{clip_id}_{base.RUN_SPEC['run_id']}.json"
        params_payload = {
            "run_id": base.RUN_SPEC["run_id"],
            "clip_id": clip_id,
            "clip_path": str(clip_path),
            "gt_dir": str(gt_dir),
            "input_width": clip_cfg["input_width"],
            "stride": stride,
            "full_frame": clip_cfg.get("full_frame", False),
            "seed": {"frame_index": 0, "json": None, "bbox_pad_px": clip_cfg["seed"].get("bbox_pad_px", 6), "box_xyxy": None, "negatives": None},
            "reseed": clip_cfg.get("reseed", {}),
            "model": {"weights": str(context["weights"]), "device": device, "dtype": str(context["dtype"])},
            "script_hash": script_hash if base.RUN_SPEC["logging"].get("write_script_hash", False) else "N/A",
            "git_commit": base.get_git_commit(Path.cwd()) if base.RUN_SPEC["logging"].get("capture_git_commit", False) else "N/A",
        }
        base.write_json(params_path, params_payload)

    re_prompt_path = clip_dir / f"re_prompts_{clip_id}_{base.RUN_SPEC['run_id']}.csv"
    base.write_reprompt_csv(re_prompt_path, [])

    log_path = clip_dir / f"pilot_{clip_id}_{base.RUN_SPEC['run_id']}.log"
    base.ensure_dir(log_path.parent)
    with log_path.open("w", encoding="utf-8") as f:
        for line in log_lines:
            f.write(f"{line}\n")

    return {"clip_id": clip_id, "summary": summary, "per_frame_path": per_frame_path, "per_frame_rows": len(frame_rows), "overlay_path": overlay_path, "re_prompts_path": re_prompt_path, "log_path": log_path, "params_path": clip_dir / f"params_{clip_id}_{base.RUN_SPEC['run_id']}.json", "reseed_events": [], "fps_measured": fps_measured, "iou_median": iou_med, "biou_median": biou_med, "jitter_med": (base.nan_percentiles(jitter_vals, [50])[0] if jitter_vals else math.nan), "empty_pct": empty_pct}


def _install_hooks_and_overrides(args) -> None:
    orig_create_logger = base.create_logger
    def create_logger_wrapper():
        lines, log = orig_create_logger()
        _STATE["log"] = log
        return lines, log
    base.create_logger = create_logger_wrapper  # type: ignore[attr-defined]

    def _lfs_no_seed(clip_cfg, clip_path, gt_dir, target_width):
        frames, (tgt_w, tgt_h), fps = base.read_resize_frames(str(clip_path), target_width)
        try:
            shots = detect_shots(clip_path, total_frames=len(frames), fps=fps, method="adaptive", min_shot_len_s=1.0, adaptive_sensitivity=3)
            clip_cfg["shot_bounds"] = [(int(s.start), int(s.end)) for s in shots]
        except Exception:
            clip_cfg["shot_bounds"] = [(0, int(len(frames)))]
        clip_cfg.setdefault("reseed", {})["enabled"] = False
        clip_cfg["reseed"]["max_events"] = 0
        clip_cfg["reseed"]["triggers"] = {}
        return frames, (tgt_w, tgt_h), fps, 0, None, None, None, None, {}
    base.load_frames_and_seed = _lfs_no_seed  # type: ignore[attr-defined]

    orig_write_reprompt_csv = base.write_reprompt_csv
    def write_reprompt_csv_wrapper(path, rows):
        out = orig_write_reprompt_csv(path, rows)
        try:
            from csv import DictWriter
            if _STATE["shot_rows"]:
                with path.open("a", newline="") as f:
                    f.write("\n")
                    writer = DictWriter(f, fieldnames=["shot_idx","start","end","mode","score","box_json"])
                    writer.writeheader()
                    for r in _STATE["shot_rows"]:
                        writer.writerow(r)
        except Exception:
            pass
        try:
            import cv2, numpy as np
            frames = _STATE.get("last_frames") or []
            for ev in _STATE.get("events", []):
                i = int(ev.get("shot_index", 0))
                s = int(ev.get("start", 0))
                if 0 <= s < len(frames):
                    arr = np.asarray(frames[s])
                    if arr.ndim == 3 and arr.shape[2] == 3:
                        bgr = cv2.cvtColor(arr.astype(np.uint8, copy=False), cv2.COLOR_RGB2BGR)
                        boxes = ev.get("boxes")
                        labels = ev.get("labels") or []
                        if isinstance(boxes, list) and boxes and isinstance(boxes[0], list):
                            colors = [(0,200,0),(0,0,255),(255,128,0),(255,0,255),(0,255,255)]
                            for j, bx in enumerate(boxes):
                                if isinstance(bx, list) and len(bx) == 4:
                                    x0,y0,x1,y1 = map(int, bx)
                                    color = colors[j % len(colors)]
                                    cv2.rectangle(bgr, (x0,y0), (x1,y1), color, 2)
                                    tag = str(labels[j]) if j < len(labels) else f"{j+1}"
                                    cv2.putText(bgr, tag, (x0, max(0, y0-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                        else:
                            bx = ev.get("box", [])
                            if isinstance(bx, list) and len(bx) == 4:
                                x0,y0,x1,y1 = map(int, bx)
                                cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 200, 0), 2)
                                lbl0 = (labels[0] if labels else ev.get("label", ""))
                                if lbl0:
                                    cv2.putText(bgr, str(lbl0), (x0, max(0, y0-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2, cv2.LINE_AA)
                        out_jpg = path.parent / f"shot_{i:03d}_seed.jpg"
                        cv2.imwrite(str(out_jpg), bgr)
        except Exception:
            pass
        return out
    base.write_reprompt_csv = write_reprompt_csv_wrapper  # type: ignore[attr-defined]

    base.process_clip = process_clip_yolo  # type: ignore[attr-defined]


if __name__ == "__main__":
    args = parse_args()
    _install_hooks_and_overrides(args)
    req = args.clips or []
    _ensure_requested_in_runspec(req, Path(args.data_root))
    base.parse_args = parse_args  # type: ignore[attr-defined]
    base.select_device = select_device  # type: ignore[attr-defined]
    base.main()




