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
from autoprompt_moondream import MoondreamBoxPromptor, PROMPT_TERMS


def parse_args() -> base.argparse.Namespace:
    parser = base.argparse.ArgumentParser(description="SAM-2 with Moondream auto-prompt (per-shot sessions, multi-object)")
    parser.add_argument("--data-root", dest="data_root", default="D:\Billboard_Project - Copy\sam2\data")
    parser.add_argument("--weights", default="D:\Billboard_Project - Copy\sam2\models\sam2.1-hiera-tiny")
    parser.add_argument("--runs-root", dest="runs_root", default="runs")
    parser.add_argument("--clips", nargs="*", help="Optional subset of clip IDs to process")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    # Auto-prompt / Moondream
    parser.add_argument("--auto-prompt", action="store_true", default=False)
    parser.add_argument("--moondream-model", default="vikhyatk/moondream2")
    parser.add_argument("--moondream-device", default=None)
    parser.add_argument("--moondream-threshold", type=float, default=0.10)
    parser.add_argument("--prompts", type=str, default=None)
    parser.add_argument("--prompts-file", type=str, default=None)
    parser.add_argument("--autoprompt-fallback", choices=["none"], default="none")
    parser.add_argument("--reseed-cooldown", type=int, default=3)
    parser.add_argument("--probe-first-k", type=int, default=5)
    parser.add_argument("--reseed-topk", type=int, default=1, help="Max boxes per dynamic reseed")
    parser.add_argument("--empty-streak", type=int, default=3, help="Consecutive empty frames to trigger reseed")
    parser.add_argument("--empty-area-px", type=int, default=50, help="Mask area (px) treated as empty")
    parser.add_argument("--save-reseed-jpg", action="store_true", default=True, help="Save JPGs for reseed events")
    parser.add_argument("--no-reseed-jpg", action="store_true", default=False, help="Disable reseed JPG saving")
    parser.add_argument("--input-width", type=int, default=None, help="Override clip input width")
    parser.add_argument("--stride", type=int, default=None, help="Override per-frame stride")
    parser.add_argument("--sam2-dtype", choices=["float32","float16","bfloat16"], default=None, help="Override model dtype")
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


_MD_STATE: Dict[str, object] = {"shot_rows": [], "events": [], "last_frames": [], "log": None}


def _ensure_requested_in_runspec(requested: List[str], data_root: Path) -> None:
    if not requested:
        return
    existing = {cfg.get("id") for cfg in base.RUN_SPEC.get("clips", [])}
    for cid in requested:
        if cid in existing:
            continue
        if (data_root / "clips" / f"{cid}.mp4").exists():
            base.RUN_SPEC.setdefault("clips", []).append({
                "id": cid,
                "input_width": 1280,
                "stride": 1,
                "full_frame": True,
                "seed": {"mode": "none", "from_gt_bbox": False, "negatives": None, "bbox_pad_px": 6},
                "reseed": {"enabled": False, "triggers": {}, "action": "reseed_with_box_plus_neg", "cooldown_frames": 0, "max_events": 0},
            })


def _prompts_from_args(args) -> List[str]:
    if getattr(args, "prompts", None):
        parts = [p.strip() for p in str(args.prompts).replace(";", ",").split(",")]
        return [p for p in parts if p]
    if getattr(args, "prompts_file", None):
        p = Path(args.prompts_file)
        if p.exists():
            try:
                return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
            except Exception:
                pass
    return list(PROMPT_TERMS)


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
    # Move to numpy
    if hasattr(post, 'detach'):
        try:
            post = post.detach().cpu().numpy()
        except Exception:
            post = np.array(post)
    # Robust collapse to (H, W)
    arr = post
    if arr is None:
        prob = np.zeros((H, W), dtype=np.float32)
    else:
        if isinstance(arr, np.ndarray):
            if arr.ndim == 2:
                prob = arr
            elif arr.ndim == 3:
                # Try (N,H,W) or (H,W,N)
                if arr.shape[-2:] == (H, W):
                    prob = np.max(arr, axis=0)
                elif arr.shape[0:2] == (H, W):
                    prob = np.max(arr, axis=2)
                else:
                    # Fallback: max across first axis
                    prob = np.max(arr, axis=0)
            else:
                # Collapse all extra dims down to HxW by iterative max
                while arr.ndim > 2:
                    arr = np.max(arr, axis=0)
                prob = arr
        else:
            # Unknown type -> zeros
            prob = np.zeros((H, W), dtype=np.float32)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    mask = base.prob_to_mask(prob, threshold=base.MASK_THRESHOLD, shape=(H, W))
    return mask, runtime_ms

def process_clip_moondream(
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
    args = (_MD_STATE.get("args") if isinstance(_MD_STATE, dict) else None) or parse_args()

    clip_id = clip_cfg["id"]
    clip_dir = run_dir / clip_id
    base.ensure_dir(clip_dir)
    log_lines, log = base.create_logger()
    _MD_STATE["log"] = log

    data_root: Path = context["data_root"]
    clip_path = data_root / "clips" / f"{clip_id}.mp4"
    gt_dir = data_root / "gt_frames" / clip_id
    if not clip_path.exists():
        raise SystemExit(f"Clip not found: {clip_path}")

    # Read frames directly (no GT seeding)
    # eff_input_width set after args retrieval
    args = (_MD_STATE.get("args") if isinstance(_MD_STATE, dict) else None) or parse_args()
    eff_input_width = int(args.input_width) if getattr(args, "input_width", None) else int(clip_cfg["input_width"])
    frames, (tgt_w, tgt_h), fps = base.read_resize_frames(str(clip_path), eff_input_width)
    _MD_STATE["last_frames"] = frames

    # Disable reseed in cfg (effective for params dump)
    clip_cfg.setdefault("reseed", {})["enabled"] = False
    clip_cfg["reseed"]["max_events"] = 100
    clip_cfg["reseed"]["triggers"] = {}
    clip_cfg.setdefault("seed", {})
    clip_cfg["seed"].update({"mode": "none", "from_gt_bbox": False, "negatives": None, "bbox_pad_px": int(clip_cfg["seed"].get("bbox_pad_px", 6))})

    device = context["device"]
    log(f"Processing {clip_id}")
    # Optional dtype override
    try:
        dt_map = {"float32": base.torch.float32, "float16": base.torch.float16, "bfloat16": base.torch.bfloat16}
        if getattr(args, "sam2_dtype", None):
            target_dt = dt_map.get(str(args.sam2_dtype))
            if target_dt is not None:
                try:
                    model.to(device=device, dtype=target_dt)
                    context["dtype"] = target_dt
                    log(f"  Dtype override: {args.sam2_dtype}")
                except Exception:
                    log(f"  Dtype override failed; keeping default")
    except Exception:
        pass
    log(f"  Device: {device}")
    log(f"  Frames decoded: {len(frames)} @ {fps:.2f} fps -> {tgt_w}x{tgt_h}")

    # Shots
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

    # Moondream promptor
    # args already set above
    prompts = _prompts_from_args(args)
    run_dev = args.moondream_device or device
    promptor = None
    if args.auto_prompt:
        try:
            promptor = MoondreamBoxPromptor(
                model_id=args.moondream_model,
                device=run_dev,
                threshold=float(args.moondream_threshold),
                prompts=prompts,
            )
        except Exception:
            promptor = None
    log(f"Autoprompt[Moondream] model={args.moondream_model} thr={args.moondream_threshold} device={run_dev}")
    log(f"Prompts used: {', '.join(prompts)}")
    log(f"Shots detected: {len(shot_bounds)}")
    cooldown_frames = max(0, int(getattr(args, "reseed_cooldown", 0)))
    empty_thr_px = max(1, int(getattr(args, "empty_area_px", 50)))
    reseed_topk = max(1, int(getattr(args, "reseed_topk", 1)))
    empty_streak_req = max(1, int(getattr(args, "empty_streak", 3)))
    save_reseed_jpg = bool(getattr(args, "save_reseed_jpg", True) and not getattr(args, "no_reseed_jpg", False))
    empty_thr_px = max(1, int(getattr(args, "empty_area_px", 1)))
    probe_first_k = max(1, int(getattr(args, "probe_first_k", 1)))
    cooldown_frames = max(0, int(getattr(args, "reseed_cooldown", 0)))

    total_frames = len(frames)
    masks_full: List[np.ndarray] = [np.zeros((tgt_h, tgt_w), dtype=np.uint8) for _ in range(total_frames)]
    frame_runtimes: Dict[int, float] = {}
    stride = max(1, int(args.stride)) if getattr(args, "stride", None) else max(1, int(clip_cfg.get("stride", 1)))
    target_indices = set()

    # Per-shot inference
    with base.torch.inference_mode():
        for i, (s, e) in enumerate(shot_bounds, start=1):
            s = int(max(0, min(s, total_frames - 1))) if total_frames > 0 else 0
            e = int(max(s + 1, min(e, total_frames)))
            local_len = e - s
            if local_len <= 0:
                continue
            # Build session frames slice
            shot_frames = frames[s:e]

            # Run Moondream on first K frames of the shot (multi-object)
            preds = []
            if promptor is not None and shot_frames:
                max_probe = min(int(probe_first_k), local_len)
                for off in range(max_probe):
                    bgr0 = _to_bgr(frames[s + off])
                    if bgr0 is None:
                        continue
                    try:
                        preds = promptor.predict_topk(bgr0, k=3) if hasattr(promptor, "predict_topk") else ([promptor(bgr0)] if promptor(bgr0) is not None else [])
                    except Exception:
                        preds = []
                    if preds:
                        break

            if not preds:
                boxes = []
                labels = []
                scores = []
                n_seed_objs = 0
                log(f"Shot {i}/{len(shot_bounds)}: moondream NONE - dynamic reseed enabled - frames {s}-{e-1}")

            # Build seed boxes (one or many)
            boxes: List[List[int]] = []
            labels: List[str] = []
            scores: List[float] = []
            for p in preds[:3]:
                try:
                    x0, y0, x1, y1 = p.as_int_tuple()
                except Exception:
                    continue
                boxes.append([int(x0), int(y0), int(x1), int(y1)])
                labels.append(str(getattr(p, "label", "")))
                try:
                    scores.append(float(getattr(p, "score", 0.0)))
                except Exception:
                    scores.append(0.0)

            n_seed_objs = len(boxes)
            session = processor.init_video_session(
                video=shot_frames,
                inference_device=device,
                processing_device=device,
                video_storage_device=device,
                dtype=context["dtype"],
            )

            # Seed once at frame 0 when boxes exist; otherwise rely on dynamic reseed
            n_seed_objs = len(boxes)
            if boxes:
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

            # Logs and per-shot JPG
            if boxes:
                log(f"Shot {i}/{len(shot_bounds)}: moondream ok (n={len(boxes)}) - seeding @ {s}; scores=" + ",".join([f"{x:.2f}" for x in scores]))
                try:
                    bgr = _to_bgr(frames[s])
                    if bgr is not None:
                        colors = [(0,200,0),(0,0,255),(255,128,0),(255,0,255),(0,255,255)]
                        for j, bx in enumerate(boxes):
                            x0,y0,x1,y1 = map(int, bx)
                            color = colors[j % len(colors)]
                            cv2.rectangle(bgr, (x0,y0), (x1,y1), color, 2)
                        tag = ",".join([f"{x:.2f}" for x in scores])
                        cv2.putText(bgr, tag, (boxes[0][0], max(0, boxes[0][1]-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
                        out_jpg = clip_dir / f"shot_{i:03d}_seed.jpg"
                        cv2.imwrite(str(out_jpg), bgr)
                except Exception:
                    pass
            else:
                log(f"Shot {i}/{len(shot_bounds)}: moondream NONE - dynamic reseed enabled - frames {s}-{e-1}")

            # Inference across this shot
            cooldown_remaining = 0
            for li in range(local_len):
                if cooldown_remaining > 0:
                    cooldown_remaining -= 1
                gi = s + li
                                # Ensure at least one object exists before inference; try on-demand reseed when none
                did_initial_reseed = False
                if ("n_seed_objs" in locals()) and int(n_seed_objs) == 0 and promptor is not None:
                    bgr_cur0 = _to_bgr(frames[gi])
                    reseed0 = []
                    if bgr_cur0 is not None:
                        try:
                            reseed0 = promptor.predict_topk(bgr_cur0, k=int(reseed_topk)) if hasattr(promptor, "predict_topk") else ([promptor(bgr_cur0)] if promptor(bgr_cur0) is not None else [])
                        except Exception:
                            reseed0 = []
                    boxes_re0 = []
                    for p0 in (reseed0[:3] if isinstance(reseed0, list) else []):
                        try:
                            x0,y0,x1,y1 = p0.as_int_tuple()
                            boxes_re0.append([int(x0),int(y0),int(x1),int(y1)])
                        except Exception:
                            pass
                    if boxes_re0:
                        target_n0 = len(boxes_re0)
                        obj_ids_re0 = list(range(1, target_n0+1))
                        try:
                            processor.add_inputs_to_inference_session(
                                inference_session=session,
                                frame_idx=int(li),
                                obj_ids=obj_ids_re0,
                                input_boxes=[boxes_re0],
                                input_points=None,
                                input_labels=None,
                            )
                            n_seed_objs = int(target_n0)
                            did_initial_reseed = True
                            # log explicit reseed
                            try:
                                _log = _MD_STATE.get("log")
                                if callable(_log):
                                    _log(f"Reseed@{gi} (shot {i}, local {li}) due to no initial objects; n={len(boxes_re0)}")
                            except Exception:
                                pass
                            if cooldown_frames > 0:
                                cooldown_remaining = int(cooldown_frames)
                                try:
                                    _log = _MD_STATE.get("log")
                                    if callable(_log):
                                        _log(f"    cooldown set to {cooldown_frames} frames")
                                except Exception:
                                    pass
                            # Save annotated JPG for this reseed (no-initial-objects)
                            try:
                                import cv2 as _cv2, numpy as _np
                                arr = _np.asarray(frames[gi])
                                if arr.ndim == 3 and arr.shape[2] == 3:
                                    bgr = _cv2.cvtColor(arr.astype(_np.uint8, copy=False), _cv2.COLOR_RGB2BGR)
                                    if save_reseed_jpg:
                                        colors = [(0,200,0),(0,0,255),(255,128,0),(255,0,255),(0,255,255)]
                                        for jj, bx in enumerate(boxes_re0):
                                            x0,y0,x1,y1 = map(int, bx)
                                            color = colors[jj % len(colors)]
                                            _cv2.rectangle(bgr, (x0,y0), (x1,y1), color, 2)
                                        tag = ",".join([f"{getattr(p, 'score', 0.0):.2f}" for p in (reseed0[:3] if isinstance(reseed0, list) else [])])
                                        _cv2.putText(bgr, tag, (boxes_re0[0][0], max(0, boxes_re0[0][1]-6)), _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, _cv2.LINE_AA)
                                        out_jpg = clip_dir / f"shot_{i:03d}_reseed_{gi:06d}_init.jpg"
                                        _cv2.imwrite(str(out_jpg), bgr)
                            except Exception:
                                pass
                        except Exception:
                            pass

                if ("n_seed_objs" in locals()) and int(n_seed_objs) == 0:
                    # Still no objects; produce empty mask without calling the model
                    import numpy as _np
                    mask = _np.zeros((tgt_h, tgt_w), dtype=_np.uint8)
                    runtime_ms = 0.0
                else:
                    mask, runtime_ms = _infer_frame_multi(model, processor, session, li, (tgt_w, tgt_h))
                masks_full[gi] = mask
                if (gi - (0)) % stride == 0:
                    frame_runtimes[gi] = frame_runtimes.get(gi, 0.0) + runtime_ms

                # Dynamic reseed: empty mask trigger using Moondream
                try:
                    import numpy as _np
                except Exception:
                    _np = None
                try:
                    area_count = float((mask > 0).sum())
                except Exception:
                    area_count = 0.0
                # Update empty streak
                try:
                    empty_streak = empty_streak + 1 if area_count <= float(empty_thr_px) else 0
                except NameError:
                    empty_streak = 1 if area_count <= float(empty_thr_px) else 0
                if empty_streak >= int(empty_streak_req) and cooldown_remaining == 0:
                    bgr_cur = _to_bgr(frames[gi])
                elif area_count <= float(empty_thr_px) and cooldown_remaining > 0:
                    try:
                        _log = _MD_STATE.get("log")
                        if callable(_log):
                            _log(f"    empty mask but cooldown active ({cooldown_remaining} left) — skip reseed")
                    except Exception:
                        pass
                    reseed_preds = []
                    if bgr_cur is not None:
                        try:
                            reseed_preds = promptor.predict_topk(bgr_cur, k=int(reseed_topk)) if hasattr(promptor, "predict_topk") else ([promptor(bgr_cur)] if promptor(bgr_cur) is not None else [])
                        except Exception:
                            reseed_preds = []
                    if reseed_preds:
                        boxes_re = []
                        for p in reseed_preds[:3]:
                            try:
                                x0,y0,x1,y1 = p.as_int_tuple()
                                boxes_re.append([int(x0),int(y0),int(x1),int(y1)])
                            except Exception:
                                pass
                        if boxes_re:
                            if ("n_seed_objs" in locals() and int(n_seed_objs) == 0):
                                n_seed_objs = int(len(boxes_re))
                            try:
                                obj_ids_re = list(range(1, len(boxes_re)+1))
                                processor.add_inputs_to_inference_session(
                                    inference_session=session,
                                    frame_idx=int(li),
                                    obj_ids=obj_ids_re,
                                    input_boxes=[boxes_re],
                                    input_points=None,
                                    input_labels=None,
                                )
                                mask2, runtime_ms2 = _infer_frame_multi(model, processor, session, li, (tgt_w, tgt_h))
                                masks_full[gi] = mask2
                                frame_runtimes[gi] = frame_runtimes.get(gi, 0.0) + runtime_ms2
                                cooldown_remaining = int(cooldown_frames)
                                empty_streak = 0
                                try:
                                    log = _MD_STATE.get("log")
                                    if callable(log):
                                        log(f"    cooldown set to {cooldown_frames} frames")
                                except Exception:
                                    pass
                                try:
                                    _MD_STATE["events"].append({"shot_index": i, "start": s, "end": e, "mode": "reseed-empty", "frame": int(gi), "boxes": boxes_re})
                                except Exception:
                                    pass
                            except Exception:
                                pass
                            # Save annotated JPG for this reseed (empty-mask)
                            try:
                                if save_reseed_jpg:
                                    import cv2 as _cv2, numpy as _np
                                    arr = _np.asarray(frames[gi])
                                    if arr.ndim == 3 and arr.shape[2] == 3:
                                        bgr = _cv2.cvtColor(arr.astype(_np.uint8, copy=False), _cv2.COLOR_RGB2BGR)
                                        colors = [(0,200,0),(0,0,255),(255,128,0),(255,0,255),(0,255,255)]
                                        for jj, bx in enumerate(boxes_re):
                                            x0,y0,x1,y1 = map(int, bx)
                                            color = colors[jj % len(colors)]
                                            _cv2.rectangle(bgr, (x0,y0), (x1,y1), color, 2)
                                        tag = ",".join([f"{getattr(p, 'score', 0.0):.2f}" for p in (reseed_preds[:3] if isinstance(reseed_preds, list) else [])])
                                        _cv2.putText(bgr, tag, (boxes_re[0][0], max(0, boxes_re[0][1]-6)), _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, _cv2.LINE_AA)
                                        out_jpg = clip_dir / f"shot_{i:03d}_reseed_{gi:06d}.jpg"
                                        _cv2.imwrite(str(out_jpg), bgr)
                            except Exception:
                                pass
            # Record shot row event
            try:
                from json import dumps as _dumps
                _MD_STATE["shot_rows"].append({
                    "shot_idx": i,
                    "start": s,
                    "end": e,
                    "mode": "moondream",
                    "score": ",".join([f"{x:.2f}" for x in scores]),
                    "box_json": _dumps(boxes),
                })
                _MD_STATE["events"].append({"shot_index": i, "start": s, "end": e, "mode": "moondream", "scores": scores, "boxes": boxes, "labels": labels})
            except Exception:
                pass

    # Overlay
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    overlay_path = clip_dir / f"overlay_{clip_id}_{base.RUN_SPEC['run_id']}.mp4"
    writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (tgt_w, tgt_h))
    for frame, mask in zip(frames, masks_full):
        rgb = np.array(frame)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        bgr = base.overlay_mask(rgb, cleaned)
        writer.write(bgr)
    writer.release()
    if overlay_path.exists() and overlay_path.stat().st_size == 0:
        log(f"Overlay failed to write: {overlay_path}")
        raise SystemExit(1)

    # Metrics per labeled frame
    from statistics import median
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

    per_frame_path = clip_dir / f"per_frame_{clip_id}_{base.RUN_SPEC['run_id']}.csv"
    base.write_per_frame_csv(per_frame_path, frame_rows, ["clip_id","frame_no","iou","biou","area_px","centroid_x","centroid_y","centroid_shift_px","centroid_shift_norm","is_empty","runtime_ms","roi_w","roi_h"])

    # Summary metrics
    import numpy as _np
    def _nanp(vals):
        arr = _np.asarray(vals, dtype=float)
        return arr[~_np.isnan(arr)]
    ious = _nanp([r["iou"] for r in frame_rows])
    biou = _nanp([r["biou"] for r in frame_rows])
    shifts = _nanp([r["centroid_shift_norm"] for r in frame_rows])
    empty_pct = 100.0 * (sum(1 for r in frame_rows if int(r["is_empty"]) == 1) / float(max(1, len(frame_rows))))
    iou_med = float(_np.median(ious)) if ious.size else float("nan")
    biou_med = float(_np.median(biou)) if biou.size else float("nan")
    jitter_med = float(_np.median(shifts)) if shifts.size else float("nan")
    fps_measured = float(len(frame_runtimes)) / (sum(frame_runtimes.values()) / 1000.0) if frame_runtimes else float("nan")

    summary = {
        "clip_id": clip_id,
        "iou_median": iou_med,
        "biou_median": biou_med,
        "jitter_norm_median_pct": jitter_med,
        "empty_frames_pct": float(empty_pct),
        "proc_fps_measured": fps_measured,
        "proc_fps_theoretical": float(fps),
        "roi_w": tgt_w,
        "roi_h": tgt_h,
        "target_W": tgt_w,
        "target_H": tgt_h,
        "stride": stride,
    }

    # Write summary CSV
    fields = [
        "clip_id",
        "iou_median","biou_median","jitter_norm_median_pct","empty_frames_pct",
        "proc_fps_measured","proc_fps_theoretical","roi_w","roi_h","target_W","target_H","stride",
    ]
    summary_path = clip_dir / f"summary_{base.RUN_SPEC['run_id']}.csv"
    with summary_path.open("w", newline="") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(summary)

    # Params JSON (seed fields are neutralized)
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
            "seed": {
                "frame_index": 0,
                "json": None,
                "bbox_pad_px": clip_cfg["seed"].get("bbox_pad_px", 6),
                "box_xyxy": None,
                "negatives": None,
            },
            "reseed": clip_cfg.get("reseed", {}),
            "model": {"weights": str(context["weights"]), "device": device, "dtype": str(context["dtype"])},
            "script_hash": script_hash if base.RUN_SPEC["logging"].get("write_script_hash", False) else "N/A",
            "git_commit": base.get_git_commit(Path.cwd()) if base.RUN_SPEC["logging"].get("capture_git_commit", False) else "N/A",
        }
        base.write_json(params_path, params_payload)

    # reseed events CSV (append our rows via wrapper)
    re_prompt_path = clip_dir / f"re_prompts_{clip_id}_{base.RUN_SPEC['run_id']}.csv"
    base.write_reprompt_csv(re_prompt_path, [])

    log_path = clip_dir / f"pilot_{clip_id}_{base.RUN_SPEC['run_id']}.log"
    base.ensure_dir(log_path.parent)
    if base.RUN_SPEC["logging"].get("echo_config_to_log", False):
        log("  Config snapshot: ")
        cfg_snapshot = {
            "clip": clip_cfg,
            "seed_frame": 0,
            "seed_box": [],
            "device": device,
            "dtype": str(context["dtype"]),
            "weights": str(context["weights"]),
        }
        log(json.dumps(cfg_snapshot, indent=2))
    with log_path.open("w", encoding="utf-8") as f:
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
        "params_path": clip_dir / f"params_{clip_id}_{base.RUN_SPEC['run_id']}.json",
        "reseed_events": [],
        "fps_measured": fps_measured,
        "iou_median": iou_med,
        "biou_median": biou_med,
        "jitter_med": jitter_med,
        "empty_pct": empty_pct,
    }


def _install_hooks_and_overrides(args) -> None:
    _MD_STATE["args"] = args
    # Capture logger reference
    orig_create_logger = base.create_logger
    def create_logger_wrapper():
        lines, log = orig_create_logger()
        _MD_STATE["log"] = log
        return lines, log
    base.create_logger = create_logger_wrapper  # type: ignore[attr-defined]

    # Neutralize GT seed in loader (and attach shot bounds)
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

    # Append our per-shot rows and JPGs when base writes re_prompts_*.csv
    orig_write_reprompt_csv = base.write_reprompt_csv
    def write_reprompt_csv_wrapper(path, rows):
        out = orig_write_reprompt_csv(path, rows)
        # Append our small table
        try:
            from csv import DictWriter
            if _MD_STATE["shot_rows"]:
                with path.open("a", newline="") as f:
                    f.write("\n")
                    writer = DictWriter(f, fieldnames=["shot_idx","start","end","mode","score","box_json"])
                    writer.writeheader()
                    for r in _MD_STATE["shot_rows"]:
                        writer.writerow(r)
        except Exception:
            pass
        # Append reseed events with image filenames
        try:
            evs = _MD_STATE.get("events") or []
            if evs:
                from csv import DictWriter as _DW2
                with path.open("a", newline="") as f2:
                    f2.write("\n")
                    writer2 = _DW2(f2, fieldnames=["shot_idx","frame","mode","image","boxes_json"])
                    writer2.writeheader()
                    import json as _json
                    for ev in evs:
                        try:
                            writer2.writerow({
                                "shot_idx": ev.get("shot_index",""),
                                "frame": ev.get("frame",""),
                                "mode": ev.get("mode",""),
                                "image": ev.get("image",""),
                                "boxes_json": _json.dumps(ev.get("boxes", [])),
                            })
                        except Exception:
                            pass
        except Exception:
            pass
        return out
    base.write_reprompt_csv = write_reprompt_csv_wrapper  # type: ignore[attr-defined]

    # Replace the processing function with per-shot session variant
    base.process_clip = process_clip_moondream  # type: ignore[attr-defined]


if __name__ == "__main__":
    args = parse_args()
    # Ensure pass/fail thresholds exist to avoid KeyError in write_run_notes
    try:
        _m = base.RUN_SPEC.setdefault("metrics", {})
        _m.setdefault("pass_fail", {
            "gentle": {"iou_median_min": 0.0, "jitter_norm_median_pct_max": 1e9},
            "fast": {"iou_median_min": 0.0},
        })
    except Exception:
        pass

    # Install overrides before base.main()
    _install_hooks_and_overrides(args)

    # Extend RUN_SPEC with any requested clip IDs not preconfigured
    req = args.clips or []
    _ensure_requested_in_runspec(req, Path(args.data_root))

    # Monkey-patch only the CLI and device selection; leave everything else identical
    base.parse_args = parse_args  # type: ignore[attr-defined]
    base.select_device = select_device  # type: ignore[attr-defined]
    base.main()









