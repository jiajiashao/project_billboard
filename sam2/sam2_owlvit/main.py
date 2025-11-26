# Standard library imports for file I/O, math operations, and type hints
import sys
import json
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Path setup block: Ensures the current directory is in Python's module search path
# This allows imports from sam2_smoke and sam2_pilot modules to resolve correctly
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import base SAM-2 implementation and specialized modules
import sam2_base as base  # Base SAM-2 tracking implementation
from shot_detection import detect_shots  # Shot boundary detection for video segmentation
from autoprompt_owlvit import OwlVitBoxPromptor, PROMPT_TERMS  # OWL-ViT vision-language model for auto-prompting


def parse_args() -> base.argparse.Namespace:
    # Command-line argument parser for SAM-2 with OWL-ViT auto-prompting
    # This variant processes video in shots and uses OWL-ViT to automatically detect objects
    parser = base.argparse.ArgumentParser(description="SAM-2 with OWL-ViT auto-prompt (per-shot sessions)")
    parser.add_argument("--data-root", dest="data_root", default="./../data")  # Root directory for input data
    parser.add_argument("--weights", default="facebook/sam2.1-hiera-tiny")  # SAM-2 model weights
    parser.add_argument("--runs-root", dest="runs_root", default="runs")  # Output directory for results
    parser.add_argument("--clips", nargs="*", help="Optional subset of clip IDs to process")  # Filter specific clips
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda")  # Computation device
    # Auto-prompt / OWL-ViT configuration
    parser.add_argument("--auto-prompt", action="store_true", default=True)  # Enable OWL-ViT auto-prompting
    parser.add_argument("--owlvit-model", default="google/owlv2-base-patch16-ensemble")  # OWL-ViT model identifier
    parser.add_argument("--owlvit-device", default=None)  # Device for OWL-ViT (None = use main device)
    parser.add_argument("--owlvit-score-thr", type=float, default=0.15)  # Confidence threshold for detections
    parser.add_argument("--prompts", type=str, default=None)  # Comma-separated prompt terms (e.g., "billboard,sign")
    parser.add_argument("--prompts-file", type=str, default=None)  # File containing prompt terms (one per line)
    parser.add_argument("--autoprompt-fallback", choices=["none", "gt"], default="none")  # Fallback to GT if auto-prompt fails
    return parser.parse_args()


def select_device(preferred: Optional[str]) -> str:
    # Device selection logic: handles user preference and fallback scenarios
    # Selects the computation device (CPU, CUDA, or MPS) based on availability
    if preferred:
        # If user specified a device, validate it's available before using it
        if preferred == "cuda" and not base.torch.cuda.is_available():
            print("Requested cuda but unavailable; falling back to cpu")
            return "cpu"
        if preferred == "mps" and not base.torch.backends.mps.is_available():
            print("Requested mps but unavailable; falling back to cpu")
            return "cpu"
        return preferred
    # Auto-detection: if no preference specified, choose best available device
    # Priority: CUDA > MPS > CPU
    if base.torch.cuda.is_available():
        return "cuda"
    return "mps" if base.torch.backends.mps.is_available() else "cpu"


# Global state dictionary for OWL-ViT processing
# Tracks shot information, events, frames, and logger across the processing pipeline
_OWL_STATE = {"shot_rows": [], "events": [], "last_frames": [], "log": None}


def _ensure_requested_in_runspec(requested: List[str], data_root: Path) -> None:
    # Ensures that all requested clip IDs are present in RUN_SPEC configuration
    # Adds clips with auto-prompt configuration (no GT seeding, reseeding disabled)
    if not requested:
        return  # No clips requested, nothing to do
    existing = {cfg.get("id") for cfg in base.RUN_SPEC.get("clips", [])}
    for cid in requested:
        if cid in existing:
            continue  # Clip already configured, skip
        # Only add if video file exists (GT not required for auto-prompt mode)
        if (data_root / "clips" / f"{cid}.mp4").exists():
            base.RUN_SPEC["clips"].append({
                "id": cid,
                "input_width": 1280,
                "stride": 1,
                "full_frame": True,
                # Auto-prompt mode: no GT seeding required
                "seed": {"mode": "none", "from_gt_bbox": False, "negatives": None, "bbox_pad_px": 6},
                # Reseeding disabled in auto-prompt mode
                "reseed": {"enabled": False, "triggers": {}, "action": "reseed_with_box_plus_neg", "cooldown_frames": 0, "max_events": 0},
            })


def _prompts_from_args(args) -> List[str]:
    # Extract prompt terms from command-line arguments or files
    # Priority: --prompts > --prompts-file > PROMPT_TERMS
    # Returns list of prompt strings to use with OWL-ViT for object detection
    if getattr(args, "prompts", None):
        # Parse comma-separated prompts from command line
        return [p.strip() for p in str(args.prompts).split(",") if p.strip()]
    if getattr(args, "prompts_file", None):
        # Read prompts from file (one per line, # for comments)
        p = Path(args.prompts_file)
        if p.exists():
            try:
                return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
            except Exception:
                pass
    # Fallback to default prompt terms from autoprompt_owlvit module
    return list(PROMPT_TERMS)


def _to_bgr(frame) -> Optional["base.np.ndarray"]:
    # Convert frame (PIL Image or NumPy array) to BGR format for OpenCV
    # Returns None if conversion fails
    try:
        import numpy as np
        import cv2
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return None  # Not a valid RGB image
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        try:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
        except Exception:
            return arr  # Return as-is if conversion fails
    except Exception:
        return None


def _first_gt_box_in_segment(gt_dir: Path, s: int, e: int, tgt_w: int, tgt_h: int, pad: int = 6) -> Optional[Tuple[int, Tuple[float, float, float, float]]]:
    # Find first labeled frame within [s, e) and return (frame_idx, xyxy) in target resolution
    # Used for GT fallback when OWL-ViT detection fails
    try:
        files = sorted(gt_dir.glob("frame_*.json"))
        for path in files:
            try:
                idx = int(path.stem.split("_")[-1])  # Extract frame number
            except Exception:
                continue
            if idx < s or idx >= e:
                continue  # Skip frames outside segment
            src_w, src_h, polys = base.load_labelme_polys(path)
            if polys:
                # Convert polygons to bounding box in target resolution
                box = base.bbox_from_polys_scaled(polys, src_w, src_h, tgt_w, tgt_h, pad)
                return idx, box
    except Exception:
        return None
    return None


def _nearest_gt_box_in_segment(gt_dir: Path, s: int, e: int, tgt_w: int, tgt_h: int, pad: int = 6) -> Optional[Tuple[int, Tuple[float, float, float, float]]]:
    # Find the labeled frame within [s, e) nearest to s; return (frame_idx, xyxy) in target resolution
    # Used for GT fallback - finds the GT box closest to the shot start
    try:
        files = sorted(gt_dir.glob("frame_*.json"))
        nearest = None
        best_dist = None
        for path in files:
            try:
                idx = int(path.stem.split("_")[-1])  # Extract frame number
            except Exception:
                continue
            if idx < s or idx >= e:
                continue  # Skip frames outside segment
            src_w, src_h, polys = base.load_labelme_polys(path)
            if polys:
                # Convert polygons to bounding box in target resolution
                box = base.bbox_from_polys_scaled(polys, src_w, src_h, tgt_w, tgt_h, pad)
                d = abs(idx - s)  # Distance from shot start
                if best_dist is None or d < best_dist:
                    best_dist = d
                    nearest = (idx, box)  # Keep track of nearest frame
        return nearest
    except Exception:
        return None


def _infer_frame_multi(model, processor, session, frame_idx: int, tgt_size: tuple[int, int]):
    # Run inference on a single frame for multi-object tracking
    # Handles multi-object masks by taking maximum across objects
    # Returns binary mask and inference runtime in milliseconds
    import numpy as np
    import time
    H, W = tgt_size[1], tgt_size[0]  # Height and width
    start = time.perf_counter()
    # Run model inference on the specified frame index
    output = model(inference_session=session, frame_idx=frame_idx)
    # Post-process the model output masks
    post = processor.post_process_masks(
        [output.pred_masks],
        original_sizes=[[H, W]],
        binarize=False,  # Keep as probability map
    )[0]
    # Move to numpy: convert PyTorch tensor to NumPy array if needed
    if hasattr(post, 'detach'):
        try:
            post = post.detach().cpu().numpy()
        except Exception:
            post = np.array(post)
    # Reduce to (H, W): handle various array shapes and dimensions
    prob = None
    arr = post
    if arr is None:
        prob = np.zeros((H, W), dtype=np.float32)  # Empty mask if None
    else:
        if arr.ndim == 2:
            prob = arr  # Already 2D
        elif arr.ndim == 3:
            # Multi-object mask: try different axis layouts
            if arr.shape[-2:] == (H, W):
                prob = np.max(arr, axis=0)  # (N, H, W) -> max across objects
            elif arr.shape[0:2] == (H, W):
                prob = np.max(arr, axis=2)  # (H, W, N) -> max across last axis
            else:
                prob = np.max(arr, axis=0)  # Fallback: max across first axis
        else:
            # Collapse all extra dimensions down to HxW by iterative max
            while arr.ndim > 2:
                arr = np.max(arr, axis=0)
            prob = arr
    runtime_ms = (time.perf_counter() - start) * 1000.0  # Measure inference time
    # Convert probability map to binary mask
    mask = base.prob_to_mask(prob, threshold=base.MASK_THRESHOLD, shape=(H, W))
    return mask, runtime_ms

def process_clip_owlvit(
    clip_cfg: Dict,
    context: Dict,
    model: "base.Sam2VideoModel",
    processor: "base.Sam2VideoProcessor",
    run_dir: Path,
    git_commit: str,
    script_hash: str,
) -> Dict[str, object]:
    # Main function to process a single video clip using OWL-ViT auto-prompting
    # Processes video in shots, uses OWL-ViT to detect objects, then tracks with SAM-2
    # Supports multi-object tracking within each shot and GT fallback option
    import cv2
    import numpy as np

    # Set up output directory and logging
    clip_id = clip_cfg["id"]
    clip_dir = run_dir / clip_id
    base.ensure_dir(clip_dir)
    log_lines, log = base.create_logger()
    _OWL_STATE["log"] = log  # Store logger in global state

    # Set up file paths
    data_root: Path = context["data_root"]
    clip_path = data_root / "clips" / f"{clip_id}.mp4"
    gt_dir = data_root / "gt_frames" / clip_id
    if not clip_path.exists():
        raise SystemExit(f"Clip not found: {clip_path}")

    # Read frames directly (no GT seeding required for auto-prompt mode)
    frames, (tgt_w, tgt_h), fps = base.read_resize_frames(str(clip_path), clip_cfg["input_width"])
    _OWL_STATE["last_frames"] = frames  # Store frames in global state

    # Disable reseed in cfg (effective for params dump)
    # Auto-prompt mode doesn't use reseeding - each shot is processed independently
    clip_cfg.setdefault("reseed", {})["enabled"] = False
    clip_cfg["reseed"]["max_events"] = 0
    clip_cfg["reseed"]["triggers"] = {}
    clip_cfg.setdefault("seed", {})
    # Configure seed mode to "none" - no GT-based seeding
    clip_cfg["seed"].update({"mode": "none", "from_gt_bbox": False, "negatives": None, "bbox_pad_px": int(clip_cfg["seed"].get("bbox_pad_px", 6))})

    # Log processing information
    device = context["device"]
    log(f"Processing {clip_id}")
    log(f"  Device: {device}")
    log(f"  Frames decoded: {len(frames)} @ {fps:.2f} fps -> {tgt_w}x{tgt_h}")

    # Shot detection: segment video into shots for independent processing
    # Each shot is processed separately with its own SAM-2 session
    try:
        shots = detect_shots(
            clip_path,
            total_frames=len(frames),
            fps=fps,
            method="adaptive",  # Adaptive shot detection method
            min_shot_len_s=1.0,  # Minimum shot length in seconds
            adaptive_sensitivity=3,  # Sensitivity parameter for adaptive detection
        )
        shot_bounds = [(int(s.start), int(s.end)) for s in shots]  # Convert to (start, end) tuples
    except Exception:
        # Fallback: treat entire video as single shot if detection fails
        shot_bounds = [(0, int(len(frames)))]
    clip_cfg["shot_bounds"] = shot_bounds  # Store shot bounds in config

    # Initialize OWL-ViT promptor for automatic object detection
    # OWL-ViT is a vision-language model that can detect objects based on text prompts
    args = parse_args()
    prompts = _prompts_from_args(args)  # Get prompt terms (e.g., "billboard", "sign")
    run_dev = args.owlvit_device or device  # Use specified device or fallback to main device
    promptor = None
    if args.auto_prompt:
        try:
            # Initialize OWL-ViT model for object detection
            promptor = OwlVitBoxPromptor(
                model_id=args.owlvit_model,
                device=run_dev,
                prompts=prompts,  # List of prompt terms to search for
                score_thr=float(args.owlvit_score_thr),  # Confidence threshold for detections
                nms_iou=0.5,  # Non-maximum suppression IoU threshold
            )
        except Exception:
            promptor = None  # Continue without auto-prompting if initialization fails
    log(f"Autoprompt[OWL-ViT] model={args.owlvit_model} score_thr={args.owlvit_score_thr} device={run_dev}")
    log(f"Prompts used: {', '.join(prompts)}")
    log(f"Shots detected: {len(shot_bounds)}")

    # Initialize tracking state
    total_frames = len(frames)
    masks_full: List[np.ndarray] = [np.zeros((tgt_h, tgt_w), dtype=np.uint8) for _ in range(total_frames)]  # Store masks for all frames
    frame_runtimes: Dict[int, float] = {}  # Track inference time per frame
    stride = max(1, int(clip_cfg.get("stride", 1)))  # Frame stride for metrics computation
    target_indices = set()  # Frames to compute metrics for

    # Per-shot inference: process each shot independently with its own SAM-2 session
    # This allows handling scene changes and different objects in different shots
    with base.torch.inference_mode():  # Disable gradient computation for inference
        for i, (s, e) in enumerate(shot_bounds, start=1):
            # Clamp shot boundaries to valid frame range
            s = int(max(0, min(s, total_frames - 1))) if total_frames > 0 else 0
            e = int(max(s + 1, min(e, total_frames)))
            local_len = e - s
            if local_len <= 0:
                continue  # Skip empty shots
            # Build session frames slice for this shot
            shot_frames = frames[s:e]

            # Run OWL-ViT on global start frame (multi-object detection)
            # OWL-ViT detects objects in the first frame of each shot
            preds = []
            if promptor is not None and shot_frames:
                bgr0 = _to_bgr(frames[s])  # Convert first frame to BGR
                if bgr0 is not None:
                    try:
                        # Get top-k detections (up to 3 objects)
                        preds = promptor.predict_topk(bgr0, k=3) or []
                    except Exception:
                        preds = []

            # Filter detections by confidence threshold
            # Keep only boxes with score >= threshold (default 0.15)
            kept = []
            for p in preds:
                try:
                    sc = float(getattr(p, "score", 0.0))  # Get confidence score
                except Exception:
                    sc = 0.0
                if sc >= float(args.owlvit_score_thr):
                    kept.append(p)  # Keep if above threshold
            preds = kept

            # Fallback to GT if no qualified OWL-ViT detections
            # This allows graceful degradation when auto-prompting fails
            fb_box = None
            if (not preds) and (getattr(args, 'autoprompt_fallback','none') == 'gt'):
                pad = int(clip_cfg.get("seed", {}).get("bbox_pad_px", 6))
                # Find nearest GT box in this shot segment
                fb = _nearest_gt_box_in_segment(gt_dir, s, e, tgt_w, tgt_h, pad)
                if fb is not None:
                    _, fb_box = fb  # Extract bounding box

            # Skip shot if no detections and no GT fallback
            if (not preds) and fb_box is None:
                log(f"Shot {i}/{len(shot_bounds)}: owlvit NONE - skip - frames {s}-{e-1}")
                _OWL_STATE["events"].append({"shot_index": i, "start": s, "end": e, "mode": "fallback-skip", "score": "", "box": [], "label": ""})
                _OWL_STATE["shot_rows"].append({"shot_idx": i, "start": s, "end": e, "mode": "fallback-skip", "score": "", "box_json": "[]"})
                continue

            # Build seed boxes from OWL-ViT detections or GT fallback (one or many objects)
            # Extract bounding boxes, labels, and confidence scores
            boxes = []
            labels = []
            scores = []
            if preds:
                # Use OWL-ViT detections
                for p in preds[:3]:  # Process up to 3 detections
                    try:
                        x0, y0, x1, y1 = p.as_int_tuple(width=tgt_w, height=tgt_h)  # Get bounding box coordinates
                    except Exception:
                        continue  # Skip invalid predictions
                    boxes.append([int(x0), int(y0), int(x1), int(y1)])
                    labels.append(str(getattr(p, "label", "")))  # Object label (e.g., "billboard")
                    try:
                        scores.append(float(getattr(p, "score", 0.0)))  # Confidence score
                    except Exception:
                        scores.append(0.0)
                mode = "owlvit"  # Track that we used OWL-ViT
            else:
                # Use GT fallback box
                x0, y0, x1, y1 = int(fb_box[0]), int(fb_box[1]), int(fb_box[2]), int(fb_box[3])
                boxes.append([x0, y0, x1, y1])
                labels.append("1")  # Generic label for GT fallback
                scores.append(0.0)  # No score for GT
                mode = "fallback-gt"  # Track that we used GT fallback

            # Initialize SAM-2 video session for this shot
            # Each shot gets its own session for independent tracking
            session = processor.init_video_session(
                video=shot_frames,
                inference_device=device,
                processing_device=device,
                video_storage_device=device,
                dtype=context["dtype"],
            )

            # Seed exactly once at local frame 0 with multi-box union
            # All detected objects are tracked simultaneously in this shot
            seed_boxes = [boxes]  # Format: [[[x0,y0,x1,y1], ...]] for each object
            obj_ids = list(range(1, len(boxes)+1))  # Assign unique IDs to each object
            processor.add_inputs_to_inference_session(
                inference_session=session,
                frame_idx=0,  # Seed at first frame of shot (local index)
                obj_ids=obj_ids,  # IDs for each object
                input_boxes=seed_boxes,  # Bounding boxes for seeding
                input_points=None,  # No point prompts
                input_labels=None,  # No point labels
            )

            # Logs and CSV/jpg bookkeeping: record shot information for later analysis
            if mode == "owlvit":
                log(f"Shot {i}/{len(shot_bounds)}: owlvit ok (n={len(boxes)}) - seeding @ {s}; scores=" + ",".join([f"{x:.2f}" for x in scores]))
            else:
                log(f"Shot {i}/{len(shot_bounds)}: {mode} ok (n={len(boxes)}) - seeding @ {s}; scores=" + ",".join([f"{x:.2f}" for x in scores]))
            try:
                log(f"GD seed set at shot {i} start, local frame 0 -> {boxes}")
            except Exception:
                pass  # Continue if logging fails
            try:
                import json as _json
                # Store shot event information in global state
                _OWL_STATE["events"].append({
                    "shot_index": i,
                    "start": int(s),
                    "end": int(e),
                    "mode": mode,  # "owlvit" or "fallback-gt"
                    "score": (scores[0] if scores else ""),
                    "box": (boxes[0] if boxes else []),
                    "boxes": boxes,  # All boxes for multi-object
                    "label": (labels[0] if labels else ""),
                    "labels": labels,  # All labels
                    "scores": scores,  # All scores
                })
                # Store shot row for CSV export
                _OWL_STATE["shot_rows"].append({
                    "shot_idx": i,
                    "start": int(s),
                    "end": int(e),
                    "mode": mode,
                    "score": (scores[0] if scores else ""),
                    "box_json": _json.dumps(boxes),  # Boxes as JSON string
                })
            except Exception:
                pass  # Continue if event recording fails

            # Per-frame inference within this shot: track objects through all frames in the shot
            # local_idx = local index (within shot), global_idx = global index (within entire video)
            for local_idx in range(local_len):
                mask, runtime_ms = _infer_frame_multi(model, processor, session, local_idx, (tgt_w, tgt_h))
                global_idx = s + local_idx  # Convert local index to global index
                masks_full[global_idx] = mask  # Store mask at global frame index
                # Track runtime for frames that match stride
                if (global_idx - s) % stride == 0:
                    target_indices.add(global_idx)
                    frame_runtimes[global_idx] = frame_runtimes.get(global_idx, 0.0) + runtime_ms

    # Generate overlay video: visualize tracking masks overlaid on original frames
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    overlay_path = clip_dir / f"overlay_{clip_id}_{base.RUN_SPEC['run_id']}.mp4"
    writer = cv2.VideoWriter(
        str(overlay_path),
        cv2.VideoWriter_fourcc(*"mp4v"),  # MPEG-4 codec
        fps,
        (tgt_w, tgt_h),
    )
    # Process each frame: clean mask and overlay on original video
    for frame, mask in zip(frames, masks_full):
        rgb = np.array(frame)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)  # Remove small noise
        bgr = base.overlay_mask(rgb, cleaned)  # Overlay mask on frame
        writer.write(bgr)
    writer.release()
    # Validate that video was written successfully
    if overlay_path.exists() and overlay_path.stat().st_size == 0:
        log(f"Overlay failed to write: {overlay_path}")
        log("If codecs are missing, try: brew install ffmpeg")
        raise SystemExit(1)

    # Metrics per labeled frame: compare predictions with ground truth annotations
    # This allows evaluation even though auto-prompt mode doesn't use GT for seeding
    label_paths = sorted(gt_dir.glob("frame_*.json"))
    frame_rows: List[Dict[str, object]] = []
    prev_metric_centroid = (math.nan, math.nan)  # Track centroid for jitter calculation
    for label_path in label_paths:
        frame_idx = int(label_path.stem.split("_")[-1])
        if frame_idx >= len(masks_full):
            continue  # Skip frames outside valid range
        src_w, src_h, polys = base.load_labelme_polys(label_path)
        # Skip frames with empty ground truth if configured
        if base.RUN_SPEC["metrics"]["scoring_policy"]["skip_frames_with_empty_gt"] and not polys:
            continue
        gt_mask = base.polys_to_mask(polys, src_w, src_h, tgt_w, tgt_h)
        if base.RUN_SPEC["metrics"]["scoring_policy"]["skip_frames_with_empty_gt"] and not gt_mask.any():
            continue

        # Compare predicted mask with ground truth mask
        pred_mask = masks_full[frame_idx] > 0
        gt_mask_bool = gt_mask > 0

        # Compute IoU (Intersection over Union) and BIoU (Boundary IoU) metrics
        iou = base.compute_iou(pred_mask, gt_mask_bool)
        biou = base.compute_iou(base.band_mask(pred_mask), base.band_mask(gt_mask_bool))
        area = float(pred_mask.sum())  # Predicted mask area
        cx, cy = base.centroid_from_mask(pred_mask)  # Predicted centroid
        is_empty = 1 if area <= base.AREA_EPS else 0

        # Calculate jitter: frame-to-frame centroid movement (indicates tracking stability)
        shift_px = math.nan
        shift_norm = math.nan
        if frame_idx in frame_runtimes and not math.isnan(cx) and not math.isnan(prev_metric_centroid[0]):
            dx = cx - prev_metric_centroid[0]
            dy = cy - prev_metric_centroid[1]
            shift_px = math.hypot(dx, dy)  # Euclidean distance
            shift_norm = (shift_px / max(1.0, tgt_w)) * 100.0  # Normalized to frame width percentage
        # Update previous centroid for next iteration
        if frame_idx in frame_runtimes and not math.isnan(cx):
            prev_metric_centroid = (cx, cy)
        elif frame_idx in frame_runtimes and math.isnan(cx):
            prev_metric_centroid = (math.nan, math.nan)

        # Store per-frame metrics
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

    # Write per-frame metrics to CSV file for detailed analysis
    per_frame_path = clip_dir / f"per_frame_{clip_id}_{base.RUN_SPEC['run_id']}.csv"
    base.write_per_frame_csv(per_frame_path, frame_rows, [
        "clip_id","frame_no","iou","biou","area_px","centroid_x","centroid_y","centroid_shift_px","centroid_shift_norm","is_empty","runtime_ms","roi_w","roi_h",
    ])

    # Compute summary statistics across all frames
    ious = [row["iou"] for row in frame_rows]
    biou_vals = [row["biou"] for row in frame_rows]
    jitter_vals = [val for val in (row["centroid_shift_norm"] for row in frame_rows) if isinstance(val, float) and not math.isnan(val)]
    area_vals = [row["area_px"] for row in frame_rows]
    empty_vals = [row["is_empty"] for row in frame_rows]

    # Calculate percentiles for IoU, BIoU, and jitter metrics
    iou_p25, iou_med, iou_p75 = base.nan_percentiles(ious, [25, 50, 75])
    biou_p25, biou_med, biou_p75 = base.nan_percentiles(biou_vals, [25, 50, 75])
    jitter_med, jitter_p95 = base.nan_percentiles(jitter_vals, [50, 95]) if jitter_vals else (math.nan, math.nan)

    # Calculate area statistics: mean, standard deviation, and coefficient of variation
    area_mean = base.nan_mean(area_vals)
    if not math.isnan(area_mean):
        arr = np.array(area_vals, dtype=np.float32)
        area_std = float(np.std(arr))
    else:
        area_std = math.nan
    area_cv = math.nan if math.isnan(area_mean) or area_mean == 0 else (area_std / area_mean) * 100.0  # Coefficient of variation
    empty_pct = (sum(empty_vals) / len(empty_vals) * 100.0) if empty_vals else math.nan  # Percentage of empty frames

    # Calculate processing performance metrics
    processed_count = len(target_indices)
    total_time_s = sum(frame_runtimes.get(idx, 0.0) for idx in target_indices) / 1000.0
    fps_measured = processed_count / total_time_s if total_time_s > 0 else 0.0  # Actual processing FPS
    fps_theoretical = fps / stride  # Theoretical FPS based on video FPS and stride

    # Create summary dictionary with aggregated metrics for this clip
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

    # Log summary statistics to console
    log("  Processed {} frames at ~{:.2f} fps (theoretical {:.2f})".format(processed_count, fps_measured, fps_theoretical))
    log("  Metrics: IoU_med={:.3f}, BIoU_med={:.3f}, jitter_med={:.3f}%/frame".format(
        iou_med if not math.isnan(iou_med) else float("nan"),
        biou_med if not math.isnan(biou_med) else float("nan"),
        jitter_med if not math.isnan(jitter_med) else float("nan"),
    ))

    # Write parameters JSON (seed fields are neutralized since we use auto-prompting)
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
                "frame_index": 0,  # Neutralized - not used in auto-prompt mode
                "json": None,
                "bbox_pad_px": clip_cfg["seed"].get("bbox_pad_px", 6),
                "box_xyxy": None,  # No GT-based seeding
                "negatives": None,
            },
            "reseed": clip_cfg.get("reseed", {}),
            "model": {"weights": str(context["weights"]), "device": device, "dtype": str(context["dtype"])},
            "script_hash": script_hash if base.RUN_SPEC["logging"].get("write_script_hash", False) else "N/A",
            "git_commit": base.get_git_commit(Path.cwd()) if base.RUN_SPEC["logging"].get("capture_git_commit", False) else "N/A",
        }
        base.write_json(params_path, params_payload)

    # Write reseed events CSV (empty in auto-prompt mode, but wrapper will append shot rows)
    re_prompt_path = clip_dir / f"re_prompts_{clip_id}_{base.RUN_SPEC['run_id']}.csv"
    base.write_reprompt_csv(re_prompt_path, [])

    # Write log file with all processing messages
    log_path = clip_dir / f"pilot_{clip_id}_{base.RUN_SPEC['run_id']}.log"
    base.ensure_dir(log_path.parent)
    if base.RUN_SPEC["logging"].get("echo_config_to_log", False):
        log("  Config snapshot: ")
        cfg_snapshot = {
            "clip": clip_cfg,
            "seed_frame": 0,  # Neutralized
            "seed_box": [],  # No GT-based seeding
            "device": device,
            "dtype": str(context["dtype"]),
            "weights": str(context["weights"]),
        }
        log(json.dumps(cfg_snapshot, indent=2))
    with log_path.open("w", encoding="utf-8") as f:
        for line in log_lines:
            f.write(f"{line}\n")

    # Return comprehensive results dictionary with all outputs and metrics
    return {
        "clip_id": clip_id,
        "summary": summary,
        "per_frame_path": per_frame_path,
        "per_frame_rows": len(frame_rows),
        "overlay_path": overlay_path,
        "re_prompts_path": re_prompt_path,
        "log_path": log_path,
        "params_path": clip_dir / f"params_{clip_id}_{base.RUN_SPEC['run_id']}.json",
        "reseed_events": [],  # Empty in auto-prompt mode
        "fps_measured": fps_measured,
        "iou_median": iou_med,
        "biou_median": biou_med,
        "jitter_med": jitter_med,
        "empty_pct": empty_pct,
    }


def _install_hooks_and_overrides(args) -> None:
    # Install hooks and overrides to customize base SAM-2 behavior for OWL-ViT auto-prompting
    # This function monkey-patches base functions to enable per-shot processing and auto-prompting
    
    # Capture logger reference: wrap create_logger to store logger in global state
    orig_create_logger = base.create_logger
    def create_logger_wrapper():
        lines, log = orig_create_logger()
        _OWL_STATE["log"] = log  # Store logger for use in other functions
        return lines, log
    base.create_logger = create_logger_wrapper  # type: ignore[attr-defined]

    # Neutralize GT seed in loader (in case other utilities call it)
    # Replace load_frames_and_seed to skip GT-based seeding and add shot detection
    def _lfs_no_seed(clip_cfg, clip_path, gt_dir, target_width):
        frames, (tgt_w, tgt_h), fps = base.read_resize_frames(str(clip_path), target_width)
        # Attach shot bounds here as well
        try:
            shots = detect_shots(clip_path, total_frames=len(frames), fps=fps, method="adaptive", min_shot_len_s=1.0, adaptive_sensitivity=3)
            clip_cfg["shot_bounds"] = [(int(s.start), int(s.end)) for s in shots]
        except Exception:
            clip_cfg["shot_bounds"] = [(0, int(len(frames)))]  # Fallback: single shot
        # Disable reseeding in auto-prompt mode
        clip_cfg.setdefault("reseed", {})["enabled"] = False
        clip_cfg["reseed"]["max_events"] = 0
        clip_cfg["reseed"]["triggers"] = {}
        # Return seed-less tuple (no GT seeding)
        return frames, (tgt_w, tgt_h), fps, 0, None, None, None, None, {}
    base.load_frames_and_seed = _lfs_no_seed  # type: ignore[attr-defined]

    # Append our per-shot rows and JPGs when base writes re_prompts_*.csv
    # This adds shot information to the reseed events CSV file and generates visualization images
    orig_write_reprompt_csv = base.write_reprompt_csv
    def write_reprompt_csv_wrapper(path, rows):
        out = orig_write_reprompt_csv(path, rows)
        # Append shot information table
        try:
            from csv import DictWriter
            if _OWL_STATE["shot_rows"]:
                with path.open("a", newline="") as f:
                    f.write("\n")  # Add separator
                    writer = DictWriter(f, fieldnames=["shot_idx","start","end","mode","score","box_json"])
                    writer.writeheader()
                    for r in _OWL_STATE["shot_rows"]:
                        writer.writerow(r)  # Write each shot row
        except Exception:
            pass  # Continue if appending fails
        # Save per-shot annotated JPGs, drawing multiple boxes if present
        # This creates visualization images showing detected bounding boxes
        try:
            import cv2, numpy as np, json as _json
            frames = _OWL_STATE.get("last_frames") or []
            for ev in _OWL_STATE.get("events", []):
                i = int(ev.get("shot_index", 0))
                s = int(ev.get("start", 0))
                if 0 <= s < len(frames):
                    arr = np.asarray(frames[s])
                    if arr.ndim == 3 and arr.shape[2] == 3:
                        bgr = cv2.cvtColor(arr.astype(np.uint8, copy=False), cv2.COLOR_RGB2BGR)
                        boxes = ev.get("boxes")  # Get all boxes for multi-object
                        labels = ev.get("labels") or []
                        # Draw multiple boxes with different colors
                        if isinstance(boxes, list) and boxes and isinstance(boxes[0], list):
                            colors = [(0,200,0),(0,0,255),(255,128,0),(255,0,255),(0,255,255)]  # Green, Red, Orange, Magenta, Cyan
                            for j, bx in enumerate(boxes):
                                if isinstance(bx, list) and len(bx) == 4:
                                    x0,y0,x1,y1 = map(int, bx)
                                    color = colors[j % len(colors)]  # Cycle through colors
                                    cv2.rectangle(bgr, (x0,y0), (x1,y1), color, 2)  # Draw rectangle
                                    tag = str(labels[j]) if j < len(labels) else f"{j+1}"  # Label or number
                                    cv2.putText(bgr, tag, (x0, max(0, y0-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                        else:
                            # Single box fallback
                            bx = ev.get("box", [])
                            if isinstance(bx, list) and len(bx) == 4:
                                x0,y0,x1,y1 = map(int, bx)
                                cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 200, 0), 2)
                                lbl0 = (labels[0] if labels else ev.get("label", ""))
                                if lbl0:
                                    cv2.putText(bgr, str(lbl0), (x0, max(0, y0-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2, cv2.LINE_AA)
                        out_jpg = path.parent / f"shot_{i:03d}_seed.jpg"
                        cv2.imwrite(str(out_jpg), bgr)  # Save visualization
        except Exception:
            pass  # Continue if visualization fails
        return out
    base.write_reprompt_csv = write_reprompt_csv_wrapper  # type: ignore[attr-defined]

    # Replace the processing function with per-shot session variant
    # This is the main override that enables OWL-ViT auto-prompting
    base.process_clip = process_clip_owlvit  # v3 multi-object  # type: ignore[attr-defined]


if __name__ == "__main__":
    # Main execution block: sets up OWL-ViT auto-prompting and runs SAM-2 tracking
    args = parse_args()
    
    # Step 1: Install hooks and overrides before base.main()
    # This replaces base functions with OWL-ViT-aware versions
    _install_hooks_and_overrides(args)
    
    # Step 2: Ensure RUN_SPEC has a clips key
    if "clips" not in base.RUN_SPEC:
        base.RUN_SPEC["clips"] = []

    # Step 3: Extend RUN_SPEC with any requested clip IDs not preconfigured
    # Adds clips with auto-prompt configuration (no GT seeding required)
    req = args.clips or []
    _ensure_requested_in_runspec(req, Path(args.data_root))

    # Step 4: Monkey-patch CLI and device selection functions
    # Replace base module's functions with our custom versions
    # This allows customization without modifying base code
    base.parse_args = parse_args  # type: ignore[attr-defined]
    base.select_device = select_device  # type: ignore[attr-defined]
    
    # Step 5: Call the main function from base module to start processing
    # This executes the SAM-2 tracking pipeline with OWL-ViT auto-prompting
    # Each video is processed in shots, with OWL-ViT detecting objects in each shot
    # Supports GT fallback if auto-prompting fails (when --autoprompt-fallback=gt)
    base.main()



















