# Standard library imports for argument parsing, file I/O, and utilities
import argparse
import csv
import json
import math
import datetime as dt
from pathlib import Path
from typing import List, Dict, Tuple

# Third-party imports for image processing and deep learning
import cv2  # OpenCV for image/video operations
import numpy as np  # NumPy for numerical array operations
from torch import cuda  # PyTorch CUDA utilities

# Reuse SAM-2 utilities from the project
import sam2_base as base  # Base SAM-2 tracking implementation and utilities
from shot_detection import detect_shots  # Shot boundary detection for video segmentation


def parse_args() -> argparse.Namespace:
    # Command-line argument parser for YOLO + SAM-2 billboard segmentation
    # This script uses YOLO for object detection and SAM-2 for video tracking
    p = argparse.ArgumentParser(description="Segment billboards: YOLO seeding + SAM-2 propagation (v4-like outputs)")
    # Input/Output configuration
    p.add_argument("--data", required=True, help="Video file or directory (recurses for common video types)")  # Input video(s)
    p.add_argument("--out-dir", default="runs/yolo_sam2", help="Output root; per-clip folder gets timestamp suffix")  # Output directory
    # YOLO model configuration
    p.add_argument("--yolo-model",default="./weights/best.pt", help="Path to YOLO model weights")  # YOLO model file path
    p.add_argument("--yolo-conf", type=float, default=0.20, help="YOLO confidence threshold")  # Minimum confidence for detections
    p.add_argument("--yolo-max-objects", type=int, default=3, help="Max YOLO boxes per shot start")  # Maximum objects to track per shot
    # SAM-2 model configuration
    p.add_argument("--sam2-weights", default="facebook/sam2.1-hiera-tiny", help="HuggingFace weights for SAM-2 video")  # SAM-2 model identifier
    p.add_argument("--input-width", type=int, default=1280, help="Target decode width (frames resized keeping aspect)")  # Target frame width
    # Device and performance settings
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda", help="Run device for both models")  # Computation device
    p.add_argument("--stride", type=int, default=1, help="Metrics sampling stride (visual only)")  # Frame stride for metrics
    # Shot detection configuration
    p.add_argument("--shot-mode", choices=["auto", "single"], default="auto", help="Detect shots or treat whole video as one shot")  # Shot detection mode
    # Ground truth configuration
    p.add_argument("--gt-root", default="..\sam2\data\gt_frames")  # Ground truth directory for metrics computation
    return p.parse_args()


def ensure_dir(p: Path) -> None:
    # Create directory and all parent directories if they don't exist
    # Safe to call multiple times (won't raise error if directory exists)
    p.mkdir(parents=True, exist_ok=True)


def list_sources(src: Path) -> List[Path]:
    # Find all video files from input path
    # If input is a file, return it; if directory, recursively find all video files
    if src.is_file():
        return [src]  # Single file
    # Common video file extensions to search for
    video_patterns = ["*.mp4", "*.mkv", "*.mov", "*.avi", "*.m4v", "*.mpg", "*.mpeg", "*.wmv", "*.ts", "*.webm"]
    out: List[Path] = []
    # Recursively search for all video files matching patterns
    for pat in video_patterns:
        out.extend(src.rglob(pat))
    return sorted(out)  # Return sorted list of video files


def bgr_from_frame(frame) -> np.ndarray:
    # Convert frame (PIL Image or NumPy array) to BGR format for OpenCV/YOLO
    # YOLO expects BGR format, not RGB
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[2] == 3:
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        try:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
        except Exception:
            return arr  # Return as-is if conversion fails
    raise ValueError("unexpected frame array shape")


def yolo_boxes(model, frame_bgr: np.ndarray, conf: float, imgsz: int, device: str, max_det: int) -> Tuple[List[List[int]], List[float]]:
    # Run YOLO object detection on a single frame
    # Returns bounding boxes and confidence scores, sorted by confidence (highest first)
    # boxes format: [[x0, y0, x1, y1], ...] in pixel coordinates
    res = model.predict(source=frame_bgr, imgsz=imgsz, conf=conf, device=device, verbose=False, max_det=max_det)
    boxes: List[List[int]] = []
    scores: List[float] = []
    if not res:
        return boxes, scores  # No detections
    
    r = res[0]  # Get first (and typically only) result
    battr = getattr(r, "boxes", None)
    if battr is None or getattr(battr, "xyxy", None) is None:
        return boxes, scores  # No boxes in result
    
    # Extract bounding box coordinates and confidence scores
    xyxy = battr.xyxy.detach().cpu().numpy()  # Bounding boxes in (x0, y0, x1, y1) format
    confs = battr.conf.detach().cpu().numpy() if getattr(battr, "conf", None) is not None else np.zeros((xyxy.shape[0],), dtype=np.float32)
    # Sort by confidence (descending) and take top max_det
    order = np.argsort(-confs)  # Negative for descending order
    for i in order[: int(max(1, max_det))]:
        x0, y0, x1, y1 = [int(round(float(v))) for v in xyxy[i].tolist()]
        boxes.append([x0, y0, x1, y1])
        scores.append(float(confs[i]))
    return boxes, scores


def process_video(path: Path, args: argparse.Namespace, out_root: Path) -> Dict[str, object]:
    # Main function to process a single video using YOLO detection + SAM-2 tracking
    # Processes video in shots, uses YOLO to detect objects in each shot, then tracks with SAM-2
    
    # Configure computation device and data type
    dev = base.select_device(args.device)
    dtype = base.torch.float32

    # Load models: YOLO for object detection and SAM-2 for video tracking
    from ultralytics import YOLO  # type: ignore
    yolo = YOLO(args.yolo_model)  # Load YOLO model from weights file
    sam2_model = base.Sam2VideoModel.from_pretrained(args.sam2_weights).to(device=dev, dtype=dtype)  # Load SAM-2 model
    sam2_model.eval()  # Set to evaluation mode
    sam2_proc = base.Sam2VideoProcessor.from_pretrained(args.sam2_weights)  # Load SAM-2 processor

    # Decode and resize video frames
    frames, (tgt_w, tgt_h), fps = base.read_resize_frames(str(path), args.input_width)

    # Robust SAM-2 inference helper for multi-object outputs
    # Handles multi-object masks by taking maximum across objects
    import time as _t
    def _infer_frame_multi(frame_idx: int):
        # Run inference on a single frame and return binary mask with runtime
        H, W = tgt_h, tgt_w
        start = _t.perf_counter()
        output = sam2_model(inference_session=session, frame_idx=frame_idx)
        # Post-process the model output masks
        post = sam2_proc.post_process_masks([output.pred_masks], original_sizes=[[H, W]], binarize=False)[0]
        # Move to numpy: convert PyTorch tensor to NumPy array if needed
        if hasattr(post, "detach"):
            try:
                post = post.detach().cpu().numpy()
            except Exception:
                post = np.array(post)
        # Reduce to (H, W): handle various array shapes and dimensions
        arr = post
        if arr is None:
            prob = np.zeros((H, W), dtype=np.float32)  # Empty mask if None
        else:
            if arr.ndim == 2:
                prob = arr  # Already 2D
            elif arr.ndim == 3:
                # Multi-object mask: try different axis layouts
                if arr.shape[0] != H and arr.shape[-2:] == (H, W):
                    prob = np.max(arr, axis=0)  # (N, H, W) -> max across objects
                elif arr.shape[0:2] == (H, W):
                    prob = np.max(arr, axis=2)  # (H, W, N) -> max across last axis
                else:
                    # Collapse all extra dimensions down to HxW by iterative max
                    while arr.ndim > 2:
                        arr = np.max(arr, axis=0)
                    prob = arr
            else:
                # Collapse all extra dimensions down to HxW by iterative max
                while arr.ndim > 2:
                    arr = np.max(arr, axis=0)
                prob = arr
        runtime_ms = (_t.perf_counter() - start) * 1000.0  # Measure inference time
        # Convert probability map to binary mask
        mask = base.prob_to_mask(prob, threshold=base.MASK_THRESHOLD, shape=(H, W))
        return mask, runtime_ms

    # Shot detection: segment video into shots for independent processing
    # Each shot is processed separately with its own SAM-2 session
    if args.shot_mode == "auto":
        try:
            shots = detect_shots(path, total_frames=len(frames), fps=fps, method="adaptive", min_shot_len_s=1.0, adaptive_sensitivity=3)
            shot_bounds = [(int(s.start), int(s.end)) for s in shots]  # Convert to (start, end) tuples
        except Exception:
            shot_bounds = [(0, int(len(frames)))]  # Fallback: treat entire video as single shot
    else:
        shot_bounds = [(0, int(len(frames)))]  # Single shot mode: process entire video as one shot

    # Set up output directory with timestamp to avoid overwrites
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    clip_dir = out_root / f"{path.stem}_{ts}"
    ensure_dir(clip_dir)
    overlay_path = clip_dir / f"overlay_{path.stem}_sam2.mp4"
    writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (tgt_w, tgt_h))

    # Initialize tracking state and logging
    masks_full: List[np.ndarray] = [np.zeros((tgt_h, tgt_w), dtype=np.uint8) for _ in range(len(frames))]  # Store masks for all frames
    frame_runtimes: Dict[int, float] = {}  # Track inference time per frame
    stride = max(1, int(args.stride))  # Frame stride for metrics computation
    log_lines: List[str] = []  # Store log messages for file output
    def _log(m: str) -> None:
        print(m)  # Print to console
        log_lines.append(m)  # Store in list for later writing to file
    re_prompt_rows: List[Dict[str, object]] = []  # Track shot information for CSV export

    # Log processing information
    _log(f"Processing {path}")
    _log(f"  Device: {dev}")
    _log(f"  Frames decoded: {len(frames)} @ {fps:.2f} fps -> {tgt_w}x{tgt_h}")
    _log(f"Autoprompt[YOLO] model={args.yolo_model} conf={args.yolo_conf}")
    _log(f"Shots detected: {len(shot_bounds)}")

    # Per-shot inference: process each shot independently with its own SAM-2 session
    # This allows handling scene changes and different objects in different shots
    with base.torch.inference_mode():  # Disable gradient computation for inference
        for i, (s, e) in enumerate(shot_bounds, start=1):
            # Clamp shot boundaries to valid frame range
            s = int(max(0, min(s, len(frames) - 1)))
            e = int(max(s + 1, min(e, len(frames))))
            if e - s <= 0:
                continue  # Skip empty shots
            
            # Run YOLO on shot start frame to detect objects
            # YOLO detects objects in the first frame of each shot
            try:
                bgr0 = bgr_from_frame(frames[s])  # Convert first frame to BGR for YOLO
            except Exception:
                bgr0 = None
            boxes, scores = ([], [])
            if bgr0 is not None:
                # Run YOLO detection with confidence threshold and max objects limit
                boxes, scores = yolo_boxes(yolo, bgr0, conf=args.yolo_conf, imgsz=args.input_width, device=args.device or dev, max_det=args.yolo_max_objects)
            
            # Skip shot if no objects detected
            if not boxes:
                _log(f"Shot {i}/{len(shot_bounds)}: YOLO NONE - skip frames {s}-{e-1}")
                re_prompt_rows.append({"shot_idx": i, "start": int(s), "end": int(e), "mode": "yolo-none", "score": "", "box_json": "[]"})
                continue
            
            # Record shot information for CSV export
            re_prompt_rows.append({"shot_idx": i, "start": int(s), "end": int(e), "mode": "yolo", "score": (f"{scores[0]:.3f}" if scores else ""), "box_json": json.dumps(boxes)})
            
            # Initialize SAM-2 video session for this shot
            # Each shot gets its own session for independent tracking
            session = sam2_proc.init_video_session(video=frames[s:e], inference_device=dev, processing_device=dev, video_storage_device=dev, dtype=dtype)
            # Seed SAM-2 with YOLO detections: all detected objects tracked simultaneously
            seed_boxes = [boxes]  # Format: [[[x0,y0,x1,y1], ...]] for each object
            obj_ids = list(range(1, len(boxes) + 1))  # Assign unique IDs to each object
            sam2_proc.add_inputs_to_inference_session(inference_session=session, frame_idx=0, obj_ids=obj_ids, input_boxes=seed_boxes, input_points=None, input_labels=None)
            
            # Save YOLO seed annotated JPG in clip folder for visualization
            # Draws bounding boxes and confidence scores on the seed frame
            try:
                img_seed = bgr0.copy() if bgr0 is not None else None
                if img_seed is not None:
                    for (x0,y0,x1,y1), sc in zip(boxes, scores):
                        cv2.rectangle(img_seed, (int(x0),int(y0)), (int(x1),int(y1)), (0,200,0), 2)  # Green rectangle
                        tag = f"billboard {sc:.2f}"  # Label with confidence score
                        # Draw text with black outline and white fill for visibility
                        cv2.putText(img_seed, tag, (int(x0), max(0, int(y0)-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)  # Black outline
                        cv2.putText(img_seed, tag, (int(x0), max(0, int(y0)-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)  # White fill
                    cv2.imwrite(str(clip_dir / f"shot_{i:03d}_seed.jpg"), img_seed)  # Save visualization
            except Exception:
                pass  # Continue if visualization fails
            
            _log("Seeded {} boxes at shot {} with scores=".format(len(boxes), s) + ",".join([f"{x:.2f}" for x in scores]))
            
            # Per-frame segmentation: track objects through all frames in the shot
            # off = offset within shot (local index), s + off = global frame index
            for off in range(0, e - s):
                mask, runtime_ms = _infer_frame_multi(off)  # Run inference at local frame index
                masks_full[s + off] = mask  # Store mask at global frame index
                # Track runtime for frames that match stride
                if ((s + off) - s) % stride == 0:
                    frame_runtimes[s + off] = frame_runtimes.get(s + off, 0.0) + runtime_ms

    # Generate overlay video and compute metrics if ground truth is available
    # Processes frames sequentially: overlay mask on frame, then compute metrics if GT exists
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gt_dir = Path(args.gt_root) / path.stem if args.gt_root else None  # GT directory (optional)

    frame_rows: List[Dict[str, object]] = []
    prev_metric_centroid = (math.nan, math.nan)  # Track centroid for jitter calculation
    for idx, (frame, mask) in enumerate(zip(frames, masks_full)):
        # Generate overlay: clean mask and overlay on original frame
        rgb = np.array(frame)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)  # Remove small noise
        bgr = base.overlay_mask(rgb, cleaned)  # Overlay mask on frame
        writer.write(bgr)  # Write to overlay video
        
        # Compute metrics if ground truth is available
        if gt_dir is not None:
            gt_json = gt_dir / f"frame_{idx:06d}.json"  # Look for GT annotation file
            if gt_json.exists():
                try:
                    src_w, src_h, polys = base.load_labelme_polys(gt_json)
                except (KeyError, ValueError, Exception):
                    continue  # Skip if GT file is invalid
                if polys:
                    # Convert GT polygons to mask and compare with prediction
                    gt_mask = base.polys_to_mask(polys, src_w, src_h, tgt_w, tgt_h)
                    pred_mask = cleaned > 0
                    gt_mask_bool = gt_mask > 0
                    
                    # Compute IoU (Intersection over Union) and BIoU (Boundary IoU) metrics
                    iou = base.compute_iou(pred_mask, gt_mask_bool)
                    biou = base.compute_iou(base.band_mask(pred_mask), base.band_mask(gt_mask_bool))
                    area = float(pred_mask.sum())  # Predicted mask area
                    cx, cy = base.centroid_from_mask(pred_mask)  # Predicted centroid
                    
                    # Calculate jitter: frame-to-frame centroid movement (indicates tracking stability)
                    shift_px = math.nan
                    shift_norm = math.nan
                    if idx in frame_runtimes and not math.isnan(cx) and not math.isnan(prev_metric_centroid[0]):
                        dx = cx - prev_metric_centroid[0]
                        dy = cy - prev_metric_centroid[1]
                        shift_px = (dx * dx + dy * dy) ** 0.5  # Euclidean distance
                        shift_norm = (shift_px / max(1.0, tgt_w)) * 100.0  # Normalized to frame width percentage
                    # Update previous centroid for next iteration
                    if idx in frame_runtimes:
                        prev_metric_centroid = (cx, cy) if not math.isnan(cx) else (math.nan, math.nan)
                    
                    # Store per-frame metrics
                    frame_rows.append({
                        "clip_id": path.stem,
                        "frame_no": idx,
                        "iou": float(iou),
                        "biou": float(biou),
                        "area_px": float(area),
                        "centroid_x": cx,
                        "centroid_y": cy,
                        "centroid_shift_px": shift_px,
                        "centroid_shift_norm": shift_norm,
                        "is_empty": int(area <= base.AREA_EPS),
                        "runtime_ms": frame_runtimes.get(idx, math.nan),
                        "roi_w": tgt_w,
                        "roi_h": tgt_h,
                    })
    writer.release()  # Close overlay video file

    # Write per-frame metrics to CSV file for detailed analysis
    per_frame_path = clip_dir / f"per_frame_{path.stem}_sam2.csv"
    if frame_rows:
        fields = ["clip_id","frame_no","iou","biou","area_px","centroid_x","centroid_y","centroid_shift_px","centroid_shift_norm","is_empty","runtime_ms","roi_w","roi_h"]
        with per_frame_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in frame_rows:
                w.writerow(r)

    # Summary metrics: compute aggregated statistics across all frames
    def _nan_percentiles(arr: List[float], qs: List[int]):
        # Calculate percentiles while ignoring NaN values
        # Returns tuple of percentiles corresponding to quantiles in qs
        xs = [x for x in arr if isinstance(x, float) and not math.isnan(x)]  # Filter out NaN
        if not xs:
            return tuple([math.nan for _ in qs])  # All NaN if no valid data
        xs = sorted(xs)  # Sort for percentile calculation
        out = []
        for q in qs:
            k = int(round((q / 100.0) * (len(xs) - 1)))  # Index for q-th percentile
            out.append(float(xs[k]))
        return tuple(out)

    # Extract metric values and calculate percentiles
    ious = [r["iou"] for r in frame_rows] if frame_rows else []
    biou_vals = [r["biou"] for r in frame_rows] if frame_rows else []
    iou_p25, iou_med, iou_p75 = _nan_percentiles(ious, [25, 50, 75]) if frame_rows else (math.nan, math.nan, math.nan)
    biou_p25, biou_med, biou_p75 = _nan_percentiles(biou_vals, [25, 50, 75]) if frame_rows else (math.nan, math.nan, math.nan)

    # Write log file with all processing messages
    log_path = clip_dir / f"pilot_{path.stem}_sam2.log"
    with log_path.open("w", encoding="utf-8") as f:
        for line in log_lines:
            f.write(line + "\n")

    # Write parameters JSON file with run configuration
    params_path = clip_dir / f"params_{path.stem}_sam2.json"
    params_payload = {
        "clip_id": path.stem,
        "clip_path": str(path),
        "input_width": args.input_width,
        "device": base.select_device(args.device),
        "yolo": {"model": args.yolo_model, "conf": args.yolo_conf, "max_objects": args.yolo_max_objects},  # YOLO configuration
        "sam2": {"weights": args.sam2_weights},  # SAM-2 configuration
        "shots": shot_bounds,  # Shot boundaries
    }
    with params_path.open("w") as f:
        json.dump(params_payload, f, indent=2)

    # Write re-prompt events CSV: records shot information and YOLO detections
    re_csv = clip_dir / f"re_prompts_{path.stem}_sam2.csv"
    if re_prompt_rows:
        with re_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["shot_idx","start","end","mode","score","box_json"])
            w.writeheader()
            for r in re_prompt_rows:
                w.writerow(r)

    # Write summary CSV (fix2-like format for consistency)
    summary_path = clip_dir / "summary_fix2.csv"
    
    # Calculate jitter percentiles: frame-to-frame centroid movement statistics
    jitter_list = [r["centroid_shift_norm"] for r in frame_rows if isinstance(r.get("centroid_shift_norm"), float) and not math.isnan(r["centroid_shift_norm"])]
    def _np_percentile(vals, q):
        # Calculate percentile using NumPy
        if not vals:
            return math.nan
        arr = np.array(vals, dtype=np.float32)
        try:
            return float(np.percentile(arr, q))
        except Exception:
            return math.nan
    jitter_med = _np_percentile(jitter_list, 50)  # Median jitter
    jitter_p95 = _np_percentile(jitter_list, 95)  # 95th percentile jitter
    
    # Calculate area coefficient of variation: measures consistency of mask size
    area_vals = [r["area_px"] for r in frame_rows] if frame_rows else []
    area_cv_pct = math.nan
    if area_vals:
        arr_area = np.array(area_vals, dtype=np.float32)
        m = float(np.mean(arr_area))
        if m > 0:
            area_cv_pct = float(np.std(arr_area) / m * 100.0)  # Coefficient of variation as percentage
    
    # Calculate percentage of empty frames
    empty_frames_pct = (sum(1 for r in frame_rows if r["is_empty"] == 1) / len(frame_rows) * 100.0) if frame_rows else math.nan
    
    # Calculate processing performance metrics
    processed_count = len(frame_runtimes)
    total_time_s = sum(frame_runtimes.values()) / 1000.0 if frame_runtimes else 0.0
    proc_fps_measured = processed_count / total_time_s if total_time_s > 0 else 0.0  # Actual processing FPS
    proc_fps_theoretical = fps / stride  # Theoretical FPS based on video FPS and stride
    
    # Write summary CSV with fix2 fields for consistency with other scripts
    sum_fields = [
        "clip_id","iou_median","iou_p25","iou_p75","biou_median","biou_p25","biou_p75",
        "jitter_norm_median_pct","jitter_norm_p95_pct","area_cv_pct","empty_frames_pct",
        "proc_fps_measured","proc_fps_theoretical","roi_w","roi_h","target_W","target_H","stride"
    ]
    with summary_path.open("w", newline="") as fsum:
        wsum = csv.DictWriter(fsum, fieldnames=sum_fields)
        wsum.writeheader()
        wsum.writerow({
            "clip_id": path.stem,
            "iou_median": iou_med,
            "iou_p25": iou_p25,
            "iou_p75": iou_p75,
            "biou_median": biou_med,
            "biou_p25": biou_p25,
            "biou_p75": biou_p75,
            "jitter_norm_median_pct": jitter_med,
            "jitter_norm_p95_pct": jitter_p95,
            "area_cv_pct": area_cv_pct,
            "empty_frames_pct": empty_frames_pct,
            "proc_fps_measured": proc_fps_measured,
            "proc_fps_theoretical": proc_fps_theoretical,
            "roi_w": tgt_w,
            "roi_h": tgt_h,
            "target_W": tgt_w,
            "target_H": tgt_h,
            "stride": stride,
        })
    # Return summary dictionary with output paths and key metrics
    return {
        "video": str(path),
        "overlay": str(overlay_path),
        "per_frame_csv": str(per_frame_path),
        "log": str(log_path),
        "params": str(params_path),
        "re_prompts": str(re_csv),
        "fps": fps,
        "iou_median": iou_med,
        "biou_median": biou_med,
    }


def main() -> None:
    # Main entry point: processes one or more videos with YOLO + SAM-2
    # Supports single file or directory of videos
    args = parse_args()
    out_root = Path(args.out_dir)
    ensure_dir(out_root)  # Create output directory
    
    # Find all video sources (single file or recursive directory search)
    src = Path(args.data)
    sources = list_sources(src if src.exists() else src)
    if not sources:
        raise SystemExit(f"No sources found under {args.data}")

    # Process each video sequentially
    results: List[Dict[str, object]] = []
    for p in sources:
        print(f"Processing video: {p}")
        res = process_video(p, args, out_root)  # Process single video
        print(json.dumps(res, indent=2))  # Print results to console
        results.append(res)

    # Write combined summary JSON with results from all videos
    summary = out_root / "summary.csv"
    with summary.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary written to {summary}")


if __name__ == "__main__":
    # Script entry point: run main function when executed directly
    main()




