# Standard library imports for argument parsing, file I/O, and utilities
import argparse
import csv
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Third-party imports for image processing and deep learning
import cv2  # OpenCV for image/video operations
import numpy as np  # NumPy for numerical array operations
import torch  # PyTorch for deep learning framework

# Import SAM-2 model classes from HuggingFace transformers
# These are required for video segmentation tasks
try:
    from transformers import Sam2VideoModel, Sam2VideoProcessor
except ModuleNotFoundError:
    # Graceful error handling if transformers library is not installed
    print(
        "Missing dependency: transformers. Please install the SAM-2 requirements "
        "(transformers>=4.43.0 huggingface_hub opencv-python pillow numpy pandas tqdm)."
    )
    raise

# Import utility functions from sam2_smoke module
# These handle ground truth data loading, point sampling, and frame processing
from sam2_smoke import (
    earliest_label_json,  # Find the first labeled frame in ground truth
    load_labelme_polys,  # Load polygon annotations from LabelMe JSON format
    make_clicks_with_negatives,  # Generate positive and negative click points for prompting
    overlay_mask,  # Overlay segmentation mask on video frame
    prob_to_mask,  # Convert probability map to binary mask
    read_resize_frames,  # Read and resize video frames
    roi_from_polys_scaled,  # Extract region of interest from polygons
)


def parse_args() -> argparse.Namespace:
    # Command-line argument parser for SAM-2 pilot runner
    # Defines all configurable parameters that can be set via command line
    parser = argparse.ArgumentParser(description="SAM-2 pilot runner")
    parser.add_argument("--clips", nargs="+", help="List of MP4 clips to process")  # Required list of video files
    parser.add_argument("--gt-root", "--gt_root", dest="gt_root", default="data/gt_frames")  # Ground truth directory
    parser.add_argument("--weights", default="facebook/sam2.1-hiera-tiny")  # SAM-2 model weights identifier
    parser.add_argument("--target-width", "--target_width", dest="target_width", type=int, default=640)  # Target frame width
    parser.add_argument("--stride", type=int, default=2)  # Frame stride (process every Nth frame)
    parser.add_argument("--roi-pad", "--roi_pad", dest="roi_pad", type=int, default=16)  # Padding around ROI in pixels
    parser.add_argument("--outdir", default="outputs/sam2_gt")  # Output directory for results
    parser.add_argument(
        "--max-frames",
        "--max_frames",
        dest="max_frames",
        type=int,
        default=None,
        help="Optional limit on number of frames to decode per clip",  # Limit frames for testing
    )
    return parser.parse_args()


def polys_to_mask(
    polys: Sequence[Tuple[int, np.ndarray]],
    src_w: int,
    src_h: int,
    tgt_w: int,
    tgt_h: int,
) -> np.ndarray:
    # Convert polygon annotations to binary mask, scaling from source to target dimensions
    # Groups polygons by ID and fills them to create a mask
    if not polys:
        return np.zeros((tgt_h, tgt_w), dtype=np.uint8)  # Empty mask if no polygons

    # Calculate scaling factors for width and height
    sx = tgt_w / max(1, src_w)
    sy = tgt_h / max(1, src_h)

    # Group polygons by their group ID
    grouped: Dict[int, List[np.ndarray]] = {}
    for gid, pts in polys:
        pts = np.asarray(pts, dtype=np.float32)
        scaled = np.column_stack((pts[:, 0] * sx, pts[:, 1] * sy))  # Scale coordinates
        grouped.setdefault(int(gid), []).append(scaled)

    # Create mask by filling polygons
    mask = np.zeros((tgt_h, tgt_w), dtype=np.uint8)
    for segments in grouped.values():
        polygons = [np.round(seg).astype(np.int32) for seg in segments if seg.shape[0] >= 3]  # Need at least 3 points
        if polygons:
            cv2.fillPoly(mask, polygons, 255)  # Fill polygons with white (255)

    return mask


def band_mask(mask_bool: np.ndarray, dilate_px: int = 3) -> np.ndarray:
    # Create a band mask representing the boundary/edge of the input mask
    # Used for Boundary IoU (BIoU) metric computation
    if not mask_bool.any():
        return np.zeros_like(mask_bool, dtype=bool)  # Empty mask returns empty band

    mask = mask_bool.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grad = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)  # Extract boundary using gradient
    if dilate_px > 0:
        # Dilate the boundary to create a band of specified width
        k = 2 * dilate_px + 1
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        grad = cv2.dilate(grad, dilate_kernel)
    return grad > 0


def compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    # Compute Intersection over Union (IoU) metric between two binary masks
    # Returns 1.0 if both masks are empty (perfect match)
    inter = np.logical_and(a, b).sum()  # Intersection: pixels in both masks
    union = np.logical_or(a, b).sum()  # Union: pixels in either mask
    if union == 0:
        return 1.0  # Both empty: perfect match
    return float(inter) / float(union)  # IoU = intersection / union


def centroid_from_mask(mask: np.ndarray) -> Tuple[float, float]:
    # Calculate the centroid (center of mass) of a binary mask
    # Returns (x, y) coordinates, or (NaN, NaN) if mask is empty
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return math.nan, math.nan  # Empty mask
    return float(xs.mean()), float(ys.mean())  # Mean of x and y coordinates


def write_per_frame_csv(path: Path, rows: List[Dict[str, object]], columns: Sequence[str]) -> None:
    # Write per-frame metrics to a CSV file
    # Handles NaN values by converting them to empty strings for CSV compatibility
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            serialised = {}
            for key, value in row.items():
                # Convert NaN to empty string for CSV (CSV doesn't support NaN)
                if isinstance(value, float) and math.isnan(value):
                    serialised[key] = ""
                else:
                    serialised[key] = value
            writer.writerow(serialised)


def nan_percentiles(data: Iterable[float], q: Sequence[float]) -> List[float]:
    # Calculate percentiles while ignoring NaN values
    # Returns list of percentiles corresponding to the quantiles in q
    arr = np.array([d for d in data if not math.isnan(d)], dtype=np.float32)
    if arr.size == 0:
        return [math.nan for _ in q]  # All NaN if no valid data
    return np.percentile(arr, q=q).tolist()


def nan_mean(data: Iterable[float]) -> float:
    # Calculate mean while ignoring NaN values
    # Returns NaN if no valid data points exist
    arr = np.array([d for d in data if not math.isnan(d)], dtype=np.float32)
    if arr.size == 0:
        return math.nan
    return float(arr.mean())


def run_clip(
    clip_path: Path,
    args: argparse.Namespace,
    output_root: Path,
) -> Dict[str, object]:
    # Main function to process a single video clip through SAM-2 tracking pipeline
    # Handles frame loading, ROI extraction, model inference, metrics computation, and output generation
    clip_id = clip_path.stem

    # Load and resize video frames to target width
    frames, (tgt_w, tgt_h), fps = read_resize_frames(str(clip_path), args.target_width)
    # Optionally limit the number of frames for testing/debugging
    if args.max_frames is not None:
        max_f = max(0, int(args.max_frames))
        frames = frames[:max_f]
        if not frames:
            print("No frames remaining after applying --max-frames limit")
            raise SystemExit(1)

    # Load ground truth annotations and find seed frame
    gt_dir = Path(args.gt_root) / clip_id
    seed_idx, seed_json = earliest_label_json(gt_dir)  # Find first labeled frame
    if seed_idx >= len(frames):
        print(
            f"Seed frame {seed_idx} exceeds available frames ({len(frames)}). "
            "Increase --max-frames or target-width to decode more frames."
        )
        raise SystemExit(1)

    # Load seed frame polygons (ground truth annotations)
    src_w, src_h, seed_polys = load_labelme_polys(seed_json)
    if not seed_polys:
        print(f"No 'billboard' polygons in {seed_json.name}")
        raise SystemExit(1)

    # Extract Region of Interest (ROI) and generate click prompts
    # ROI is a cropped region around the object to improve tracking efficiency
    pad = int(args.roi_pad)
    clicks = (None, None)  # Will store (points, labels) tuple
    roi = (0, 0, tgt_w, tgt_h)  # Default to full frame
    # Try up to 2 attempts to generate valid clicks (retry with larger pad if needed)
    for attempt in range(2):
        roi = roi_from_polys_scaled(seed_polys, src_w, src_h, tgt_w, tgt_h, pad)  # Extract ROI from polygons
        # Generate positive (inside object) and negative (outside object) click points
        clicks = make_clicks_with_negatives(
            seed_polys,
            src_w,
            src_h,
            tgt_w,
            tgt_h,
            roi,
            pos_per_obj=12,  # Number of positive points per object
            neg_stride=30,  # Stride for sampling negative points
            neg_offset=6,  # Offset from object boundary for negative points
        )
        if clicks[0] is not None:
            break  # Successfully generated clicks
        print("Could not sample interior points; increasing ROI pad by +8 and retrying...")
        pad += 8  # Increase padding and retry
    if clicks[0] is None:
        # Failed to generate clicks after retries
        x0, y0, x1, y1 = roi
        print(
            f"Failed to sample interior/negative points from {seed_json.name} within ROI"
            f" ({x0},{y0},{x1},{y1})."
        )
        raise SystemExit(1)

    # Extract ROI dimensions
    x0, y0, x1, y1 = roi
    roi_w = x1 - x0
    roi_h = y1 - y0

    # Log processing information
    print(f"Processing {clip_id}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"  # Use MPS (Apple Silicon) if available
    dtype = torch.float32
    print(f"  Device: {device}")
    print(f"  Seeding from {seed_json.name}")
    print(f"  ROI: ({x0},{y0},{x1},{y1}) size={roi_w}x{roi_h}")

    # Crop all frames to ROI for more efficient processing
    frames_roi = [frame.crop((x0, y0, x1, y1)) for frame in frames]

    # Load SAM-2 model and processor
    model = Sam2VideoModel.from_pretrained(args.weights).to(device=device, dtype=dtype)
    processor = Sam2VideoProcessor.from_pretrained(args.weights)
    # Initialize video session with ROI-cropped frames
    session = processor.init_video_session(
        video=frames_roi,
        inference_device=device,
        processing_device=device,
        video_storage_device=device,
        dtype=dtype,
    )

    # Add seed prompts (click points) to the inference session at the seed frame
    # This initializes tracking by telling the model what to track
    processor.add_inputs_to_inference_session(
        inference_session=session,
        frame_idx=seed_idx,
        obj_ids=1,
        input_points=clicks[0],  # Positive and negative click points
        input_labels=clicks[1],  # Labels: 1 for positive, 0 for negative
    )

    # Initialize tracking state
    total_frames = len(frames)
    masks_full: List[Optional[np.ndarray]] = [None] * total_frames  # Store masks for all frames
    runtime_ms: Dict[int, float] = {}  # Track inference time per frame
    roi_shape = [[roi_h, roi_w]]  # ROI dimensions for post-processing

    # Process seed frame: run inference to get initial mask
    seed_start = time.perf_counter()
    seed_output = model(inference_session=session, frame_idx=seed_idx)
    seed_time = time.perf_counter() - seed_start
    # Post-process seed mask: convert model output to probability map
    seed_post = processor.post_process_masks(
        [seed_output.pred_masks], original_sizes=roi_shape, binarize=False
    )[0]
    if isinstance(seed_post, torch.Tensor):
        seed_post = seed_post.detach().cpu().numpy()  # Convert to NumPy
    # Handle multi-object masks by taking maximum probability
    seed_prob = np.max(seed_post, axis=0) if seed_post.ndim == 3 else seed_post
    seed_roi = prob_to_mask(seed_prob, shape=(roi_h, roi_w))  # Convert to binary mask
    # Place ROI mask back into full-frame coordinates
    seed_full = np.zeros((tgt_h, tgt_w), dtype=np.uint8)
    seed_full[y0:y1, x0:x1] = seed_roi[:roi_h, :roi_w]
    masks_full[seed_idx] = seed_full
    runtime_ms[seed_idx] = seed_time * 1000.0  # Store runtime in milliseconds

    # Determine which frames to process based on stride
    stride = max(1, int(args.stride))
    target_indices = set(range(seed_idx, total_frames, stride))  # Frames to compute metrics for
    remaining_targets = sorted(idx for idx in target_indices if idx > seed_idx)  # Exclude seed frame

    # Propagate tracking through remaining frames using video iterator
    # This efficiently processes frames sequentially using temporal information
    processed = 0
    propagation_elapsed = 0.0
    if remaining_targets:
        start_time = time.perf_counter()
        last_time = start_time
        # Use propagate_in_video_iterator for efficient sequential processing
        for output in model.propagate_in_video_iterator(
            inference_session=session, start_frame_idx=seed_idx + 1
        ):
            idx = output.frame_idx
            now = time.perf_counter()
            frame_runtime = (now - last_time) * 1000.0  # Measure time per frame
            last_time = now

            # Skip frames outside valid range or not in target set
            if idx >= total_frames:
                continue
            if idx not in target_indices:
                continue

            # Post-process mask: convert model output to probability map
            post = processor.post_process_masks(
                [output.pred_masks], original_sizes=roi_shape, binarize=False
            )[0]
            if isinstance(post, torch.Tensor):
                post = post.detach().cpu().numpy()
            # Handle multi-object masks by taking maximum probability
            prob = np.max(post, axis=0) if post.ndim == 3 else post
            mask_roi = prob_to_mask(prob, shape=(roi_h, roi_w))  # Convert to binary mask
            # Place ROI mask back into full-frame coordinates
            full_mask = np.zeros((tgt_h, tgt_w), dtype=np.uint8)
            full_mask[y0:y1, x0:x1] = mask_roi[:roi_h, :roi_w]
            masks_full[idx] = full_mask
            runtime_ms[idx] = frame_runtime

            processed += 1
            if processed == len(remaining_targets):
                break  # Stop when all target frames are processed
        propagation_elapsed = time.perf_counter() - start_time

    # Calculate processing performance metrics
    total_processed = len(target_indices)
    total_time = seed_time + propagation_elapsed
    fps_measured = total_processed / max(1e-6, total_time)  # Actual processing FPS

    # Fill in missing masks: use last known mask for frames that weren't processed
    # This ensures every frame has a mask for visualization
    last_mask = np.zeros((tgt_h, tgt_w), dtype=np.uint8)
    for i in range(total_frames):
        if masks_full[i] is None:
            masks_full[i] = last_mask.copy()  # Use previous mask if current frame not processed
        else:
            last_mask = masks_full[i]  # Update last known mask

    # Generate overlay video: visualize tracking masks overlaid on original frames
    overlay_dir = output_root / clip_id / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = overlay_dir / f"{clip_id}_sam2.mp4"
    writer = cv2.VideoWriter(
        str(overlay_path),
        cv2.VideoWriter_fourcc(*"mp4v"),  # MPEG-4 codec
        fps,
        (tgt_w, tgt_h),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Process each frame: clean mask and overlay on original video
    for frame, mask in zip(frames, masks_full):
        rgb = np.array(frame)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)  # Remove small noise
        bgr = overlay_mask(rgb, cleaned)  # Overlay mask on frame
        writer.write(bgr)
    writer.release()
    # Validate that video was written successfully
    if overlay_path.exists() and overlay_path.stat().st_size == 0:
        print(f"Overlay failed to write: {overlay_path}")
        print("If codecs are missing, try: brew install ffmpeg")
        raise SystemExit(1)

    # Compute metrics by comparing predictions with ground truth annotations
    label_paths = sorted(gt_dir.glob("frame_*.json"))
    frame_data: List[Dict[str, object]] = []
    prev_centroid = (math.nan, math.nan)  # Track centroid for jitter calculation

    theoretical_fps = fps / stride  # Theoretical FPS based on video FPS and stride

    # Process each ground truth frame to compute metrics
    for label_path in label_paths:
        frame_idx = int(label_path.stem.split("_")[-1])
        # Skip seed frame and frames outside valid range
        if frame_idx == seed_idx or frame_idx >= len(masks_full):
            continue
        if frame_idx < seed_idx:
            continue

        # Load ground truth polygons and convert to mask
        src_w_i, src_h_i, polys = load_labelme_polys(label_path)
        gt_mask_full = polys_to_mask(polys, src_w_i, src_h_i, tgt_w, tgt_h)

        # Extract ROI regions for comparison (metrics computed on ROI, not full frame)
        pred_mask_full = masks_full[frame_idx] > 0
        gt_mask_bool = gt_mask_full > 0

        pred_roi = pred_mask_full[y0:y1, x0:x1]  # Predicted mask in ROI
        gt_roi = gt_mask_bool[y0:y1, x0:x1]  # Ground truth mask in ROI

        # Compute IoU (Intersection over Union) metric
        iou = compute_iou(pred_roi, gt_roi)

        # Compute BIoU (Boundary IoU): IoU of boundary bands
        pred_band = band_mask(pred_roi)
        gt_band = band_mask(gt_roi)
        biou = compute_iou(pred_band, gt_band)

        # Compute mask statistics
        area = float(pred_roi.sum())  # Predicted mask area
        cx, cy = centroid_from_mask(pred_roi)  # Predicted centroid

        is_empty = 1 if math.isnan(cx) else 0  # Flag for empty masks

        # Calculate jitter: frame-to-frame centroid movement (indicates tracking stability)
        shift_px = math.nan
        shift_norm = math.nan
        if frame_idx in runtime_ms and not math.isnan(cx) and not math.isnan(prev_centroid[0]):
            dx = cx - prev_centroid[0]
            dy = cy - prev_centroid[1]
            shift_px = math.hypot(dx, dy)  # Euclidean distance
            shift_norm = (shift_px / max(1, roi_w)) * 100.0  # Normalized to ROI width percentage
        # Update previous centroid for next iteration
        if frame_idx in runtime_ms and math.isnan(cx):
            prev_centroid = (math.nan, math.nan)
        elif frame_idx in runtime_ms and not math.isnan(cx):
            prev_centroid = (cx, cy)

        runtime_val = runtime_ms.get(frame_idx, math.nan)  # Get inference time for this frame

        # Store per-frame metrics in row dictionary
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

    # Write per-frame metrics to CSV file for detailed analysis
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

    # Compute summary statistics across all frames
    ious = [row["iou"] for row in frame_data]
    biou_vals = [row["biou"] for row in frame_data]
    jitter_vals = [
        val
        for val in (row["centroid_shift_norm"] for row in frame_data)
        if isinstance(val, float) and not math.isnan(val)
    ]
    area_vals = [row["area_px"] for row in frame_data]
    empty_vals = [row["is_empty"] for row in frame_data]

    # Calculate percentiles for IoU, BIoU, and jitter metrics
    iou_p25, iou_med, iou_p75 = nan_percentiles(ious, [25, 50, 75])
    biou_p25, biou_med, biou_p75 = nan_percentiles(biou_vals, [25, 50, 75])
    jitter_med, jitter_p95 = nan_percentiles(jitter_vals, [50, 95]) if jitter_vals else (math.nan, math.nan)

    # Calculate area statistics: mean, standard deviation, and coefficient of variation
    area_mean = nan_mean(area_vals)
    area_std = math.nan
    if not math.isnan(area_mean):
        arr = np.array(area_vals, dtype=np.float32)
        area_std = float(np.std(arr))
    area_cv = math.nan if math.isnan(area_mean) or area_mean == 0 else (area_std / area_mean) * 100.0  # Coefficient of variation
    empty_pct = (sum(empty_vals) / len(empty_vals) * 100.0) if empty_vals else math.nan  # Percentage of empty frames

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
        "proc_fps_theoretical": theoretical_fps,
        "roi_w": roi_w,
        "roi_h": roi_h,
        "target_W": tgt_w,
        "target_H": tgt_h,
        "stride": stride,
    }

    # Log summary statistics to console
    print(
        "  Processed {} frames at ~{:.2f} fps (theoretical {:.2f})".format(
            total_processed, fps_measured, theoretical_fps
        )
    )

    return summary


def main() -> None:
    # Main entry point: orchestrates processing of multiple video clips
    # Handles argument parsing, clip processing, and result aggregation
    args = parse_args()
    if not args.clips:
        raise SystemExit("No clips provided")  # Validate that clips are specified

    # Set up output directory and initialize summary list
    output_root = Path(args.outdir)
    summaries: List[Dict[str, object]] = []

    # Process each clip and collect summaries
    for clip in args.clips:
        summary = run_clip(Path(clip), args, output_root)
        summaries.append(summary)

    # Write aggregated summary CSV file with metrics from all processed clips
    # This provides a high-level overview of tracking performance across all clips
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


# Entry point: execute main function when script is run directly
if __name__ == "__main__":
    main()
