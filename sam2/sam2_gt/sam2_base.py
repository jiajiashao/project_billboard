# Standard library imports for argument parsing, file I/O, and utilities
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
# These handle ground truth data loading, bounding box operations, and frame processing
from sam2_smoke import (
    bbox_from_polys_scaled,  # Convert polygons to scaled bounding boxes
    earliest_label_json,  # Find the first labeled frame in ground truth
    load_labelme_polys,  # Load polygon annotations from LabelMe JSON format
    make_box_with_edge_negatives,  # Generate negative points around bounding box edges
    overlay_mask,  # Overlay segmentation mask on video frame
    prob_to_mask,  # Convert probability map to binary mask
    read_resize_frames,  # Read and resize video frames
)
# Import utility functions from sam2_pilot module
# These handle mask operations, metrics computation, and output writing
from sam2_pilot import (
    band_mask,  # Create band mask for boundary IoU computation
    centroid_from_mask,  # Calculate centroid coordinates from mask
    compute_iou,  # Compute Intersection over Union metric
    nan_mean,  # Calculate mean handling NaN values
    nan_percentiles,  # Calculate percentiles handling NaN values
    polys_to_mask,  # Convert polygons to binary mask
    write_per_frame_csv,  # Write per-frame metrics to CSV file
)


# Main configuration dictionary that defines run parameters, clip settings, metrics, and logging options
# This is the central configuration that controls all aspects of the SAM-2 tracking pipeline
RUN_SPEC = {
    "run_id": "gt",  # Unique identifier for this run (used in output filenames)
    "model": {
        "name": "sam2",  # Model name identifier
        "random_seed": 42,  # Random seed for reproducibility
        "deterministic": True,  # Enable deterministic algorithms for consistent results
    },
    # "clips": [
    #     {
    #         "id": "clip_gentle",
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
    #             "enabled": False,
    #             "triggers": {
    #                 "centroid_jump_pct_of_width": 5000.0,
    #                 "area_change_pct": 30000.0,
    #                 "empty_mask_consecutive": 20000,
    #             },
    #             "action": "reseed_with_box_plus_neg",
    #             "cooldown_frames": 0,
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
    #             "enabled": False,
    #             "triggers": {
    #                 "centroid_jump_pct_of_width": 5.0,
    #                 "area_change_pct": 30.0,
    #                 "empty_mask_consecutive": 2,
    #             },
    #             "action": "reseed_with_box_plus_neg",
    #             "cooldown_frames": 0,
    #             "max_events": 0,
    #         },
    #     },
    # ],
    # Metrics configuration: defines which metrics to compute and how to score frames
    "metrics": {
        "compute": ["iou", "biou", "jitter_norm_pct"],  # List of metrics to calculate (IoU, Boundary IoU, jitter)
        "scoring_policy": {
            "skip_frames_with_empty_gt": True,  # Skip frames where ground truth has no annotations
            "include_seed_frame": True,  # Include the seed frame in metric calculations
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
    # Logging configuration: controls what metadata is saved with each run
    "logging": {
        "write_params_json": True,  # Save run parameters to JSON file
        "write_script_hash": True,  # Include script hash for version tracking
        "capture_git_commit": True,  # Record git commit hash if available
        "echo_config_to_log": True,  # Include full config in log file
    },
    # Safety configuration: how to handle existing output directories
    "safety": {
        "if_out_dir_exists": "create_new",  # Create new timestamped directory if output exists
    },
}

# Constants for mask processing and thresholding
AREA_EPS = 1.0  # Minimum area (in pixels) to consider a mask as non-empty
MASK_THRESHOLD = 0.5  # Probability threshold for converting probability maps to binary masks


def parse_args() -> argparse.Namespace:
    # Command-line argument parser for SAM-2 pilot runner
    # Defines all configurable parameters that can be set via command line
    parser = argparse.ArgumentParser(description="SAM-2 pilot runner for Fix-2")
    parser.add_argument("--data-root", dest="data_root", default="data")  # Root directory containing clips and ground truth
    parser.add_argument("--weights", default="facebook/sam2.1-hiera-tiny")  # SAM-2 model weights identifier
    parser.add_argument("--runs-root", dest="runs_root", default="runs")  # Root directory for output runs
    parser.add_argument("--clips", nargs="*", help="Optional subset of clip IDs to process")  # Filter specific clips
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cuda")  # Computation device selection
    parser.add_argument("--reseed", action="store_true", default=False, help="Enable dynamic reseeding (off by default)")  # Enable reseeding
    return parser.parse_args()


def select_device(preferred: Optional[str]) -> str:
    # Device selection logic with fallback handling
    # Selects the computation device (CPU, CUDA, or MPS) based on availability
    if preferred:
        # If user specified a device, validate availability (note: bug in original code checks mps for cuda)
        if preferred == "cuda" and not torch.backends.mps.is_available():
            print("Requested mps device but it is unavailable; falling back to cpu")
            return "cpu"
        return preferred
    # Auto-select: prefer CUDA if available, otherwise CPU
    return "cuda" if torch.backends.cuda.is_available() else "cpu"


def set_random_state(seed: int) -> None:
    # Set random seeds for reproducibility across Python, NumPy, and PyTorch
    # Ensures deterministic behavior when deterministic mode is enabled
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)  # Enable deterministic algorithms
    except RuntimeError:
        pass  # Continue if deterministic algorithms are not available


def ensure_dir(path: Path) -> None:
    # Create directory and all parent directories if they don't exist
    # Safe to call multiple times (won't raise error if directory exists)
    path.mkdir(parents=True, exist_ok=True)


def prepare_run_dir(root: Path, run_id: str) -> Tuple[Path, str]:
    # Prepare output directory for a run, creating timestamped version if directory exists
    # Returns the directory path and the final folder name used
    ensure_dir(root)
    candidate = root / run_id
    if not candidate.exists():
        # If base directory doesn't exist, use it directly
        candidate.mkdir()
        return candidate, candidate.name
    # If directory exists, append timestamp to create unique directory
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / f"{run_id}_{timestamp}"
    candidate.mkdir()
    return candidate, candidate.name


def get_git_commit(root: Path) -> str:
    # Retrieve the current git commit hash for version tracking
    # Returns "N/A" if not in a git repository or if git command fails
    if not (root / ".git").exists():
        return "N/A"
    import subprocess

    try:
        # Execute git command to get HEAD commit hash
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        return commit or "N/A"
    except Exception:
        return "N/A"


def clone_nested(data: Optional[Sequence]) -> Optional[List]:
    # Deep clone nested data structures using JSON serialization
    # Useful for copying configuration dictionaries without reference sharing
    if data is None:
        return None
    return json.loads(json.dumps(data))


def bbox_from_mask(
    mask_bool: np.ndarray,
    pad_px: int,
    width: int,
    height: int,
) -> Optional[Tuple[float, float, float, float]]:
    # Extract bounding box from a binary mask with optional padding
    # Returns (x0, y0, x1, y1) in XYXY format, or None if mask is empty
    mask = np.asarray(mask_bool, dtype=bool)
    if not mask.any():
        return None  # Empty mask, no bounding box

    # Find all non-zero pixel coordinates
    ys, xs = np.nonzero(mask)
    # Calculate bounding box with padding, clamped to image boundaries
    x0 = max(0, math.floor(float(xs.min()) - pad_px))
    y0 = max(0, math.floor(float(ys.min()) - pad_px))
    x1 = min(width - 1, math.ceil(float(xs.max()) + pad_px))
    y1 = min(height - 1, math.ceil(float(ys.max()) + pad_px))

    # Handle edge case where bounding box has zero or negative dimensions
    if x1 <= x0:
        x1 = min(width - 1, x0 + 1)
    if y1 <= y0:
        y1 = min(height - 1, y0 + 1)

    # Ensure minimum size and return as float coordinates
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
    # Run inference on a single frame using the SAM-2 model
    # Returns the binary mask and inference runtime in milliseconds
    start = time.perf_counter()
    # Run model inference on the specified frame index
    output = model(inference_session=session, frame_idx=frame_idx)
    runtime_ms = (time.perf_counter() - start) * 1000.0  # Measure inference time
    
    # Post-process the model output masks
    post = processor.post_process_masks(
        [output.pred_masks],
        original_sizes=[[tgt_size[1], tgt_size[0]]],  # Note: size order is (height, width)
        binarize=False,  # Keep as probability map for thresholding
    )[0]
    # Convert PyTorch tensor to NumPy array if needed
    if isinstance(post, torch.Tensor):
        post = post.detach().cpu().numpy()
    # Handle multi-object masks by taking maximum probability across objects
    prob = np.max(post, axis=0) if post.ndim == 3 else post
    # Convert probability map to binary mask using threshold
    mask = prob_to_mask(prob, threshold=MASK_THRESHOLD, shape=tgt_size[::-1])
    return mask, runtime_ms


def create_logger() -> Tuple[List[str], Callable[[str], None]]:
    # Create a logger that both prints to console and stores messages in a list
    # Returns the list of log lines and a logging function
    lines: List[str] = []

    def _log(msg: str) -> None:
        print(msg)  # Print to console
        lines.append(msg)  # Store in list for later writing to file

    return lines, _log


def write_json(path: Path, payload: Dict) -> None:
    # Write a dictionary to a JSON file with pretty formatting
    # Creates parent directories if they don't exist
    ensure_dir(path.parent)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def write_reprompt_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    # Write reseeding/re-prompting events to a CSV file
    # Records when and why tracking was reinitialized during processing
    ensure_dir(path.parent)
    fields = [
        "event_idx",  # Sequential event number
        "frame_idx",  # Frame where reseeding occurred
        "reasons",  # Comma-separated list of trigger reasons
        "centroid_jump_px",  # Centroid movement in pixels
        "centroid_jump_pct",  # Centroid movement as percentage of width
        "area_change_pct",  # Area change percentage
        "empty_streak",  # Consecutive frames with empty masks
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _iter_labelme_polys(gt_dir: Path) -> Sequence[Tuple[int, Path, int, int, List[Tuple[int, np.ndarray]]]]:
    # Iterator over LabelMe JSON annotation files in ground truth directory
    # Yields frame index, file path, source dimensions, and polygon annotations
    frame_paths = sorted(gt_dir.glob("frame_*.json"))
    for path in frame_paths:
        frame_idx = int(path.stem.split("_")[-1])  # Extract frame number from filename
        src_w, src_h, polys = load_labelme_polys(path)  # Load polygons from JSON
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
    # Load video frames and prepare seed initialization data for tracking
    # Returns frames, dimensions, FPS, seed frame info, bounding box, points, labels, and ground truth boxes
    frames, (tgt_w, tgt_h), fps = read_resize_frames(str(clip_path), target_width)
    seed_idx, seed_json = earliest_label_json(gt_dir)  # Find first labeled frame
    seed_src_w = seed_src_h = None
    seed_polys: Optional[List[Tuple[int, np.ndarray]]] = None
    pad = int(clip_cfg["seed"]["bbox_pad_px"])
    gt_boxes: Dict[int, Tuple[float, float, float, float]] = {}  # Store GT boxes for all frames
    
    # Iterate through ground truth annotations to find first frame with polygons
    for idx, json_path, src_w, src_h, polys in _iter_labelme_polys(gt_dir):
        if polys:
            # Found first frame with annotations - use as seed
            gt_boxes[idx] = bbox_from_polys_scaled(polys, src_w, src_h, tgt_w, tgt_h, pad)
            seed_idx = idx
            seed_json = json_path
            seed_src_w = src_w
            seed_src_h = src_h
            seed_polys = polys
            break
    
    # Validate that seed frame was found
    if seed_polys is None:
        raise SystemExit(f"No 'billboard' polygons found under {gt_dir}")
    if seed_idx >= len(frames):
        raise SystemExit(
            f"Seed frame {seed_idx} exceeds available frames ({len(frames)}). "
            "Increase input width or verify the clip."
        )
    
    # Generate seed bounding box from polygons
    box = bbox_from_polys_scaled(seed_polys, seed_src_w, seed_src_h, tgt_w, tgt_h, pad)
    points = None
    labels = None
    neg_cfg = clip_cfg["seed"].get("negatives", {})
    
    # Generate negative points if configured (used to help model understand what NOT to track)
    if neg_cfg.get("mode") == "edge_fence":
        _, points, labels = make_box_with_edge_negatives(
            seed_polys,
            seed_src_w,
            seed_src_h,
            tgt_w,
            tgt_h,
            pad,
            neg_count=int(neg_cfg.get("count", 0)),  # Number of negative points
            neg_offset=int(neg_cfg.get("offset_px", 0)),  # Offset from bounding box edge
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
    # Main function to process a single video clip through SAM-2 tracking pipeline
    # Handles frame loading, model inference, reseeding, metrics computation, and output generation
    clip_id = clip_cfg["id"]
    clip_dir = run_dir / clip_id
    ensure_dir(clip_dir)
    log_lines, log = create_logger()  # Create logger for this clip

    # Set up file paths for input video and ground truth annotations
    data_root: Path = context["data_root"]
    clip_path = data_root / "clips" / f"{clip_id}.mp4"
    gt_dir = data_root / "gt_frames" / clip_id
    if not clip_path.exists():
        raise SystemExit(
            f"Clip not found: {clip_path}. Try: ls {clip_path.parent} to verify the filename."
        )

    # Load video frames and prepare seed data for tracking initialization
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

    # Log processing information
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

    # Initialize SAM-2 video session with all frames
    # This sets up the model's internal state for tracking across the video
    session = processor.init_video_session(
        video=frames,
        inference_device=device,
        processing_device=device,
        video_storage_device=device,
        dtype=context["dtype"],
    )

    # Prepare seed prompts (bounding box, points, labels) for initializing tracking
    bbox_pad_px = int(clip_cfg.get("seed", {}).get("bbox_pad_px", 0))
    seed_prompt_boxes = [[[seed_box[0], seed_box[1], seed_box[2], seed_box[3]]]]  # Format: [[[x0,y0,x1,y1]]]
    seed_prompt_points = clone_nested(seed_points)  # Deep copy to avoid reference issues
    seed_prompt_labels = clone_nested(seed_labels)  # Deep copy to avoid reference issues

    # Add seed prompts to the inference session at the seed frame
    # This tells the model what to track starting from the seed frame
    processor.add_inputs_to_inference_session(
        inference_session=session,
        frame_idx=seed_idx,
        obj_ids=1,
        input_boxes=seed_prompt_boxes,
        input_points=seed_prompt_points,
        input_labels=seed_prompt_labels,
    )

    # Initialize tracking state variables
    total_frames = len(frames)
    masks_full: List[np.ndarray] = [np.zeros((tgt_h, tgt_w), dtype=np.uint8) for _ in range(total_frames)]  # Store masks for all frames
    frame_runtimes: Dict[int, float] = {}  # Track inference time per frame
    stride = max(1, int(clip_cfg["stride"]))  # Frame stride (1 = process every frame)
    target_indices = set(range(seed_idx, total_frames, stride))  # Frames to compute metrics for

    # Reseeding state tracking: monitor tracking quality and reinitialize when needed
    prev_centroid: Optional[Tuple[float, float]] = None  # Previous frame's centroid position
    prev_area: Optional[float] = None  # Previous frame's mask area
    empty_streak = 0  # Consecutive frames with empty masks
    cooldown_remaining = 0  # Frames remaining before next reseed is allowed
    reseed_cfg = clip_cfg.get("reseed", {})
    reseed_enabled = reseed_cfg.get("enabled", False)  # Whether reseeding is enabled
    triggers = reseed_cfg.get("triggers", {})  # Thresholds that trigger reseeding
    cooldown_frames = int(reseed_cfg.get("cooldown_frames", 0))  # Minimum frames between reseeds
    max_events = int(reseed_cfg.get("max_events", 0))  # Maximum number of reseed events
    reseed_events: List[Dict[str, object]] = []  # Record of all reseed events
    reseed_count = 0  # Counter for reseed events

    # Main tracking loop: process frames sequentially from seed frame to end
    with torch.inference_mode():  # Disable gradient computation for inference
        for idx in range(seed_idx, total_frames):
            # Run inference to get mask for current frame
            mask, runtime_ms = infer_frame(model, processor, session, idx, (tgt_w, tgt_h))
            masks_full[idx] = mask  # Store mask for later use
            if idx in target_indices:
                # Only compute metrics and check reseeding for target frames (based on stride)
                frame_runtimes[idx] = frame_runtimes.get(idx, 0.0) + runtime_ms

                # Analyze mask quality: compute area, centroid, and detect empty masks
                mask_bool = mask > 0
                area = float(mask_bool.sum())  # Total area of mask in pixels
                is_empty = area <= AREA_EPS  # Check if mask is effectively empty
                centroid = centroid_from_mask(mask_bool) if not is_empty else (math.nan, math.nan)

                # Track consecutive empty frames
                if is_empty:
                    empty_streak += 1
                else:
                    empty_streak = 0

                # Calculate centroid jump: how much the object moved between frames
                jump_px = math.nan
                jump_pct = math.nan
                if not is_empty and prev_centroid is not None and not math.isnan(prev_centroid[0]):
                    dx = centroid[0] - prev_centroid[0]
                    dy = centroid[1] - prev_centroid[1]
                    jump_px = math.hypot(dx, dy)  # Euclidean distance
                    jump_pct = (jump_px / max(1.0, tgt_w)) * 100.0  # Normalized to frame width

                # Calculate area change: how much the mask size changed
                area_change_pct = math.nan
                if prev_area is not None and prev_area > AREA_EPS and area > AREA_EPS:
                    area_change_pct = abs(area - prev_area) / prev_area * 100.0

                # Check if reseeding is allowed and if any triggers are activated
                triggers_allowed = reseed_enabled and cooldown_remaining == 0 and reseed_count < max_events
                reasons: List[str] = []
                if triggers_allowed:
                    # Check each trigger condition and add reason if threshold exceeded
                    if not math.isnan(jump_pct) and jump_pct > float(triggers.get("centroid_jump_pct_of_width", math.inf)):
                        reasons.append("centroid_jump")  # Object moved too far
                    if not math.isnan(area_change_pct) and area_change_pct > float(
                        triggers.get("area_change_pct", math.inf)
                    ):
                        reasons.append("area_change")  # Mask size changed too much
                    if empty_streak >= int(triggers.get("empty_mask_consecutive", math.inf)):
                        reasons.append("empty_mask_consecutive")  # Too many consecutive empty frames

                # If triggers activated, perform reseeding: reinitialize tracking with new prompt
                if reasons:
                    reseed_count += 1
                    cooldown_remaining = cooldown_frames  # Start cooldown period
                    # Record reseeding event for logging and analysis
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

                    # Determine bounding box for reseeding: use current mask or fallback to GT/seed box
                    dynamic_box = bbox_from_mask(mask_bool, bbox_pad_px, tgt_w, tgt_h)
                    if dynamic_box is None:
                        # If mask is empty, use nearest ground truth box or fall back to seed box
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

                    # Add reseed prompt to inference session at current frame
                    processor.add_inputs_to_inference_session(
                        inference_session=session,
                        frame_idx=idx,
                        obj_ids=1,
                        input_boxes=reseed_boxes,
                        input_points=None,
                        input_labels=None,
                    )
                    # Re-run inference with new prompt to get improved mask
                    mask, runtime_extra = infer_frame(model, processor, session, idx, (tgt_w, tgt_h))
                    masks_full[idx] = mask
                    frame_runtimes[idx] = frame_runtimes.get(idx, 0.0) + runtime_extra
                    # Recompute mask statistics after reseeding
                    mask_bool = mask > 0
                    area = float(mask_bool.sum())
                    is_empty = area <= AREA_EPS
                    centroid = centroid_from_mask(mask_bool) if not is_empty else (math.nan, math.nan)
                    # Reset or adjust empty streak based on reseed result
                    if is_empty:
                        empty_streak = min(empty_streak, int(triggers.get("empty_mask_consecutive", 0)))
                    else:
                        empty_streak = 0
                elif cooldown_remaining > 0:
                    cooldown_remaining -= 1  # Decrement cooldown counter

                # Update tracking state for next iteration
                prev_area = area if area > AREA_EPS else prev_area
                prev_centroid = centroid if not math.isnan(centroid[0]) else prev_centroid

    # Generate overlay video: visualize tracking masks overlaid on original frames
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Morphological kernel for mask cleaning
    overlay_path = clip_dir / f"overlay_{clip_id}_{RUN_SPEC['run_id']}.mp4"
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
        bgr = overlay_mask(rgb, cleaned)  # Overlay mask on frame
        writer.write(bgr)
    writer.release()
    # Validate that video was written successfully
    if overlay_path.exists() and overlay_path.stat().st_size == 0:
        log(f"Overlay failed to write: {overlay_path}")
        log("If codecs are missing, try: brew install ffmpeg")
        raise SystemExit(1)

    # Compute metrics by comparing predictions with ground truth annotations
    label_paths = sorted(gt_dir.glob("frame_*.json"))
    frame_rows: List[Dict[str, object]] = []
    prev_metric_centroid = (math.nan, math.nan)  # Track centroid for jitter calculation

    # Process each ground truth frame to compute metrics
    for label_path in label_paths:
        frame_idx = int(label_path.stem.split("_")[-1])
        if frame_idx >= len(masks_full):
            continue  # Skip if frame index exceeds available frames
        src_w, src_h, polys = load_labelme_polys(label_path)
        # Skip frames with empty ground truth if configured
        if RUN_SPEC["metrics"]["scoring_policy"]["skip_frames_with_empty_gt"] and not polys:
            continue
        gt_mask = polys_to_mask(polys, src_w, src_h, tgt_w, tgt_h)  # Convert GT polygons to mask
        if RUN_SPEC["metrics"]["scoring_policy"]["skip_frames_with_empty_gt"] and not gt_mask.any():
            continue

        # Compare predicted mask with ground truth mask
        pred_mask = masks_full[frame_idx] > 0
        gt_mask_bool = gt_mask > 0

        # Compute IoU (Intersection over Union) and BIoU (Boundary IoU) metrics
        iou = compute_iou(pred_mask, gt_mask_bool)
        biou = compute_iou(band_mask(pred_mask), band_mask(gt_mask_bool))  # Boundary-focused IoU
        area = float(pred_mask.sum())  # Predicted mask area
        cx, cy = centroid_from_mask(pred_mask)  # Predicted centroid
        is_empty = 1 if area <= AREA_EPS else 0

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
            "runtime_ms": frame_runtimes.get(frame_idx, math.nan),
            "roi_w": tgt_w,
            "roi_h": tgt_h,
        }
        frame_rows.append(row)

    # Write per-frame metrics to CSV file for detailed analysis
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

    # Compute summary statistics across all frames
    ious = [row["iou"] for row in frame_rows]
    biou_vals = [row["biou"] for row in frame_rows]
    jitter_vals = [
        val
        for val in (row["centroid_shift_norm"] for row in frame_rows)
        if isinstance(val, float) and not math.isnan(val)
    ]
    area_vals = [row["area_px"] for row in frame_rows]
    empty_vals = [row["is_empty"] for row in frame_rows]

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

    # Log summary statistics to console and log file
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

    # Write parameters JSON file if configured
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

    # Write reseeding events to CSV file
    re_prompt_path = clip_dir / f"re_prompts_{clip_id}_{RUN_SPEC['run_id']}.csv"
    write_reprompt_csv(re_prompt_path, reseed_events)

    # Write log file with all processing messages
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

    # Return comprehensive results dictionary with all outputs and metrics
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
    # Write aggregated summary CSV file with metrics from all processed clips
    # This provides a high-level overview of tracking performance across all clips
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
    # Format boolean pass/fail value as string for logging
    return "PASS" if value else "FAIL"


def write_run_notes(
    outputs_root: Path,
    clip_results: Dict[str, Dict[str, object]],
) -> None:
    # Write run notes markdown file with summary information (currently commented out)
    # This function is prepared for generating human-readable run summaries
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_dir = outputs_root / timestamp
    ensure_dir(timestamped_dir)
    notes_path = timestamped_dir / "RUN_NOTES.md"
    now_iso = dt.datetime.now().isoformat(timespec="seconds")
    host = platform.node()  # Machine hostname
    python_version = platform.python_version()  # Python version

    #gentle = clip_results.get("clip_gentle", {})
    #fast = clip_results.get("clip_fast", {})

    #gentle_iou = gentle.get("iou_median", math.nan)
    #gentle_jitter = gentle.get("jitter_med", math.nan)
    #fast_iou = fast.get("iou_median", math.nan)

    # gentle_pass = (
    #     not math.isnan(gentle_iou)
    #     and gentle_iou >= RUN_SPEC["metrics"]["pass_fail"]["gentle"]["iou_median_min"]
    #     and not math.isnan(gentle_jitter)
    #     and gentle_jitter <= RUN_SPEC["metrics"]["pass_fail"]["gentle"]["jitter_norm_median_pct_max"]
    # )
    # fast_pass = (
    #     not math.isnan(fast_iou)
    #     and fast_iou >= RUN_SPEC["metrics"]["pass_fail"]["fast"]["iou_median_min"]
    # )

    # notes_lines = [
    #     "# Run Notes â€” SAM-2 Pilot (Fix-2)",
    #     f"Date/Time: {now_iso} | Machine: {host} | Python: {python_version}",
    #     f"Run ID: {RUN_SPEC['run_id']} | Output: {outputs_root}/{run_folder_name}",
    #     f"Model: {RUN_SPEC['model']['name']} (seed={RUN_SPEC['model']['random_seed']} | deterministic={RUN_SPEC['model']['deterministic']})",
    #     "Clips: clip_gentle, clip_fast (processed=2)",
    #     "Input width/stride: gentle=1280/1, fast=1280/1",
    #     "Seeding: earliest â€œbillboardâ€ frame; pad=6px; negatives=4 at 6px",
    #     "Auto re-prompt: centroid>5% | area>30% | emptyÃ—2; cooldown=3; max=20/30",
    #     "Artifacts: per_frame.csv, overlays/overlay.mp4, re_prompts.csv, params.json; run summary.csv",
    #     (
    #         "Result: gentle IoU_med={:.3f}, jitter_med={:.3f}%/frame | fast IoU_med={:.3f} "
    #         "(Pass/Fail gentle={}, fast={})"
    #     ).format(
    #         gentle_iou if not math.isnan(gentle_iou) else float("nan"),
    #         gentle_jitter if not math.isnan(gentle_jitter) else float("nan"),
    #         fast_iou if not math.isnan(fast_iou) else float("nan"),
    #         format_pass_fail(gentle_pass),
    #         format_pass_fail(fast_pass),
    #     ),
    # ]

    # with notes_path.open("w") as f:
    #     for line in notes_lines:
    #         f.write(f"{line}\n")


def main() -> None:
    # Main entry point: orchestrates the entire SAM-2 tracking pipeline
    # Handles initialization, model loading, clip processing, and result aggregation
    args = parse_args()

    # Set up run directory structure
    run_id = RUN_SPEC["run_id"]
    data_root = Path(args.data_root)
    runs_root = Path(args.runs_root)
    run_dir, run_folder_name = prepare_run_dir(runs_root, run_id)

    # Configure computation device and data type
    device = select_device(args.device)
    dtype = torch.float32

    # Set random seed for reproducibility
    set_random_state(RUN_SPEC["model"]["random_seed"])

    # Load SAM-2 model and processor from pretrained weights
    model = Sam2VideoModel.from_pretrained(args.weights).to(device=device, dtype=dtype)
    model.eval()  # Set model to evaluation mode (disable dropout, etc.)
    processor = Sam2VideoProcessor.from_pretrained(args.weights)

    # Create context dictionary with configuration for clip processing
    context = {
        "data_root": data_root,
        "device": device,
        "dtype": dtype,
        "weights": Path(args.weights),
        "reseed": bool(args.reseed),
    }

    # Capture version tracking information
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # Hash of script for reproducibility
    git_commit = get_git_commit(Path.cwd()) if RUN_SPEC["logging"].get("capture_git_commit", False) else "N/A"

    # Process clips: filter by selection if specified, otherwise process all
    selected_clips = set(args.clips) if args.clips else None
    clip_results: Dict[str, Dict[str, object]] = {}
    summaries: List[Dict[str, object]] = []

    # Process each clip configuration
    for clip_cfg in RUN_SPEC["clips"]:
        if selected_clips and clip_cfg["id"] not in selected_clips:
            continue  # Skip clips not in selection
        result = process_clip(clip_cfg, context, model, processor, run_dir, git_commit, script_hash)
        clip_results[clip_cfg["id"]] = result
        summaries.append(result["summary"])

    # Validate that at least one clip was processed
    if not summaries:
        raise SystemExit("No clips were processed")

    # Write aggregated summary CSV file
    summary_path = write_summary(run_dir, summaries)
    print(f"Summary saved to {summary_path}")

    # Optional: write run notes (currently commented out)
    # outputs_root = Path("outputs/sam2_pilot_fix2")
    # write_run_notes(outputs_root, clip_results)

    # Print summary status for each processed clip
    for clip_id, result in clip_results.items():
        status_line = (
            f"âœ" {clip_id}: IoU_med={result['iou_median']:.3f}, "
            f"BIoU_med={result['biou_median']:.3f}, "
            f"jitter_med={result['jitter_med']:.3f}%/frame, "
            f"empty={result['empty_pct']:.2f}%, "
            f"FPS={result['fps_measured']:.2f}"
        )
        print(status_line)


# Entry point: execute main function when script is run directly
if __name__ == "__main__":
    main()


