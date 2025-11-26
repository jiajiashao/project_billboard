# Standard library imports for argument parsing, file I/O, and system operations
import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# Third-party imports for image/video processing
import cv2  # OpenCV for video and image operations
import numpy as np  # NumPy for numerical array operations
from PIL import Image  # PIL for image format conversion

# XMem local runner aligned to SAM-2 output metrics/structure
# This script runs XMem (XMem is a video object segmentation model) and formats outputs
# to match SAM-2's metric structure for easy comparison
# - Writes per-frame CSV with SAM-2 fields
# - Writes summary CSV with SAM-2 fields
# - Creates a separate output folder per run for each clip


def ensure_dir(p: Path) -> None:
    # Create directory and all parent directories if they don't exist
    # Safe to call multiple times (won't raise error if directory exists)
    p.mkdir(parents=True, exist_ok=True)


def extract_first_frames(mp4: Path, out_dir: Path, count: Optional[int] = None, width: Optional[int] = None) -> List[Path]:
    # Extract frames from video file and save as JPEG images
    # Used to prepare input frames for XMem evaluation
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {mp4}")
    frames: List[Path] = []
    idx = 0
    while True:
        # Stop if we've extracted the requested number of frames
        if (count is not None) and (int(count) >= 0) and (idx >= int(count)):
            break
        ok, frame = cap.read()
        if not ok:
            break  # End of video
        # Optionally resize frames to target width (maintains aspect ratio)
        if width is not None and width > 0:
            h, w = frame.shape[:2]
            s = width / max(1, w)  # Scale factor
            frame = cv2.resize(frame, (int(round(w*s)), int(round(h*s))), interpolation=cv2.INTER_AREA)
        # Save frame as JPEG with zero-padded index
        fn = out_dir / f"{idx:05d}.jpg"
        cv2.imwrite(str(fn), frame)
        frames.append(fn)
        idx += 1
    cap.release()
    if (count is not None) and (int(count) >= 0) and (len(frames) < int(count)):
        print(f"Warning: only extracted {len(frames)} frames from {mp4}")
    return frames


def write_palettized_seed(src_png: Path, out_png: Path) -> None:
    # Convert seed mask to palettized PNG format required by XMem
    # XMem expects a specific palette format for seed masks
    ensure_dir(out_png.parent)
    m = np.array(Image.open(src_png).convert("L"))  # Load as grayscale
    binm = (m > 127).astype(np.uint8)  # Binarize: strict 0/1 (threshold at 127)
    if binm.sum() == 0:
        raise SystemExit(f"Seed is empty: {src_png}")
    # Create palette: first color (0) = black, second color (1) = white, rest = black
    pal = [0,0,0, 255,255,255] + [0,0,0]*254
    pm = Image.fromarray(binm, mode="P")  # Convert to palettized mode
    pm.putpalette(pal)  # Apply palette
    pm.save(out_png, optimize=False)  # Save without optimization


def run_xmem_eval(
    root: Path,
    clip: str,
    device: str = "cuda",
    size: int = 480,
    mem_every: int = 5,
    generic_path: Optional[Path] = None,
    xmem_root: Optional[Path] = None,
) -> Tuple[Path, Path]:
    # Run XMem evaluation as a subprocess
    # XMem is a video object segmentation model that requires CUDA GPU
    # This function calls the XMem eval.py script with appropriate parameters
    
    # XMem only supports CUDA; bail early with a clear message if CUDA is not available
    if str(device).lower() != "cuda":
        raise SystemExit("XMem eval currently supports only CUDA devices; run on a CUDA GPU box")
    # Check if CUDA is actually available
    try:
        import torch  # type: ignore
        has_cuda = bool(torch.cuda.is_available())
    except Exception:
        has_cuda = False
    if not has_cuda:
        raise SystemExit("XMem eval requires a CUDA-capable GPU (torch.cuda.is_available() is False)")

    # Resolve XMem repository path (defaults to root/model/xmem if not specified)
    resolved_xmem_root = (Path(xmem_root).expanduser() if xmem_root else (root / "model" / "xmem")).resolve()
    model_path = resolved_xmem_root / "saves" / "XMem-s012.pth"
    if not model_path.exists():
        raise SystemExit(f"Model checkpoint not found: {model_path}")

    # Set up input/output paths for XMem evaluation
    generic_path = (root / "work" / "xmem_seq" / f"G_{clip}") if generic_path is None else generic_path
    out_vendor = root / "outputs" / f"G_{clip}"
    ensure_dir(out_vendor)

    # Build command to run XMem eval.py script
    eval_script = (resolved_xmem_root / "eval.py").resolve()
    cmd = [
        os.fspath(Path(os.sys.executable)),  # Python interpreter
        os.fspath(eval_script),  # XMem eval script
        "--model", os.fspath(model_path),  # Model checkpoint path
        "--dataset", "G",  # Generic dataset format
        "--generic_path", os.fspath(generic_path),  # Input frames directory
        "--size", str(size),  # Processing size
        "--mem_every", str(mem_every),  # Memory update frequency
        "--output", os.fspath(out_vendor),  # Output directory
    ]
    print("CMD:", " ".join(cmd))
    try:
        # Run XMem evaluation in subprocess
        subprocess.run(cmd, cwd=os.fspath(resolved_xmem_root), check=True)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"XMem eval failed with exit code {e.returncode}")

    return out_vendor, out_vendor / clip  # Return output root and clip-specific directory


def load_labelme_billboard_mask(json_path: Path, W: int, H: int) -> Optional[np.ndarray]:
    # Load ground truth mask from LabelMe JSON annotation file
    # Extracts polygons labeled as "billboard" and converts to binary mask
    try:
        data = json.loads(json_path.read_text())
    except Exception:
        return None  # Return None if JSON parsing fails
    mask = np.zeros((H, W), np.uint8)
    picked = 0
    # Iterate through all shapes in the annotation
    for s in data.get("shapes", []):
        label = (s.get("label") or "").lower()
        if "billboard" not in label:
            continue  # Skip non-billboard shapes
        pts = s.get("points", [])
        if len(pts) >= 3:  # Need at least 3 points for a polygon
            pts = np.array(pts, dtype=np.float32)
            # Clip points to image boundaries
            pts[:,0] = np.clip(pts[:,0], 0, W-1)
            pts[:,1] = np.clip(pts[:,1], 0, H-1)
            # Fill polygon in mask
            cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
            picked += 1
    return mask if picked > 0 else None  # Return mask only if at least one polygon was found


def boundary(mask: np.ndarray, r: int = 3) -> np.ndarray:
    # Extract boundary pixels from a binary mask using morphological operations
    # Returns a mask where only boundary pixels (within radius r) are set
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))  # Elliptical kernel
    dil = cv2.dilate(mask, k)  # Dilated mask
    ero = cv2.erode(mask, k)  # Eroded mask
    return (dil ^ ero)  # XOR gives boundary pixels


def centroid_from_mask(mask_bool: np.ndarray) -> Tuple[float, float]:
    # Calculate centroid (center of mass) of a binary mask
    # Returns (x, y) coordinates or (NaN, NaN) if mask is empty
    ys, xs = np.nonzero(mask_bool)  # Get coordinates of non-zero pixels
    if len(xs) == 0:
        return (math.nan, math.nan)  # Empty mask
    cx = float(xs.mean())  # Mean x coordinate
    cy = float(ys.mean())  # Mean y coordinate
    return (cx, cy)


def nan_percentiles(values: List[float], percents: List[int]) -> Tuple[float, ...]:
    # Calculate percentiles while ignoring NaN values
    # Returns tuple of percentiles corresponding to quantiles in percents
    arr = np.array(values, dtype=np.float32)
    if arr.size == 0:
        return tuple(float("nan") for _ in percents)
    return tuple(float(np.nanpercentile(arr, p)) for p in percents)


def nan_mean(values: List[float]) -> float:
    # Calculate mean while ignoring NaN values
    # Returns mean or NaN if array is empty
    arr = np.array(values, dtype=np.float32)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def prepare_run_dir(root: Path, run_id: str) -> Tuple[Path, str]:
    # Create a unique run directory with timestamp suffix if directory already exists
    # Prevents overwriting previous runs
    ensure_dir(root)
    candidate = root / run_id
    if not candidate.exists():
        candidate.mkdir()
        return candidate, candidate.name  # Use requested name if available
    # Add timestamp suffix if directory exists
    ts = time.strftime('%Y%m%d_%H%M%S')
    candidate = root / f"{run_id}_{ts}"
    candidate.mkdir()
    return candidate, candidate.name


def write_per_frame_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    # Write per-frame metrics to CSV file
    # Used to store detailed metrics for each frame (IoU, BIoU, centroid, etc.)
    ensure_dir(path.parent)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def compute_metrics_sam2_like(root: Path, clip: str, frames_dir: Path, mask_dir: Path, stride: int, fps_video: float, run_dir: Path, run_id: str) -> Tuple[Path, Dict[str, object]]:
    # Compute metrics in SAM-2 format for comparison
    # This function aligns XMem outputs with SAM-2 metric structure (IoU, BIoU, jitter, etc.)
    gt_dir = root / "data" / "gt_frames" / clip
    if not frames_dir.is_dir():
        raise SystemExit(f"Frames folder not found: {frames_dir}")
    if not mask_dir.is_dir():
        raise SystemExit(f"Mask folder not found: {mask_dir}")

    # Load frame and mask files
    frames = sorted(frames_dir.glob("*.jpg"))
    masks = sorted(mask_dir.glob("*.png"))
    if not frames or not masks:
        raise SystemExit("Missing frames or masks for metrics.")

    # Create mapping from stem (filename without extension) to file path
    # This allows matching frames with their corresponding masks
    def stem(p: Path) -> str: return p.stem
    frame_map = {stem(p): p for p in frames}
    mask_map  = {stem(p): p for p in masks}
    indices = sorted(set(frame_map).intersection(mask_map))  # Find common indices
    if not indices:
        raise SystemExit("No matching frame/mask stems.")

    # Get frame dimensions from first frame
    H, W = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR).shape[:2]

    frame_rows: List[Dict[str, object]] = []
    prev_centroid = (math.nan, math.nan)  # Track previous centroid for jitter calculation

    # Compute per-frame metrics for each frame with both prediction and ground truth
    for idx in indices:
        # Try to find ground truth JSON file (supports both 5-digit and 6-digit frame numbering)
        j6 = gt_dir / f"frame_{idx.zfill(6)}.json"
        j5 = gt_dir / f"frame_{idx.zfill(5)}.json"
        gt_json = j6 if j6.is_file() else (j5 if j5.is_file() else None)
        if not gt_json:
            continue  # Skip frames without ground truth
        
        # Load ground truth mask from LabelMe JSON
        gt = load_labelme_billboard_mask(gt_json, W, H)
        if gt is None or int(gt.sum()) == 0:
            continue  # Skip empty ground truth masks
        
        # Load predicted mask from XMem output
        pred = cv2.imread(str(mask_map[idx]), cv2.IMREAD_UNCHANGED)
        if pred is None:
            continue  # Skip if mask file can't be read
        # Handle multi-channel masks (take first channel if needed)
        if pred.ndim == 3:
            pred = pred[...,0]
        pred_bool = (pred > 0)  # Convert to boolean mask
        gt_bool = (gt > 0)  # Convert to boolean mask
        
        # Compute IoU (Intersection over Union): standard segmentation metric
        inter = int(np.logical_and(gt_bool, pred_bool).sum())  # Intersection pixels
        union = int(np.logical_or (gt_bool, pred_bool).sum())  # Union pixels
        iou = (inter/union) if union>0 else 0.0
        
        # Compute BIoU (Boundary IoU): IoU computed only on boundary pixels
        # This metric focuses on boundary accuracy rather than overall mask overlap
        b_gt   = boundary(gt_bool.astype(np.uint8), 3)  # Ground truth boundary
        b_pred = boundary(pred_bool.astype(np.uint8), 3)  # Predicted boundary
        b_inter = int(np.logical_and(b_gt>0, b_pred>0).sum())  # Boundary intersection
        b_union = int(np.logical_or (b_gt>0, b_pred>0).sum())  # Boundary union
        biou = (b_inter/b_union) if b_union>0 else 0.0

        # Compute mask area and check if empty
        area = float(pred_bool.sum())
        is_empty = 1 if area <= 1.0 else 0
        cx, cy = centroid_from_mask(pred_bool)  # Calculate centroid

        # Calculate jitter: frame-to-frame centroid movement (indicates tracking stability)
        shift_px = math.nan
        shift_norm = math.nan
        if not math.isnan(cx) and not math.isnan(prev_centroid[0]):
            dx = cx - prev_centroid[0]
            dy = cy - prev_centroid[1]
            shift_px = float(math.hypot(dx, dy))  # Euclidean distance
            shift_norm = (shift_px / max(1.0, float(W))) * 100.0  # Normalized to frame width percentage
        # Update previous centroid for next iteration
        if not math.isnan(cx):
            prev_centroid = (cx, cy)
        else:
            prev_centroid = (math.nan, math.nan)

        # Store per-frame metrics in SAM-2 format
        row = {
            'clip_id': clip,
            'frame_no': int(idx),
            'iou': float(iou),
            'biou': float(biou),
            'area_px': float(area),
            'centroid_x': float(cx) if not math.isnan(cx) else math.nan,
            'centroid_y': float(cy) if not math.isnan(cy) else math.nan,
            'centroid_shift_px': shift_px,
            'centroid_shift_norm': shift_norm,
            'is_empty': is_empty,
            'runtime_ms': math.nan,  # XMem runtime not tracked per-frame
            # ROI columns omitted to match SAM-2 uniform outputs
        }
        frame_rows.append(row)

    # Write per-frame CSV with all computed metrics
    per_frame_path = run_dir / f"per_frame_{clip}_{run_id}.csv"
    per_frame_cols = [
        'clip_id','frame_no','iou','biou','area_px','centroid_x','centroid_y',
        'centroid_shift_px','centroid_shift_norm','is_empty','runtime_ms'
    ]
    write_per_frame_csv(per_frame_path, frame_rows, per_frame_cols)

    # Summary aggregation: compute statistics across all frames
    ious = [r['iou'] for r in frame_rows]
    bious = [r['biou'] for r in frame_rows]
    jitter_vals = [val for val in (r['centroid_shift_norm'] for r in frame_rows) if isinstance(val, float) and not math.isnan(val)]
    area_vals = [r['area_px'] for r in frame_rows]
    empty_vals = [r['is_empty'] for r in frame_rows]

    # Calculate percentiles for IoU, BIoU, and jitter
    iou_p25, iou_med, iou_p75 = nan_percentiles(ious, [25, 50, 75])
    biou_p25, biou_med, biou_p75 = nan_percentiles(bious, [25, 50, 75])
    jitter_med, jitter_p95 = nan_percentiles(jitter_vals, [50, 95]) if jitter_vals else (float('nan'), float('nan'))

    # Calculate area coefficient of variation: measures consistency of mask size
    area_mean = nan_mean(area_vals)
    area_std = float('nan')
    if not math.isnan(area_mean):
        arr = np.array(area_vals, dtype=np.float32)
        area_std = float(np.std(arr))
    area_cv = float('nan') if math.isnan(area_mean) or area_mean == 0 else (area_std / area_mean) * 100.0
    # Calculate percentage of empty frames
    empty_pct = (sum(empty_vals) / len(empty_vals) * 100.0) if empty_vals else float('nan')

    # Calculate processing FPS metrics
    stride = max(1, int(stride))
    fps_theoretical = (float(fps_video) / float(stride)) if fps_video and fps_video > 0 else 0.0
    fps_measured = 0.0  # XMem eval is external; per-frame runtime unavailable here

    # Create summary dictionary matching SAM-2 format
    summary = {
        'clip_id': clip,
        'iou_median': iou_med,
        'iou_p25': iou_p25,
        'iou_p75': iou_p75,
        'biou_median': biou_med,
        'biou_p25': biou_p25,
        'biou_p75': biou_p75,
        'jitter_norm_median_pct': jitter_med,
        'jitter_norm_p95_pct': jitter_p95,
        'area_cv_pct': area_cv,
        'empty_frames_pct': empty_pct,
        'proc_fps_measured': fps_measured,
        'proc_fps_theoretical': fps_theoretical,
        'target_W': int(W),
        'target_H': int(H),
        'stride': stride,
    }

    return per_frame_path, summary


def write_overlay_video(frames_dir: Path, masks_dir: Path, out_mp4: Path, fps: float = 10.0, alpha: float = 0.35) -> Optional[Path]:
    # Generate overlay video showing masks overlaid on original frames
    # Masks are blended with original frames using alpha transparency
    try:
        frames = sorted(frames_dir.glob('*.jpg'))
        if not frames:
            print(f'No frames in {frames_dir} for overlay.')
            return None
        # Read first frame to get dimensions
        first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
        if first is None:
            print(f'Failed to read first frame for overlay: {frames[0]}')
            return None
        H, W = first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_mp4), fourcc, float(fps), (W, H))
        color = (170, 0, 255)  # BGR magenta color for mask overlay
        
        # Process each frame and overlay corresponding mask
        for fp in frames:
            frm = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            if frm is None:
                continue
            # Find corresponding mask file
            mp = masks_dir / (fp.stem + '.png')
            m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
            if m is None:
                writer.write(frm)  # Write frame without mask if mask not found
                continue
            # Handle multi-channel masks
            if m.ndim == 3:
                m = m[...,0]
            mask = (m > 0)  # Convert to boolean mask
            # Create overlay: colorize mask pixels
            overlay = frm.copy()
            overlay[mask] = color
            # Blend overlay with original frame using alpha transparency
            blended = cv2.addWeighted(overlay, alpha, frm, 1.0 - alpha, 0)
            writer.write(blended)
        writer.release()
        print(f'Overlay saved to: {out_mp4}')
        return out_mp4
    except Exception as e:
        print(f'Overlay generation failed: {e}')
        return None


def write_summary(run_dir: Path, summaries: List[Dict[str, object]], run_id: str) -> Path:
    # Write summary CSV with aggregated metrics across all clips
    # Format matches SAM-2 summary structure for easy comparison
    summary_path = run_dir / f"summary_{run_id}.csv"
    fields = [
        'clip_id',
        'iou_median','iou_p25','iou_p75',
        'biou_median','biou_p25','biou_p75',
        'jitter_norm_median_pct','jitter_norm_p95_pct',
        'area_cv_pct','empty_frames_pct',
        'proc_fps_measured','proc_fps_theoretical',
        # ROI removed to mirror SAM-2 uniform outputs
        'target_W','target_H','stride',
    ]
    with summary_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in summaries:
            w.writerow({k: row.get(k, '') for k in fields})  # Only write fields that exist
    return summary_path


def main():
    # Main entry point: orchestrates XMem evaluation pipeline
    # 1. Extract frames from video
    # 2. Prepare seed mask
    # 3. Run XMem evaluation
    # 4. Compute metrics in SAM-2 format
    # 5. Generate overlay video and summary outputs
    
    ap = argparse.ArgumentParser(description="Run XMem locally with SAM-2-like outputs")
    ap.add_argument('--root', default="./../", help='Project root containing data/, model/ etc')
    ap.add_argument('--clip', default='clip_fast', help='Clip ID, e.g. clip_fast')
    ap.add_argument('--device', default='cuda', choices=['cuda','cpu'])  # XMem requires CUDA
    ap.add_argument('--frames', type=int, default=-1, help='How many initial frames to prepare (-1 = all)')
    ap.add_argument('--width', type=int, default=None, help='Resize width for extracted frames (keeps aspect)')
    ap.add_argument('--stride', type=int, default=1, help='Stride to report in summary (>=1)')
    ap.add_argument('--run-id', default='xmem', help='Run ID used for output folder and filenames')
    ap.add_argument('--run-notes', action='store_true', help='Write simple RUN_NOTES.md')
    ap.add_argument('--run-only', action='store_true', help='Skip metrics computation')
    ap.add_argument('--xmem-root', type=str, default=None, help='Override path to XMem repo (defaults to <root>/model/xmem)')
    args = ap.parse_args()

    # Resolve paths
    root = Path(args.root).expanduser().resolve()
    xmem_root_override = Path(args.xmem_root).expanduser().resolve() if args.xmem_root else None
    clip = args.clip

    # Define input/output paths
    mp4 = root / 'data' / 'clips' / f'{clip}.mp4'
    seed_src = root / 'data' / 'seeds' / clip / '00000.png'
    generic_base = root / 'work' / 'xmem_seq' / f'G_{clip}'
    frames_dir = generic_base / 'JPEGImages' / clip
    anns_dir   = generic_base / 'Annotations' / clip

    # Validate input files exist
    if not mp4.exists():
        raise SystemExit(f"Clip not found: {mp4}")
    if not seed_src.exists():
        raise SystemExit(f"Seed not found: {seed_src}")

    # Step 1: Prepare Generic dataset format for XMem
    # Extract frames from video and convert seed mask to XMem format
    print('Preparing Generic dataset...')
    extract_first_frames(mp4, frames_dir, count=int(args.frames), width=args.width)
    write_palettized_seed(seed_src, anns_dir / '00000.png')

    # Step 2: Run XMem evaluation as subprocess
    print('Running XMem eval...')
    vendor_root, vendor_clip_dir = run_xmem_eval(
        root,
        clip,
        device=args.device,
        generic_path=generic_base,
        xmem_root=xmem_root_override,
    )
    print('Vendor out:', vendor_clip_dir)

    # Step 3: Prepare run-specific output folder (with timestamp if needed)
    runs_root = root / 'outputs' / f'G_{clip}' / 'runs'
    run_dir, run_folder = prepare_run_dir(runs_root, args.run_id)

    # Step 4: Copy masks into run dir to snapshot results
    # This preserves the output masks for this specific run
    masks_src = vendor_clip_dir
    masks_dst = run_dir / 'masks'
    ensure_dir(masks_dst)
    for p in sorted(masks_src.glob('*.png')):
        shutil.copy2(p, masks_dst / p.name)

    # Step 5: Generate overlay video showing masks on original frames
    # Try to read FPS from source video
    cap = cv2.VideoCapture(str(mp4))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 0.0
    cap.release()
    write_overlay_video(frames_dir, masks_dst, run_dir / f'overlay_{clip}_{args.run_id}.mp4', fps=fps if fps>0 else 10.0, alpha=0.35)

    # Step 6: Compute metrics in SAM-2 format (if not skipped)
    summaries: List[Dict[str, object]] = []
    if not args.run_only:
        per_frame_path, summary = compute_metrics_sam2_like(root, clip, frames_dir, masks_dst, stride=int(args.stride), fps_video=fps, run_dir=run_dir, run_id=args.run_id)
        print(f'Per-frame saved to: {per_frame_path}')
        summaries.append(summary)

    # Step 7: Write summary CSV with aggregated metrics
    if summaries:
        summary_path = write_summary(run_dir, summaries, run_id=args.run_id)
        print(f'Summary saved to {summary_path}')

    # Step 8: Write run notes if requested
    if args.run_notes:
        out_notes = run_dir / 'RUN_NOTES.md'
        with out_notes.open('w', encoding='utf-8') as f:
            f.write(f'# XMem Run Notes\n')
            f.write(f'Clip: {clip}\n')
            f.write(f'Run folder: {run_dir}\n')
            if summaries:
                s = summaries[0]
                f.write(f"IoU_med={s['iou_median']:.3f}, BIoU_med={s['biou_median']:.3f}, jitter_med={s['jitter_norm_median_pct']:.3f}%/frame\n")


if __name__ == '__main__':
    # Script entry point: run main function when executed directly
    main()