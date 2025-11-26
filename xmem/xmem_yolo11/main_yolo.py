# Standard library imports for argument parsing, file I/O, and system operations
import argparse
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# Third-party imports for image/video processing
import cv2  # OpenCV for video and image operations
import numpy as np  # NumPy for numerical array operations
from PIL import Image  # PIL for image format conversion
import time  # For timestamp generation

# Make sibling sam2 repo importable for YOLO autoprompt + shot detection
# This allows importing YOLO auto-prompting and shot detection utilities from the SAM-2 module
THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
SAM2_ROOT = ROOT / 'sam2'
if str(SAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM2_ROOT))

# Import YOLO auto-prompting utilities from SAM-2 module
from autoprompt_yolo import YoloBoxPromptor  # type: ignore
from shot_detection import detect_shots  # type: ignore

# Reuse local XMem helpers from main_gt module
# These functions handle frame extraction, XMem evaluation, and metrics computation
from main_gt import (
    extract_first_frames,
    run_xmem_eval,
    prepare_run_dir,
    write_overlay_video,
    compute_metrics_sam2_like,
)


def ensure_dir(p: Path) -> None:
    # Create directory and all parent directories if they don't exist
    # Safe to call multiple times (won't raise error if directory exists)
    p.mkdir(parents=True, exist_ok=True)


def save_palettized_mask(mask_bool: np.ndarray, out_png: Path) -> None:
    # Convert boolean mask to palettized PNG format required by XMem
    # XMem expects a specific palette format for seed masks
    ensure_dir(out_png.parent)
    binm = (mask_bool.astype(np.uint8) > 0).astype(np.uint8)  # Ensure binary 0/1
    if int(binm.sum()) == 0:
        raise SystemExit(f'Seed is empty for {out_png}')
    # Create palette: first color (0) = black, second color (1) = white, rest = black
    pal = [0, 0, 0, 255, 255, 255] + [0, 0, 0] * 254
    pm = Image.fromarray(binm, mode='P')  # Convert to palettized mode
    pm.putpalette(pal)  # Apply palette
    pm.save(out_png, optimize=False)  # Save without optimization


def draw_debug_boxes(img_bgr: np.ndarray, boxes: List[Tuple[int,int,int,int]], labels: List[str], scores: List[float]) -> np.ndarray:
    # Draw bounding boxes on image for visualization/debugging
    # First box is drawn in green, others in red
    out = img_bgr.copy()
    for k, (x0, y0, x1, y1) in enumerate(boxes):
        color = (0, 200, 0) if k == 0 else (0, 0, 255)  # Green for first, red for others
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), color, 2)
        # Add label and score text
        lab = labels[k] if k < len(labels) else ''
        sc = scores[k] if k < len(scores) else None
        txt = f'{lab} {float(sc):.2f}' if isinstance(sc, (int, float, np.floating)) else str(lab)
        if txt:
            cv2.putText(out, txt, (int(x0), max(0, int(y0) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out


def boxes_union_mask(H: int, W: int, boxes: List[Tuple[int,int,int,int]], erode_iter: int = 1) -> np.ndarray:
    # Create a binary mask from union of bounding boxes
    # Optionally erode the mask to shrink it slightly (useful for seed masks)
    mask = np.zeros((H, W), dtype=np.uint8)
    # Fill each box in the mask
    for (x0, y0, x1, y1) in boxes:
        # Clip coordinates to image boundaries
        x0 = max(0, min(W - 1, int(x0)))
        y0 = max(0, min(H - 1, int(y0)))
        x1 = max(0, min(W - 1, int(x1)))
        y1 = max(0, min(H - 1, int(y1)))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 1  # Fill box region
    # Optionally erode mask to shrink it (removes boundary pixels)
    if erode_iter and erode_iter > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.erode(mask, k, iterations=int(erode_iter))
    return mask


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
    for s in data.get('shapes', []):
        label = (s.get('label') or '').lower()
        if 'billboard' not in label:
            continue  # Skip non-billboard shapes
        pts = s.get('points', [])
        if len(pts) >= 3:  # Need at least 3 points for a polygon
            pts = np.array(pts, dtype=np.float32)
            # Clip points to image boundaries
            pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
            # Fill polygon in mask
            cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
            picked += 1
    return mask if picked > 0 else None  # Return mask only if at least one polygon was found


def main():
    # Main entry point: XMem wrapper with YOLO auto-prompting
    # This script uses YOLO to automatically detect objects at shot starts,
    # then uses those detections as seeds for XMem video segmentation
    
    ap = argparse.ArgumentParser(description='XMem wrapper with YOLO shot-start auto-prompting (union, single-ID)')
    # Basic I/O and processing arguments
    ap.add_argument('--root', default=r'./../', help='XMem project root containing data/, model/, work/ etc')
    ap.add_argument('--clip', help='Clip ID, e.g. clip_fast')
    ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])  # XMem requires CUDA
    ap.add_argument('--width', type=int, default=None, help='Resize width for extracted frames (keeps aspect)')
    ap.add_argument('--stride', type=int, default=1, help='Stride to report in summary (>=1)')
    ap.add_argument('--run-id', default='xmem_yolo', help='Run ID used for output folder and filenames')
    ap.add_argument('--run-notes', action='store_true', help='Write simple RUN_NOTES.md')
    ap.add_argument('--run-only', action='store_true', help='Skip metrics computation')

    # Shot detection arguments
    ap.add_argument('--shot-detect', action='store_true', help='Enable shot detection and per-shot seeding')
    ap.add_argument('--shot-method', choices=['adaptive', 'content'], default='adaptive')
    ap.add_argument('--shot-min-seconds', type=float, default=1.0)

    # YOLO auto-prompting arguments
    ap.add_argument('--auto-prompt', action='store_true',default=True, help='Enable YOLO shot-start seeding')
    ap.add_argument('--yolo-model', default=None, help='Path to YOLO weights (xmem_yolo11/weights/best.pt)')
    ap.add_argument('--yolo-conf', type=float, default=0.20, help='YOLO confidence threshold')
    ap.add_argument('--yolo-max-objects', type=int, default=3, help='Max YOLO boxes per shot start')
    ap.add_argument('--yolo-imgsz', type=int, default=None, help='Inference size for YOLO; defaults to --width or 1280')
    ap.add_argument('--autoprompt-fallback', choices=['gt', 'skip'], default='skip')  # Fallback if YOLO fails
    ap.add_argument('--seed-erosion', type=int, default=1)  # Erosion iterations for seed mask
    ap.add_argument('--yolo-debug', action='store_true')  # Enable debug visualization
    ap.add_argument('--skip-xmem', action='store_true', help='Skip XMem eval (for debugging seeds/boxes only)')

    args = ap.parse_args()
    if not args.clip:
        raise SystemExit('Please provide --clip (clip ID or filename).')

    # Parse clip argument: handle both clip ID and full filename
    clip_arg = str(args.clip)
    clip_path = Path(clip_arg)
    known_exts = {'.mp4', '.mov', '.m4v', '.avi'}
    clip_suffix = clip_path.suffix.lower()
    has_known_ext = clip_suffix in known_exts
    clip = clip_path.stem if has_known_ext else clip_arg  # Extract clip ID
    clip_filename = clip_path.name if has_known_ext else f'{clip}.mp4'  # Full filename

    # Set up output directory with timestamp
    ts = time.strftime('%Y%m%d_%H%M%S')
    run_id = args.run_id + '_' + ts

    # Resolve project root path (try multiple locations)
    root_arg = Path(args.root).expanduser()
    if root_arg.is_absolute():
        root = root_arg
    else:
        root = (Path.cwd() / root_arg)
    if not root.exists():
        alt = ROOT / root_arg
        if alt.exists():
            root = alt
        elif root_arg.name == ROOT.name:
            root = ROOT
    if not root.exists():
        raise SystemExit(f'Project root not found: {root}')
    root = root.resolve()

    # Resolve YOLO weights; prefer the local weights/ folder next to this script
    default_yolo_weights = THIS.parent / 'weights' / 'best.pt'
    resolved_yolo_weights = Path(args.yolo_model) if args.yolo_model else default_yolo_weights
    if not resolved_yolo_weights.is_file():
        raise SystemExit(f'YOLO weights not found: {resolved_yolo_weights}')

    # Find video file
    clips_root = root / 'data' / 'clips'
    mp4 = clips_root / clip_filename
    if not mp4.exists() and not has_known_ext:
        alt = clips_root / clip_arg  # Try alternative path
        if alt.exists():
            mp4 = alt
    if not mp4.exists() and clip_path.is_file():
        mp4 = clip_path.resolve()  # Try direct path if provided
    if not mp4.exists():
        raise SystemExit(f'Clip not found: {mp4}')

    # Prepare a full-clip frames dir once
    generic_base = root / 'work' / 'xmem_seq' / f'G_{clip}_full'
    frames_dir = generic_base / 'JPEGImages' / clip
    anns_dir = generic_base / 'Annotations' / clip  # not used for full clip
    ensure_dir(frames_dir)
    # Extract all frames (or resized to width)
    extract_first_frames(mp4, frames_dir, count=-1, width=args.width)

    # Get video FPS for overlay generation
    cap = cv2.VideoCapture(str(mp4))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 0.0
    cap.release()

    # Load frame information
    frame_paths = sorted(frames_dir.glob('*.jpg'))
    if not frame_paths:
        raise SystemExit(f'No frames extracted at {frames_dir}')
    total_frames = len(frame_paths)
    H, W = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR).shape[:2]  # Get frame dimensions

    # Shot detection: segment video into shots for independent processing
    if args.shot_detect:
        try:
            shots = detect_shots(mp4, total_frames=total_frames, fps=fps, method=args.shot_method, min_shot_len_s=args.shot_min_seconds)
            shot_bounds: List[Tuple[int, int]] = [(int(s.start), int(s.end)) for s in shots]
        except Exception:
            shot_bounds = [(0, total_frames)]  # Fallback: treat entire video as single shot
    else:
        shot_bounds = [(0, total_frames)]  # Single shot mode: process entire video as one shot

    # Initialize YOLO promptor for automatic object detection
    promptor: Optional[YoloBoxPromptor] = None
    promptor_error: Optional[str] = None
    yolo_class_names = None
    if args.auto_prompt:
        # Determine inference size: use provided value, or width, or default to 1280
        imgsz = args.yolo_imgsz if args.yolo_imgsz is not None else (args.width if args.width is not None else 1280)
        run_dev = 'cuda' if args.device == 'cuda' else 'cpu'
        # Check CUDA availability
        if run_dev == 'cuda':
            try:
                import torch  # type: ignore
                if not torch.cuda.is_available():
                    run_dev = 'cpu'  # Fallback to CPU if CUDA unavailable
            except Exception:
                run_dev = 'cpu'
        try:
            # Initialize YOLO model for object detection
            promptor = YoloBoxPromptor(
                model_path=os.fspath(resolved_yolo_weights),
                device=run_dev,
                conf=float(args.yolo_conf),  # Confidence threshold
                imgsz=int(imgsz),  # Inference image size
            )
            # Try to extract class names from model
            try:
                yolo_class_names = promptor.model.names
            except Exception:
                yolo_class_names = None
        except Exception as e:
            promptor_error = str(e)
            promptor = None  # Continue without YOLO if initialization fails

    # Prepare run-specific output folder (with timestamp if needed)
    runs_root = root / 'outputs' / f'G_{clip}' / 'runs'
    run_dir, run_folder = prepare_run_dir(runs_root, run_id)

    # Prepare per-clip log header with configuration information
    header_lines = [
        f'YOLO device: {args.device}; weights={resolved_yolo_weights}; conf={getattr(args, "yolo_conf", "N/A")}; max={getattr(args, "yolo_max_objects", "N/A")}',
        f'Shots detected: {len(shot_bounds)}'
    ]
    if yolo_class_names is not None:
        header_lines.append(f'YOLO classes: {yolo_class_names}')
    if args.auto_prompt and promptor is None:
        header_lines.append(f'YOLO promptor failed to initialize; no boxes will be produced. Error: {promptor_error or "unknown"}')

    # Set up logging: store messages for later writing to file
    log_path = run_dir / f'pilot_{clip}_{run_id}.log'
    log_lines: List[str] = []
    def _log(m: str) -> None:
        print(m)  # Print to console
        log_lines.append(m)  # Store for file output
    _log(f'Processing {mp4}')
    for ln in header_lines:
        _log(ln)

    # Set up CSV file to track re-prompt events (shot information and detected boxes)
    rp_path = run_dir / f're_prompts_{clip}_{run_id}.csv'
    rp_fields = ['shot_idx', 'start', 'end', 'obj_id', 'x0', 'y0', 'x1', 'y1', 'score', 'label', 'seed_type']
    with rp_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rp_fields)
        w.writeheader()

    # Set up output directory for masks
    masks_dst = run_dir / 'masks'
    ensure_dir(masks_dst)

    # Per-shot processing loop: process each shot independently
    last_eval_error: Optional[str] = None
    for i, (s, e) in enumerate(shot_bounds, start=1):
        # Clamp shot boundaries to valid frame range
        s = max(0, int(s)); e = min(total_frames, int(e))
        if s >= e:
            continue  # Skip empty shots
        
        # Load first frame of shot for YOLO detection
        frm_path = frames_dir / f'{s:05d}.jpg'
        bgr0 = cv2.imread(str(frm_path), cv2.IMREAD_COLOR)
        if bgr0 is None:
            continue  # Skip if frame can't be loaded

        # Run YOLO detection on shot start frame
        boxes_xyxy: List[Tuple[int,int,int,int]] = []
        labels: List[str] = []
        scores: List[float] = []
        seed_type = 'none'
        if promptor is not None:
            try:
                # Get up to max_objects detections from YOLO
                preds = promptor(bgr0, max_det=int(args.yolo_max_objects))
            except Exception:
                preds = []
            # Extract bounding boxes, labels, and scores
            for p in preds:
                x0, y0, x1, y1 = p.as_int_tuple(width=W, height=H)
                boxes_xyxy.append((x0, y0, x1, y1))
                labels.append(str(getattr(p, 'label', 'billboard')))
                scores.append(float(getattr(p, 'score', 0.0)))
            if boxes_xyxy:
                seed_type = 'yolo'  # Mark seed type as YOLO

        # Fallback to ground truth if YOLO found no boxes and fallback is enabled
        if not boxes_xyxy:
            if args.autoprompt_fallback == 'gt':
                # Search for ground truth annotation in shot frames
                gt_root = root / 'data' / 'gt_frames' / clip
                for fr in range(int(s), int(e)):
                    j6 = gt_root / f'frame_{fr:06d}.json'
                    j5 = gt_root / f'frame_{fr:05d}.json'
                    jp = j6 if j6.is_file() else (j5 if j5.is_file() else None)
                    if not jp:
                        continue
                    # Load GT mask and extract bounding box
                    gt_mask = load_labelme_billboard_mask(jp, W, H)
                    if gt_mask is None or int(gt_mask.sum()) == 0:
                        continue
                    # Compute bounding box from mask
                    ys, xs = np.where(gt_mask > 0)
                    if xs.size and ys.size:
                        x0, x1 = int(xs.min()), int(xs.max())
                        y0, y1 = int(ys.min()), int(ys.max())
                        boxes_xyxy = [(x0, y0, x1, y1)]
                        labels = ['gt']
                        scores = [1.0]
                        seed_type = 'gt'  # Mark seed type as ground truth
                        break
            # else skip shot if no boxes found

        # Record shot information in CSV (union box if multiple boxes)
        if boxes_xyxy:
            # Compute union bounding box of all detected boxes
            ux0 = min(b[0] for b in boxes_xyxy)
            uy0 = min(b[1] for b in boxes_xyxy)
            ux1 = max(b[2] for b in boxes_xyxy)
            uy1 = max(b[3] for b in boxes_xyxy)
            with rp_path.open('a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=rp_fields)
                w.writerow({
                    'shot_idx': i,
                    'start': s,
                    'end': e,
                    'obj_id': 1,
                    'x0': ux0, 'y0': uy0, 'x1': ux1, 'y1': uy1,
                    'score': (scores[0] if scores else ''),
                    'label': (labels[0] if labels else ''),
                    'seed_type': seed_type,
                })
        else:
            # Record shot with no detections
            with rp_path.open('a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=rp_fields)
                w.writerow({
                    'shot_idx': i,
                    'start': s,
                    'end': e,
                    'obj_id': 1,
                    'x0': '', 'y0': '', 'x1': '', 'y1': '',
                    'score': '',
                    'label': '',
                    'seed_type': 'none',
                })
            _log(f'Shot {i}/{len(shot_bounds)}: YOLO NONE - skip frames {s}-{e-1}')

        # Save debug image with boxes drawn (if debug mode enabled)
        if args.yolo_debug:
            try:
                dbg = draw_debug_boxes(bgr0, boxes_xyxy, labels, scores)
                out_dbg = run_dir / f'debug_shot_{i:03d}.jpg'
                cv2.imwrite(str(out_dbg), dbg)
            except Exception:
                pass

        # Save YOLO seed annotated JPG in run folder for visualization
        # Draws bounding boxes and confidence scores on the seed frame
        if boxes_xyxy:
            try:
                img_seed = bgr0.copy()
                for (x0, y0, x1, y1), sc in zip(boxes_xyxy, scores):
                    cv2.rectangle(img_seed, (int(x0), int(y0)), (int(x1), int(y1)), (0, 200, 0), 2)  # Green rectangle
                    tag = f"billboard {sc:.2f}"  # Label with confidence score
                    # Draw text with black outline and white fill for visibility
                    cv2.putText(img_seed, tag, (int(x0), max(0, int(y0) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)  # Black outline
                    cv2.putText(img_seed, tag, (int(x0), max(0, int(y0) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)  # White fill
                cv2.imwrite(str(run_dir / f'shot_{i:03d}_seed.jpg'), img_seed)
            except Exception:
                pass
            # Log seeding summary
            try:
                _log("Seeded {} boxes at shot {} with scores=".format(len(boxes_xyxy), s) + ",".join([f"{x:.2f}" for x in scores]))
            except Exception:
                _log(f'Seeded {len(boxes_xyxy)} boxes at shot {s}')

        # Skip shot if no boxes found (can't create seed mask)
        if not boxes_xyxy:
            continue

        # Build shot-specific Generic dataset at a temp path
        # Each shot gets its own XMem evaluation with its own seed mask
        shot_base = root / 'work' / 'xmem_seq' / f'G_{clip}_shot_{i:03d}'
        shot_frames = shot_base / 'JPEGImages' / clip
        shot_anns = shot_base / 'Annotations' / clip
        # Clean up previous shot data if it exists
        if shot_base.exists():
            try:
                shutil.rmtree(shot_base)
            except Exception:
                pass
        ensure_dir(shot_frames); ensure_dir(shot_anns)

        # Copy frames [s, e) into shot_frames as 00000.jpg, 00001.jpg, etc.
        # XMem expects frames to be numbered starting from 00000
        for idx, fr in enumerate(range(int(s), int(e))):
            src = frames_dir / f'{fr:05d}.jpg'  # Source frame with global index
            dst = shot_frames / f'{idx:05d}.jpg'  # Destination with local index (0, 1, 2, ...)
            try:
                shutil.copy2(src, dst)
            except Exception:
                break

        # Create union seed mask from all detected boxes at 00000.png
        # This is the seed mask that XMem will use for tracking
        seed_mask = boxes_union_mask(H, W, boxes_xyxy, erode_iter=int(args.seed_erosion))
        try:
            save_palettized_mask(seed_mask, shot_anns / '00000.png')
        except Exception as e:
            print('Failed to save seed:', e)
            continue  # Skip shot if seed can't be saved

        # Run XMem evaluation for this shot and stitch masks back into run_dir/masks
        if args.skip_xmem:
            _log(f'Skipping XMem for shot {i} (--skip-xmem).')
            continue  # Skip XMem evaluation (useful for debugging seeds only)

        try:
            # Run XMem evaluation on shot-specific dataset
            _, vendor_clip_dir = run_xmem_eval(root, clip, device=args.device, generic_path=shot_base)
            # Copy masks from XMem output back to global mask directory
            # Map local frame indices back to global indices
            for p in sorted(vendor_clip_dir.glob('*.png')):
                try:
                    local_idx = int(p.stem)  # Local index within shot (0, 1, 2, ...)
                except Exception:
                    continue
                global_idx = s + local_idx  # Convert to global frame index
                out = masks_dst / f'{global_idx:05d}.png'  # Save with global index
                shutil.copy2(p, out)
        except SystemExit as e:
            last_eval_error = str(e)
            _log(f'XMem eval failed for shot {i} -> {e}')
            continue  # Continue with next shot if evaluation fails

    # Early exit if XMem was skipped (debugging mode)
    if args.skip_xmem:
        _log('Skipping XMem eval; seeds and prompt CSV are written. No overlay/metrics.')
        return

    # Validate that masks were generated
    mask_files = sorted(masks_dst.glob('*.png'))
    if not mask_files:
        reason = last_eval_error or 'no masks were produced by XMem'
        raise SystemExit(f"Aborting: no masks generated for {clip}. Last error: {reason}")

    # Generate overlay video showing masks on original frames
    write_overlay_video(frames_dir, masks_dst, run_dir / f'overlay_{clip}_{run_id}.mp4', fps=fps if fps > 0 else 10.0, alpha=0.35)

    # Compute metrics in SAM-2 format (if not skipped)
    summaries: List[Dict[str, object]] = []
    if not args.run_only:
        per_frame_path, summary = compute_metrics_sam2_like(root, clip, frames_dir, masks_dst, stride=int(args.stride), fps_video=fps, run_dir=run_dir, run_id=run_id)
        print(f'Per-frame saved to: {per_frame_path}')
        summaries.append(summary)
    
    # Write summary CSV with aggregated metrics
    if summaries:
        # Print summary header to console
        for ln in header_lines:
            print(ln)
        # Write summary CSV
        from main_gt import write_summary  # local import to avoid circular
        summary_path = write_summary(run_dir, summaries, run_id=run_id)
        print(f'Summary saved to {summary_path}')

    # Write log file with all processing messages
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            for ln in log_lines:
                f.write(ln + '\n')
        print(f'Log saved to {log_path}')
    except Exception:
        pass

    # Write run notes if requested
    if args.run_notes:
        out_notes = run_dir / 'RUN_NOTES.md'
        with out_notes.open('w', encoding='utf-8') as f:
            f.write('# XMem YOLO Auto-Prompt Run\n')
            f.write(f'Clip: {clip}\nRun folder: {run_dir}\n')
            for ln in header_lines:
                f.write(ln + '\n')
            if summaries:
                s = summaries[0]
                f.write(f"IoU_med={s['iou_median']:.3f}, BIoU_med={s['biou_median']:.3f}, jitter_med={s['jitter_norm_median_pct']:.3f}%/frame\n")


if __name__ == '__main__':
    # Script entry point: run main function when executed directly
    main()
