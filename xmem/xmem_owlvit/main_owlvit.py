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

# Make sibling sam2 repo importable for OWL-ViT + shot detection
# This allows importing OWL-ViT auto-prompting and shot detection utilities from the SAM-2 module
THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
SAM2_ROOT = ROOT / 'sam2'
if str(SAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM2_ROOT))

# Import OWL-ViT auto-prompting utilities from SAM-2 module
from autoprompt_owlvit import OwlVitBoxPromptor, PROMPT_TERMS  # type: ignore
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


def _prompts_from_args(args) -> List[str]:
    # Extract prompt terms for OWL-ViT from command-line arguments
    # Prompts can come from: direct string argument, file, or default terms
    if getattr(args, 'prompts', None):
        # Parse comma or semicolon-separated prompts from command line
        raw = str(args.prompts)
        sep = ';' if ';' in raw else ','  # Support both separators
        return [s.strip() for s in raw.split(sep) if s.strip()]
    if getattr(args, 'prompts_file', None):
        # Read prompts from file (one per line, comments start with #)
        p = Path(args.prompts_file)
        if p.exists():
            try:
                return [ln.strip() for ln in p.read_text(encoding='utf-8').splitlines() if ln.strip() and not ln.lstrip().startswith('#')]
            except Exception:
                pass
    # Fallback to default prompt terms from autoprompt_owlvit module
    return list(PROMPT_TERMS)


def main():
    # Main entry point: XMem wrapper with OWL-ViT auto-prompting
    # This script uses OWL-ViT to automatically detect objects at shot starts,
    # then uses those detections as seeds for XMem video segmentation
    
    ap = argparse.ArgumentParser(description='XMem wrapper with OWL-ViT shot-start auto-prompting (v5-matched)')
    # Basic I/O and processing arguments
    ap.add_argument('--root', default='./../', help='XMem project root containing data/, model/, work/ etc')
    ap.add_argument('--clip', help='Clip ID, e.g. clip_fast')
    ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])  # XMem requires CUDA
    ap.add_argument('--width', type=int, default=None, help='Resize width for extracted frames (keeps aspect)')
    ap.add_argument('--stride', type=int, default=1, help='Stride to report in summary (>=1)')
    ap.add_argument('--run-id', help='Run ID used for output folder and filenames')
    ap.add_argument('--run-notes', action='store_true', help='Write simple RUN_NOTES.md')
    ap.add_argument('--run-only', action='store_true', help='Skip metrics computation')

    # Shot detection arguments
    ap.add_argument('--shot-detect', action='store_true', help='Enable shot detection and per-shot seeding')
    ap.add_argument('--shot-method', choices=['adaptive', 'content'], default='adaptive')
    ap.add_argument('--shot-min-seconds', type=float, default=1.0)

    # OWL-ViT auto-prompting arguments
    ap.add_argument('--auto-prompt', default = True, action='store_true', help='Enable OWL-ViT shot-start seeding')
    ap.add_argument('--owlvit-model', default='google/owlv2-base-patch16-ensemble')
    ap.add_argument('--owlvit-device', default=None)
    ap.add_argument('--owlvit-score-thr', type=float, default=0.15)  # Minimum confidence threshold
    ap.add_argument('--prompts-file', default=None)  # File containing prompt terms
    ap.add_argument('--prompts', type=str, default=None, help='Comma or semicolon separated prompts')
    ap.add_argument('--autoprompt-fallback', choices=['gt', 'skip'], default='skip')  # Fallback if OWL-ViT fails
    ap.add_argument('--seed-erosion', type=int, default=1)  # Erosion iterations for seed mask
    ap.add_argument('--bbox-pad', type=int, default=6)  # Padding around bounding boxes
    ap.add_argument('--owlvit-debug', action='store_true')  # Enable debug visualization
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
    root = Path(args.root).expanduser().resolve()
    run_id = args.run_id or f"clip_{clip}_{ts}"  # Use provided run_id or generate with timestamp
    runs_root = root / "outputs"
    run_dir = runs_root / run_id
    ensure_dir(run_dir)
    run_folder = run_dir

    # Set up logging: store messages for later writing to file
    log_lines: List[str] = []
    def _log(msg: str) -> None:
        print(msg)  # Print to console
        log_lines.append(msg)  # Store for file output

    # Find video file
    clips_root = root / 'data' / 'clips'
    mp4 = clips_root / clip_filename
    if not mp4.exists() and not has_known_ext:
        alt = clips_root / clip_arg  # Try alternative path
        if alt.exists():
            mp4 = alt
    if not mp4.exists():
        raise SystemExit(f'Clip not found: {mp4}')

    # Set up directories for XMem Generic dataset format
    generic_base = root / 'work' / 'xmem_seq' / f'G_{clip}_full'
    frames_dir = generic_base / 'JPEGImages' / clip
    anns_dir = generic_base / 'Annotations' / clip
    ensure_dir(frames_dir)
    # Extract all frames from video
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
    # Each shot gets its own XMem evaluation with OWL-ViT seeding at shot start
    if args.shot_detect:
        try:
            shots = detect_shots(mp4, total_frames=total_frames, fps=fps, method=args.shot_method, min_shot_len_s=args.shot_min_seconds)
            shot_bounds: List[Tuple[int, int]] = [(int(s.start), int(s.end)) for s in shots]
        except Exception:
            shot_bounds = [(0, total_frames)]  # Fallback: treat entire video as single shot
    else:
        shot_bounds = [(0, total_frames)]  # Single shot mode: process entire video as one shot

    # Initialize OWL-ViT promptor for automatic object detection
    promptor: Optional[OwlVitBoxPromptor] = None
    promptor_error: Optional[str] = None
    if args.auto_prompt:
        prompts = _prompts_from_args(args)  # Get prompt terms from arguments
        # Pick device; if CUDA was requested but unavailable, fall back to CPU for OWL-ViT so we still get boxes.
        # OWL-ViT can run on CPU (slower but functional), unlike XMem which requires CUDA
        run_dev = args.owlvit_device or ('cuda' if args.device == 'cuda' else 'cpu')
        if run_dev == 'cuda':
            try:
                import torch  # type: ignore
                if not torch.cuda.is_available():
                    run_dev = 'cpu'
                    _log('OWL-ViT: requested cuda but torch.cuda.is_available() is False; falling back to CPU for prompts.')
            except Exception:
                run_dev = 'cpu'
                _log('OWL-ViT: CUDA probe failed; falling back to CPU for prompts.')
        try:
            # Initialize OWL-ViT model for object detection
            promptor = OwlVitBoxPromptor(
                model_id=args.owlvit_model,
                device=run_dev,
                prompts=prompts,
                score_thr=float(args.owlvit_score_thr),  # Minimum confidence threshold
                nms_iou=0.5,  # Non-maximum suppression IoU threshold
            )
        except Exception as e:
            promptor_error = str(e)
            promptor = None  # Continue without OWL-ViT if initialization fails

    # Log processing information
    _log(f'Processing {clip}')
    _log(f'  Device: {args.device}')
    _log(f'  Frames decoded: {len(frame_paths)} @ {fps:.2f} fps -> {W}x{H}')

    # Set up CSV file to track re-prompt events (shot information and detected boxes)
    rp_path = run_dir / f"re_prompts_clip_{clip}_{ts}.csv"
    rp_fields = ['shot_idx', 'start', 'end', 'obj_id', 'x0', 'y0', 'x1', 'y1', 'score', 'label', 'seed_type']
    with rp_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rp_fields)
        w.writeheader()

    # Set up output directory for masks
    masks_dst = run_dir / 'masks'
    ensure_dir(masks_dst)

    # Warn if OWL-ViT initialization failed
    if args.auto_prompt and promptor is None:
        msg = f'OWL-ViT promptor failed to initialize; no boxes will be produced. Error: {promptor_error or "unknown"}'
        _log(msg)

    # Per-shot processing loop: process each shot independently
    last_eval_error: Optional[str] = None
    for i, (s, e) in enumerate(shot_bounds, start=1):
        # Clamp shot boundaries to valid frame range
        s = max(0, int(s)); e = min(total_frames, int(e))
        if s >= e:
            continue  # Skip empty shots
        
        # Load first frame of shot for OWL-ViT detection
        frm_path = frames_dir / f'{s:05d}.jpg'
        bgr0 = cv2.imread(str(frm_path), cv2.IMREAD_COLOR)
        if bgr0 is None:
            continue  # Skip if frame can't be loaded

        # Initialize variables for detected boxes
        boxes_xyxy: List[Tuple[int,int,int,int]] = []
        labels: List[str] = []
        scores: List[float] = []
        seed_type = 'none'
        
        # Run OWL-ViT detection on shot start frame
        if promptor is not None:
            try:
                preds = promptor.predict_topk(bgr0, k=3) or []  # Get top 3 detections
            except Exception:
                preds = []
            # Filter predictions by confidence threshold
            kept = []
            for p in preds:
                try:
                    sc = float(getattr(p, 'score', 0.0))
                except Exception:
                    sc = 0.0
                if sc >= float(args.owlvit_score_thr):
                    kept.append(p)
            preds = kept
            # Extract bounding boxes, labels, and scores
            if preds:
                for p in preds[:3]:  # Take top 3 detections
                    try:
                        x0, y0, x1, y1 = p.as_int_tuple(width=W, height=H)
                    except Exception:
                        continue
                    boxes_xyxy.append((x0, y0, x1, y1))
                    labels.append(str(getattr(p, 'label', '')))
                    try:
                        scores.append(float(getattr(p, 'score', 0.0)))
                    except Exception:
                        scores.append(0.0)
                seed_type = 'owlvit'  # Mark seed type as OWL-ViT

        # Fallback to ground truth if OWL-ViT found no boxes and fallback is enabled
        if not boxes_xyxy and args.autoprompt_fallback == 'gt':
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

        # Save annotated seed image for visualization/debugging
        try:
            if boxes_xyxy:
                anno_src = bgr0 if bgr0 is not None else cv2.imread(str(frm_path), cv2.IMREAD_COLOR)
            
                if anno_src is not None:
                    anno = draw_debug_boxes(anno_src, boxes_xyxy, labels, scores)
                    cv2.imwrite(str(run_dir / f'shot_{i:03d}_seed.jpg'), anno)
        except Exception:
            pass
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
            _log('XMem eval failed for shot ' + str(i) + ' -> ' + str(e))
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
    write_overlay_video(frames_dir, masks_dst, run_dir / f"overlay_clip_{clip}_{ts}.mp4", fps=fps if fps > 0 else 10.0, alpha=0.35)

    # Compute metrics in SAM-2 format (if not skipped)
    summaries: List[Dict[str, object]] = []
    if not args.run_only:
        per_frame_path, summary = compute_metrics_sam2_like(root, clip, frames_dir, masks_dst, stride=int(args.stride), fps_video=fps, run_dir=run_dir, run_id=run_id)
        _log(f'Per-frame saved to: {per_frame_path}')
        summaries.append(summary)
    
    # Write summary CSV with aggregated metrics
    if summaries:
        from main_gt import write_summary
        summary_path = write_summary(run_dir, summaries, run_id=run_id)
        _log(f'Summary saved to {summary_path}')
        # Also write compatibility summary with specific field order
        try:
            compat_fields = ['clip_id','iou_median','iou_p25','iou_p75','biou_median','biou_p25','biou_p75','jitter_norm_median_pct','jitter_norm_p95_pct','area_cv_pct','empty_frames_pct','proc_fps_measured','proc_fps_theoretical','target_W','target_H','stride']
            compat_path = run_dir / 'summary_xmem.csv'
            with compat_path.open('w', newline='') as cf:
                cw = csv.DictWriter(cf, fieldnames=compat_fields)
                cw.writeheader()
                for row in summaries:
                    cw.writerow({k: row.get(k, '') for k in compat_fields})
            _log(f'Compatibility summary saved to {compat_path}')
        except Exception:
            pass

    # Write log file with all processing messages
    try:
        log_path = run_dir / f"pilot_clip_{clip}_{ts}.log"
        with log_path.open('w', encoding='utf-8') as f:
            for ln in log_lines:
                f.write(ln + '\n')
        _log(f'Log saved to {log_path}')
    except Exception:
        pass


if __name__ == '__main__':
    # Script entry point: run main function when executed directly
    main()
