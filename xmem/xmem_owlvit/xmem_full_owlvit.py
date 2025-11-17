import argparse
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
from PIL import Image
import time

# Make sibling sam2 repo importable for OWL-ViT + shot detection
THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
SAM2_ROOT = ROOT / 'sam2'
if str(SAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM2_ROOT))
# Add current directory for local imports
if str(THIS.parent) not in sys.path:
    sys.path.insert(0, str(THIS.parent))

from autoprompt_owlvit import OwlVitBoxPromptor, PROMPT_TERMS  # type: ignore
from shot_detection import detect_shots  # type: ignore

# Reuse local XMem helpers
from xmem_runs_local import (
    extract_first_frames,
    run_xmem_eval,
    prepare_run_dir,
    write_overlay_video,
    compute_metrics_sam2_like,
)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_palettized_mask(mask_bool: np.ndarray, out_png: Path) -> None:
    ensure_dir(out_png.parent)
    binm = (mask_bool.astype(np.uint8) > 0).astype(np.uint8)
    if int(binm.sum()) == 0:
        raise SystemExit(f'Seed is empty for {out_png}')
    pal = [0, 0, 0, 255, 255, 255] + [0, 0, 0] * 254
    pm = Image.fromarray(binm, mode='P')
    pm.putpalette(pal)
    pm.save(out_png, optimize=False)


def draw_debug_boxes(img_bgr: np.ndarray, boxes: List[Tuple[int,int,int,int]], labels: List[str], scores: List[float]) -> np.ndarray:
    out = img_bgr.copy()
    for k, (x0, y0, x1, y1) in enumerate(boxes):
        color = (0, 200, 0) if k == 0 else (0, 0, 255)
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), color, 2)
        lab = labels[k] if k < len(labels) else ''
        sc = scores[k] if k < len(scores) else None
        txt = f'{lab} {float(sc):.2f}' if isinstance(sc, (int, float, np.floating)) else str(lab)
        if txt:
            cv2.putText(out, txt, (int(x0), max(0, int(y0) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out


def boxes_union_mask(H: int, W: int, boxes: List[Tuple[int,int,int,int]], erode_iter: int = 1) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    for (x0, y0, x1, y1) in boxes:
        x0 = max(0, min(W - 1, int(x0)))
        y0 = max(0, min(H - 1, int(y0)))
        x1 = max(0, min(W - 1, int(x1)))
        y1 = max(0, min(H - 1, int(y1)))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 1
    if erode_iter and erode_iter > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.erode(mask, k, iterations=int(erode_iter))
    return mask


def load_labelme_billboard_mask(json_path: Path, W: int, H: int) -> Optional[np.ndarray]:
    try:
        data = json.loads(json_path.read_text())
    except Exception:
        return None
    mask = np.zeros((H, W), np.uint8)
    picked = 0
    for s in data.get('shapes', []):
        label = (s.get('label') or '').lower()
        if 'billboard' not in label:
            continue
        pts = s.get('points', [])
        if len(pts) >= 3:
            pts = np.array(pts, dtype=np.float32)
            pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
            cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
            picked += 1
    return mask if picked > 0 else None


def main():
    ap = argparse.ArgumentParser(description='XMem wrapper with OWL-ViT shot-start auto-prompting (union, single-ID)')
    # Inherit core XMem args
    ap.add_argument('--root', default=r'D:\\Billboard_Project\\xmem', help='XMem project root containing data/, model/, work/ etc')
    ap.add_argument('--clip', default='clip_fast', help='Clip ID, e.g. clip_fast')
    ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    ap.add_argument('--width', type=int, default=None, help='Resize width for extracted frames (keeps aspect)')
    ap.add_argument('--stride', type=int, default=1, help='Stride to report in summary (>=1)')
    ap.add_argument('--run-id', default='xmem_owlvit', help='Run ID used for output folder and filenames')
    ap.add_argument('--run-notes', action='store_true', help='Write simple RUN_NOTES.md')
    ap.add_argument('--run-only', action='store_true', help='Skip metrics')

    # Shot detection
    ap.add_argument('--shot-detect', action='store_true', help='Enable shot detection and per-shot seeding')
    ap.add_argument('--shot-method', choices=['adaptive', 'content'], default='adaptive')
    ap.add_argument('--shot-min-seconds', type=float, default=1.0)

    # OWL-ViT auto-prompt
    ap.add_argument('--auto-prompt', action='store_true', help='Enable OWL-ViT shot-start seeding')
    ap.add_argument('--owlvit-model', default='google/owlv2-base-patch16-ensemble')
    ap.add_argument('--owlvit-score', type=float, default=0.12)
    ap.add_argument('--owlvit-topk', type=int, default=2)
    ap.add_argument('--owlvit-nms', type=float, default=0.5)
    ap.add_argument('--owlvit-min-aspect', type=float, default=3.0)
    ap.add_argument('--owlvit-min-width-frac', type=float, default=0.12)
    ap.add_argument('--prompts-file', default=None)
    ap.add_argument('--prompts', action='append', default=None, help='Append a text prompt (repeat flag for multiple labels)')
    ap.add_argument('--autoprompt-fallback', choices=['gt', 'skip'], default='skip')
    ap.add_argument('--seed-erosion', type=int, default=1)
    ap.add_argument('--bbox-pad', type=int, default=6)
    ap.add_argument('--owlvit-debug', action='store_true')

    args = ap.parse_args()
    ts = time.strftime('%Y%m%d_%H%M%S')
    run_id = args.run_id + '_' + ts

    # Simple run logger to mirror Fix-2 style logs
    log_lines: List[str] = []
    def _log(msg: str) -> None:
        print(msg)
        log_lines.append(msg)

    root = Path(args.root)
    clip = args.clip

    mp4 = root / 'data' / 'clips' / f'{clip}.mp4'
    if not mp4.exists():
        raise SystemExit(f'Clip not found: {mp4}')

    # Prepare a full-clip frames dir once
    generic_base = root / 'work' / 'xmem_seq' / f'G_{clip}_full'
    frames_dir = generic_base / 'JPEGImages' / clip
    anns_dir = generic_base / 'Annotations' / clip  # not used for full clip
    ensure_dir(frames_dir)
    # Extract all frames (or resized to width)
    extract_first_frames(mp4, frames_dir, count=-1, width=args.width)

    # FPS for overlay/summary
    cap = cv2.VideoCapture(str(mp4))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 0.0
    cap.release()

    # Count frames
    frame_paths = sorted(frames_dir.glob('*.jpg'))
    if not frame_paths:
        raise SystemExit(f'No frames extracted at {frames_dir}')
    total_frames = len(frame_paths)
    H, W = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR).shape[:2]
    _log(f'Processing {clip}')
    _log(f'  Device: {args.device}')
    _log(f'  Frames decoded: {len(frame_paths)} @ {fps:.2f} fps -> {W}x{H}')

    # Determine shots
    if args.shot_detect:
        try:
            shots = detect_shots(mp4, total_frames=total_frames, fps=fps, method=args.shot_method, min_shot_len_s=args.shot_min_seconds)
            shot_bounds: List[Tuple[int, int]] = [(int(s.start), int(s.end)) for s in shots]
        except Exception:
            shot_bounds = [(0, total_frames)]
    else:
        shot_bounds = [(0, total_frames)]

    # Build OWL-ViT promptor
    promptor: Optional[OwlVitBoxPromptor] = None
    if args.auto_prompt:
        # Resolve prompts priority: explicit --prompts > --prompts-file > defaults
        prompts: List[str] = []
        if args.prompts:
            prompts = [p for p in args.prompts if str(p).strip()]
        elif args.prompts_file:
            try:
                prompts = [ln.strip() for ln in Path(args.prompts_file).read_text(encoding='utf-8').splitlines() if ln.strip()]
            except Exception:
                prompts = []
        if not prompts:
            prompts = list(PROMPT_TERMS)
        promptor = OwlVitBoxPromptor(
            model_id=args.owlvit_model,
            device=('cuda' if args.device == 'cuda' else 'cpu'),
            prompts=prompts,
            score_thr=float(args.owlvit_score),
            nms_iou=float(args.owlvit_nms),
        )

    # Prepare run-specific output folder
    runs_root = root / 'outputs' / f'G_{clip}' / 'runs'
    run_dir, run_folder = prepare_run_dir(runs_root, run_id)

    # Prepare per-clip log header
    header_lines = [
        f'OWLVIT device: {args.device}; model={getattr(args, "owlvit_model", "N/A")}; score_thr={getattr(args, "owlvit_score", "N/A")}; topk={getattr(args, "owlvit_topk", "N/A")}',
        f'Shots detected: {len(shot_bounds)}'
    ]

    # Re-prompt CSV path
    rp_path = run_dir / f're_prompts_{clip}_{run_id}.csv'
    rp_fields = ['shot_idx', 'start', 'end', 'obj_id', 'x0', 'y0', 'x1', 'y1', 'score', 'label', 'seed_type']
    with rp_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rp_fields)
        w.writeheader()

    # Destination masks folder
    masks_dst = run_dir / 'masks'
    ensure_dir(masks_dst)

    # Process each shot
    for i, (s, e) in enumerate(shot_bounds, start=1):
        s = max(0, int(s)); e = min(total_frames, int(e))
        if s >= e:
            continue
        # Read shot-start frame
        frm_path = frames_dir / f'{s:05d}.jpg'
        bgr0 = cv2.imread(str(frm_path), cv2.IMREAD_COLOR)
        if bgr0 is None:
            continue

        # Predict up to K boxes
        boxes_xyxy: List[Tuple[int,int,int,int]] = []
        labels: List[str] = []
        scores: List[float] = []
        seed_type = 'none'
        if promptor is not None:
            try:
                preds = promptor.predict_topk(bgr0, k=max(1, int(args.owlvit_topk)))
            except Exception:
                preds = []
            # Filter by min-aspect and min-width-frac
            for p in preds:
                x0, y0, x1, y1 = p.as_int_tuple(width=W, height=H)
                w = max(1, int(x1) - int(x0)); h = max(1, int(y1) - int(y0))
                ar = w / float(h)
                if ar < float(args.owlvit_min_aspect):
                    continue
                if (w / float(W)) < float(args.owlvit_min_width_frac):
                    continue
                boxes_xyxy.append((x0, y0, x1, y1))
                labels.append(str(getattr(p, 'label', '')))
                scores.append(float(getattr(p, 'score', 0.0)))
            if boxes_xyxy:
                # Union up to 2
                boxes_xyxy = boxes_xyxy[:2]
                labels = labels[:2]
                scores = scores[:2]
                seed_type = 'owlvit'

        # Fallback if no boxes
        if not boxes_xyxy:
            if args.autoprompt_fallback == 'gt':
                # Find first GT json in [s, e)
                gt_root = root / 'data' / 'gt_frames' / clip
                fb_box = None
                for fr in range(int(s), int(e)):
                    j6 = gt_root / f'frame_{fr:06d}.json'
                    j5 = gt_root / f'frame_{fr:05d}.json'
                    jp = j6 if j6.is_file() else (j5 if j5.is_file() else None)
                    if not jp:
                        continue
                    gt_mask = load_labelme_billboard_mask(jp, W, H)
                    if gt_mask is None or int(gt_mask.sum()) == 0:
                        continue
                    ys, xs = np.where(gt_mask > 0)
                    if xs.size and ys.size:
                        x0, x1 = int(xs.min()), int(xs.max())
                        y0, y1 = int(ys.min()), int(ys.max())
                        boxes_xyxy = [(x0, y0, x1, y1)]
                        labels = ['gt']
                        scores = [1.0]
                        seed_type = 'gt'
                        break
            # else skip

        # Log CSV row (one per shot) with union box if present
        if boxes_xyxy:
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

        # Save annotated seed image (always when boxes exist)
        try:
            # Prefer in-memory frame; fall back to reading from disk
            frame_for_annot = bgr0 if bgr0 is not None else cv2.imread(str(frm_path), cv2.IMREAD_COLOR)
            if frame_for_annot is not None and boxes_xyxy:
                anno = draw_debug_boxes(frame_for_annot, boxes_xyxy, labels, scores)
                cv2.imwrite(str(run_dir / f"shot_{i:03d}_seed.jpg"), anno)
        except Exception:
            pass
        # Optional debug image copy
        if args.owlvit_debug:
            try:
                dbg_src = bgr0 if bgr0 is not None else frame_for_annot
                if dbg_src is not None:
                    dbg = draw_debug_boxes(dbg_src, boxes_xyxy, labels, scores)
                    out_dbg = run_dir / f"debug_shot_{i:03d}.jpg"
                    cv2.imwrite(str(out_dbg), dbg)
            except Exception:
                pass

        # Skip vendor run if no seed
        if not boxes_xyxy:
            continue
        # Build shot-specific Generic dataset at a temp path
        shot_base = root / 'work' / 'xmem_seq' / f'G_{clip}_shot_{i:03d}'
        shot_frames = shot_base / 'JPEGImages' / clip
        shot_anns = shot_base / 'Annotations' / clip
        if shot_base.exists():
            try:
                shutil.rmtree(shot_base)
            except Exception:
                pass
        ensure_dir(shot_frames); ensure_dir(shot_anns)
        # Copy frames [s, e) into shot_frames as 00000.jpg ..
        for idx, fr in enumerate(range(int(s), int(e))):
            src = frames_dir / f'{fr:05d}.jpg'
            dst = shot_frames / f'{idx:05d}.jpg'
            try:
                shutil.copy2(src, dst)
            except Exception:
                break

        # Create union seed at 00000.png
        seed_mask = boxes_union_mask(H, W, boxes_xyxy, erode_iter=int(args.seed_erosion))
        try:
            save_palettized_mask(seed_mask, shot_anns / '00000.png')
        except Exception as e:
            print('Failed to save seed:', e)
            continue

        # Run vendor eval for this shot and stitch masks back into run_dir/masks
        try:
            _, vendor_clip_dir = run_xmem_eval(root, clip, device=args.device, generic_path=shot_base)
            # Copy vendor masks with global frame indices
            for p in sorted(vendor_clip_dir.glob('*.png')):
                try:
                    local_idx = int(p.stem)
                except Exception:
                    continue
                global_idx = s + local_idx
                out = masks_dst / f'{global_idx:05d}.png'
                shutil.copy2(p, out)
        except SystemExit as e:
            _log('XMem eval failed for shot ' + str(i) + ' -> ' + str(e))
            continue

    # Write overlay video
    write_overlay_video(frames_dir, masks_dst, run_dir / f'overlay_{clip}_{run_id}.mp4', fps=fps if fps > 0 else 10.0, alpha=0.35)

    # Metrics and summary
    summaries: List[Dict[str, object]] = []
    if not args.run_only:
        per_frame_path, summary = compute_metrics_sam2_like(root, clip, frames_dir, masks_dst, stride=int(args.stride), fps_video=fps, run_dir=run_dir, run_id=run_id)
        _log(f'Per-frame saved to: {per_frame_path}')
        summaries.append(summary)
    if summaries:
        # Write summary header and file
        for ln in header_lines:
            _log(ln)
        # Write summary CSV
        from xmem_runs_local import write_summary  # local import to avoid circular
        summary_path = write_summary(run_dir, summaries, run_id=run_id)
        _log(f'Summary saved to {summary_path}')
        # Also write a compatibility summary file name
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

    # Write run log similar to Fix-2
    try:
        log_path = run_dir / f'pilot_{clip}_{run_id}.log'
        with log_path.open('w', encoding='utf-8') as f:
            for ln in log_lines:
                f.write(ln + '\n')
        _log(f'Log saved to {log_path}')
    except Exception:
        pass

    if args.run_notes:
        out_notes = run_dir / 'RUN_NOTES.md'
        with out_notes.open('w', encoding='utf-8') as f:
            f.write('# XMem OWL-ViT Auto-Prompt Run\n')
            f.write(f'Clip: {clip}\nRun folder: {run_dir}\n')
            for ln in header_lines:
                f.write(ln + '\n')
            if summaries:
                s = summaries[0]
                f.write(f"IoU_med={s['iou_median']:.3f}, BIoU_med={s['biou_median']:.3f}, jitter_med={s['jitter_norm_median_pct']:.3f}%/frame\n")


if __name__ == '__main__':
    main()









