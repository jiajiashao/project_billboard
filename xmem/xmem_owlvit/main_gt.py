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

import cv2
import numpy as np
from PIL import Image

# XMem local runner aligned to SAM-2 output metrics/structure
# - Writes per-frame CSV with SAM-2 fields
# - Writes summary CSV with SAM-2 fields
# - Creates a separate output folder per run for each clip


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def extract_first_frames(mp4: Path, out_dir: Path, count: Optional[int] = None, width: Optional[int] = None) -> List[Path]:
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {mp4}")
    frames: List[Path] = []
    idx = 0
    while True:
        if (count is not None) and (int(count) >= 0) and (idx >= int(count)):
            break
        ok, frame = cap.read()
        if not ok:
            break
        if width is not None and width > 0:
            h, w = frame.shape[:2]
            s = width / max(1, w)
            frame = cv2.resize(frame, (int(round(w*s)), int(round(h*s))), interpolation=cv2.INTER_AREA)
        fn = out_dir / f"{idx:05d}.jpg"
        cv2.imwrite(str(fn), frame)
        frames.append(fn)
        idx += 1
    cap.release()
    if (count is not None) and (int(count) >= 0) and (len(frames) < int(count)):
        print(f"Warning: only extracted {len(frames)} frames from {mp4}")
    return frames


def write_palettized_seed(src_png: Path, out_png: Path) -> None:
    ensure_dir(out_png.parent)
    m = np.array(Image.open(src_png).convert("L"))
    binm = (m > 127).astype(np.uint8)  # strict 0/1
    if binm.sum() == 0:
        raise SystemExit(f"Seed is empty: {src_png}")
    pal = [0,0,0, 255,255,255] + [0,0,0]*254
    pm = Image.fromarray(binm, mode="P")
    pm.putpalette(pal)
    pm.save(out_png, optimize=False)


def run_xmem_eval(
    root: Path,
    clip: str,
    device: str = "cuda",
    size: int = 480,
    mem_every: int = 5,
    generic_path: Optional[Path] = None,
    xmem_root: Optional[Path] = None,
) -> Tuple[Path, Path]:
    # XMem only supports CUDA; bail early with a clear message if CUDA is not available
    if str(device).lower() != "cuda":
        raise SystemExit("XMem eval currently supports only CUDA devices; run on a CUDA GPU box")
    try:
        import torch  # type: ignore
        has_cuda = bool(torch.cuda.is_available())
    except Exception:
        has_cuda = False
    if not has_cuda:
        raise SystemExit("XMem eval requires a CUDA-capable GPU (torch.cuda.is_available() is False)")

    resolved_xmem_root = (Path(xmem_root).expanduser() if xmem_root else (root / "model" / "xmem")).resolve()
    model_path = resolved_xmem_root / "saves" / "XMem-s012.pth"
    if not model_path.exists():
        raise SystemExit(f"Model checkpoint not found: {model_path}")

    generic_path = (root / "work" / "xmem_seq" / f"G_{clip}") if generic_path is None else generic_path
    out_vendor = root / "outputs" / f"G_{clip}"
    ensure_dir(out_vendor)

    cmd = [
        os.fspath(Path(os.sys.executable)),
        os.fspath((resolved_xmem_root / "eval.py").resolve()),
        "--model", os.fspath(model_path),
        "--dataset", "G",
        "--generic_path", os.fspath(generic_path),
        "--size", str(size),
        "--mem_every", str(mem_every),
        "--output", os.fspath(out_vendor),
    ]
    print("CMD:", " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=os.fspath(resolved_xmem_root), check=True)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"XMem eval failed with exit code {e.returncode}")

    return out_vendor, out_vendor / clip


def load_labelme_billboard_mask(json_path: Path, W: int, H: int) -> Optional[np.ndarray]:
    try:
        data = json.loads(json_path.read_text())
    except Exception:
        return None
    mask = np.zeros((H, W), np.uint8)
    picked = 0
    for s in data.get("shapes", []):
        label = (s.get("label") or "").lower()
        if "billboard" not in label:
            continue
        pts = s.get("points", [])
        if len(pts) >= 3:
            pts = np.array(pts, dtype=np.float32)
            pts[:,0] = np.clip(pts[:,0], 0, W-1)
            pts[:,1] = np.clip(pts[:,1], 0, H-1)
            cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
            picked += 1
    return mask if picked > 0 else None


def boundary(mask: np.ndarray, r: int = 3) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
    dil = cv2.dilate(mask, k)
    ero = cv2.erode(mask, k)
    return (dil ^ ero)


def centroid_from_mask(mask_bool: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.nonzero(mask_bool)
    if len(xs) == 0:
        return (math.nan, math.nan)
    cx = float(xs.mean())
    cy = float(ys.mean())
    return (cx, cy)


def nan_percentiles(values: List[float], percents: List[int]) -> Tuple[float, ...]:
    arr = np.array(values, dtype=np.float32)
    if arr.size == 0:
        return tuple(float("nan") for _ in percents)
    return tuple(float(np.nanpercentile(arr, p)) for p in percents)


def nan_mean(values: List[float]) -> float:
    arr = np.array(values, dtype=np.float32)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def prepare_run_dir(root: Path, run_id: str) -> Tuple[Path, str]:
    ensure_dir(root)
    candidate = root / run_id
    if not candidate.exists():
        candidate.mkdir()
        return candidate, candidate.name
    ts = time.strftime('%Y%m%d_%H%M%S')
    candidate = root / f"{run_id}_{ts}"
    candidate.mkdir()
    return candidate, candidate.name


def write_per_frame_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    ensure_dir(path.parent)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def compute_metrics_sam2_like(root: Path, clip: str, frames_dir: Path, mask_dir: Path, stride: int, fps_video: float, run_dir: Path, run_id: str) -> Tuple[Path, Dict[str, object]]:
    gt_dir = root / "data" / "gt_frames" / clip
    if not frames_dir.is_dir():
        raise SystemExit(f"Frames folder not found: {frames_dir}")
    if not mask_dir.is_dir():
        raise SystemExit(f"Mask folder not found: {mask_dir}")

    frames = sorted(frames_dir.glob("*.jpg"))
    masks = sorted(mask_dir.glob("*.png"))
    if not frames or not masks:
        raise SystemExit("Missing frames or masks for metrics.")

    def stem(p: Path) -> str: return p.stem
    frame_map = {stem(p): p for p in frames}
    mask_map  = {stem(p): p for p in masks}
    indices = sorted(set(frame_map).intersection(mask_map))
    if not indices:
        raise SystemExit("No matching frame/mask stems.")

    H, W = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR).shape[:2]

    frame_rows: List[Dict[str, object]] = []
    prev_centroid = (math.nan, math.nan)

    for idx in indices:
        j6 = gt_dir / f"frame_{idx.zfill(6)}.json"
        j5 = gt_dir / f"frame_{idx.zfill(5)}.json"
        gt_json = j6 if j6.is_file() else (j5 if j5.is_file() else None)
        if not gt_json:
            continue
        gt = load_labelme_billboard_mask(gt_json, W, H)
        if gt is None or int(gt.sum()) == 0:
            continue
        pred = cv2.imread(str(mask_map[idx]), cv2.IMREAD_UNCHANGED)
        if pred is None:
            continue
        if pred.ndim == 3:
            pred = pred[...,0]
        pred_bool = (pred > 0)
        gt_bool = (gt > 0)
        inter = int(np.logical_and(gt_bool, pred_bool).sum())
        union = int(np.logical_or (gt_bool, pred_bool).sum())
        iou = (inter/union) if union>0 else 0.0
        b_gt   = boundary(gt_bool.astype(np.uint8), 3)
        b_pred = boundary(pred_bool.astype(np.uint8), 3)
        b_inter = int(np.logical_and(b_gt>0, b_pred>0).sum())
        b_union = int(np.logical_or (b_gt>0, b_pred>0).sum())
        biou = (b_inter/b_union) if b_union>0 else 0.0

        area = float(pred_bool.sum())
        is_empty = 1 if area <= 1.0 else 0
        cx, cy = centroid_from_mask(pred_bool)

        shift_px = math.nan
        shift_norm = math.nan
        if not math.isnan(cx) and not math.isnan(prev_centroid[0]):
            dx = cx - prev_centroid[0]
            dy = cy - prev_centroid[1]
            shift_px = float(math.hypot(dx, dy))
            shift_norm = (shift_px / max(1.0, float(W))) * 100.0
        if not math.isnan(cx):
            prev_centroid = (cx, cy)
        else:
            prev_centroid = (math.nan, math.nan)

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
            'runtime_ms': math.nan,
            # ROI columns omitted to match SAM-2 uniform outputs
        }
        frame_rows.append(row)

    per_frame_path = run_dir / f"per_frame_{clip}_{run_id}.csv"
    per_frame_cols = [
        'clip_id','frame_no','iou','biou','area_px','centroid_x','centroid_y',
        'centroid_shift_px','centroid_shift_norm','is_empty','runtime_ms'
    ]
    write_per_frame_csv(per_frame_path, frame_rows, per_frame_cols)

    # Summary aggregation
    ious = [r['iou'] for r in frame_rows]
    bious = [r['biou'] for r in frame_rows]
    jitter_vals = [val for val in (r['centroid_shift_norm'] for r in frame_rows) if isinstance(val, float) and not math.isnan(val)]
    area_vals = [r['area_px'] for r in frame_rows]
    empty_vals = [r['is_empty'] for r in frame_rows]

    iou_p25, iou_med, iou_p75 = nan_percentiles(ious, [25, 50, 75])
    biou_p25, biou_med, biou_p75 = nan_percentiles(bious, [25, 50, 75])
    jitter_med, jitter_p95 = nan_percentiles(jitter_vals, [50, 95]) if jitter_vals else (float('nan'), float('nan'))

    area_mean = nan_mean(area_vals)
    area_std = float('nan')
    if not math.isnan(area_mean):
        arr = np.array(area_vals, dtype=np.float32)
        area_std = float(np.std(arr))
    area_cv = float('nan') if math.isnan(area_mean) or area_mean == 0 else (area_std / area_mean) * 100.0
    empty_pct = (sum(empty_vals) / len(empty_vals) * 100.0) if empty_vals else float('nan')

    # FPS
    stride = max(1, int(stride))
    fps_theoretical = (float(fps_video) / float(stride)) if fps_video and fps_video > 0 else 0.0
    fps_measured = 0.0  # XMem eval is external; per-frame runtime unavailable here

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
    try:
        frames = sorted(frames_dir.glob('*.jpg'))
        if not frames:
            print(f'No frames in {frames_dir} for overlay.')
            return None
        first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
        if first is None:
            print(f'Failed to read first frame for overlay: {frames[0]}')
            return None
        H, W = first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_mp4), fourcc, float(fps), (W, H))
        color = (170, 0, 255)  # BGR magenta
        for fp in frames:
            frm = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            if frm is None:
                continue
            mp = masks_dir / (fp.stem + '.png')
            m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
            if m is None:
                writer.write(frm)
                continue
            if m.ndim == 3:
                m = m[...,0]
            mask = (m > 0)
            overlay = frm.copy()
            overlay[mask] = color
            blended = cv2.addWeighted(overlay, alpha, frm, 1.0 - alpha, 0)
            writer.write(blended)
        writer.release()
        print(f'Overlay saved to: {out_mp4}')
        return out_mp4
    except Exception as e:
        print(f'Overlay generation failed: {e}')
        return None


def write_summary(run_dir: Path, summaries: List[Dict[str, object]], run_id: str) -> Path:
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
            w.writerow({k: row.get(k, '') for k in fields})
    return summary_path


def main():
    ap = argparse.ArgumentParser(description="Run XMem locally with SAM-2-like outputs")
    ap.add_argument('--root', default="./../", help='Project root containing data/, model/ etc')
    ap.add_argument('--clip', default='clip_fast', help='Clip ID, e.g. clip_fast')
    ap.add_argument('--device', default='cuda', choices=['cuda','cpu'])
    ap.add_argument('--frames', type=int, default=-1, help='How many initial frames to prepare (-1 = all)')
    ap.add_argument('--width', type=int, default=None, help='Resize width for extracted frames (keeps aspect)')
    ap.add_argument('--stride', type=int, default=1, help='Stride to report in summary (>=1)')
    ap.add_argument('--run-id', default='xmem', help='Run ID used for output folder and filenames')
    ap.add_argument('--run-notes', action='store_true', help='Write simple RUN_NOTES.md')
    ap.add_argument('--run-only', action='store_true', help='Skip metrics')
    ap.add_argument('--xmem-root', type=str, default=None, help='Override path to XMem repo (defaults to <root>/model/xmem)')
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    xmem_root_override = Path(args.xmem_root).expanduser().resolve() if args.xmem_root else None
    clip = args.clip

    mp4 = root / 'data' / 'clips' / f'{clip}.mp4'
    seed_src = root / 'data' / 'seeds' / clip / '00000.png'
    generic_base = root / 'work' / 'xmem_seq' / f'G_{clip}'
    frames_dir = generic_base / 'JPEGImages' / clip
    anns_dir   = generic_base / 'Annotations' / clip

    if not mp4.exists():
        raise SystemExit(f"Clip not found: {mp4}")
    if not seed_src.exists():
        raise SystemExit(f"Seed not found: {seed_src}")

    print('Preparing Generic dataset...')
    extract_first_frames(mp4, frames_dir, count=int(args.frames), width=args.width)
    write_palettized_seed(seed_src, anns_dir / '00000.png')

    print('Running XMem eval...')
    vendor_root, vendor_clip_dir = run_xmem_eval(
        root,
        clip,
        device=args.device,
        generic_path=generic_base,
        xmem_root=xmem_root_override,
    )
    print('Vendor out:', vendor_clip_dir)

    # Prepare run-specific output folder
    runs_root = root / 'outputs' / f'G_{clip}' / 'runs'
    run_dir, run_folder = prepare_run_dir(runs_root, args.run_id)

    # Copy masks into run dir to snapshot results
    masks_src = vendor_clip_dir
    masks_dst = run_dir / 'masks'
    ensure_dir(masks_dst)
    for p in sorted(masks_src.glob('*.png')):
        shutil.copy2(p, masks_dst / p.name)

    # Write overlay video inside run dir
    # Try to read FPS from source video
    cap = cv2.VideoCapture(str(mp4))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 0.0
    cap.release()
    write_overlay_video(frames_dir, masks_dst, run_dir / f'overlay_{clip}_{args.run_id}.mp4', fps=fps if fps>0 else 10.0, alpha=0.35)

    summaries: List[Dict[str, object]] = []
    if not args.run_only:
        per_frame_path, summary = compute_metrics_sam2_like(root, clip, frames_dir, masks_dst, stride=int(args.stride), fps_video=fps, run_dir=run_dir, run_id=args.run_id)
        print(f'Per-frame saved to: {per_frame_path}')
        summaries.append(summary)

    if summaries:
        summary_path = write_summary(run_dir, summaries, run_id=args.run_id)
        print(f'Summary saved to {summary_path}')

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
    main()
