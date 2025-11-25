import argparse
import csv
import json
import math
import datetime as dt
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
from torch import cuda

# Reuse SAM-2 utilities from the project
import sam2_base as base
from shot_detection import detect_shots


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Segment billboards: YOLO seeding + SAM-2 propagation (v4-like outputs)")
    # IO
    p.add_argument("--data", required=True, help="Video file or directory (recurses for common video types)")
    p.add_argument("--out-dir", default="runs/yolo_sam2", help="Output root; per-clip folder gets timestamp suffix")
    # YOLO
    p.add_argument("--yolo-model",default="./weights/best.pt", help="Path to YOLO model weights")
    p.add_argument("--yolo-conf", type=float, default=0.20, help="YOLO confidence threshold")
    p.add_argument("--yolo-max-objects", type=int, default=3, help="Max YOLO boxes per shot start")
    # SAM-2
    p.add_argument("--sam2-weights", default="facebook/sam2.1-hiera-tiny", help="HuggingFace weights for SAM-2 video")
    p.add_argument("--input-width", type=int, default=1280, help="Target decode width (frames resized keeping aspect)")
    # Device/Perf
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda", help="Run device for both models")
    p.add_argument("--stride", type=int, default=1, help="Metrics sampling stride (visual only)")
    # Shots
    p.add_argument("--shot-mode", choices=["auto", "single"], default="auto", help="Detect shots or treat whole video as one shot")
    # GT root
    p.add_argument("--gt-root", default="..\sam2\data\gt_frames")
    return p.parse_args()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_sources(src: Path) -> List[Path]:
    if src.is_file():
        return [src]
    video_patterns = ["*.mp4", "*.mkv", "*.mov", "*.avi", "*.m4v", "*.mpg", "*.mpeg", "*.wmv", "*.ts", "*.webm"]
    out: List[Path] = []
    for pat in video_patterns:
        out.extend(src.rglob(pat))
    return sorted(out)


def bgr_from_frame(frame) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[2] == 3:
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        try:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception:
            return arr
    raise ValueError("unexpected frame array shape")


def yolo_boxes(model, frame_bgr: np.ndarray, conf: float, imgsz: int, device: str, max_det: int) -> Tuple[List[List[int]], List[float]]:
    res = model.predict(source=frame_bgr, imgsz=imgsz, conf=conf, device=device, verbose=False, max_det=max_det)
    boxes: List[List[int]] = []
    scores: List[float] = []
    if not res:
        return boxes, scores
    r = res[0]
    battr = getattr(r, "boxes", None)
    if battr is None or getattr(battr, "xyxy", None) is None:
        return boxes, scores
    xyxy = battr.xyxy.detach().cpu().numpy()
    confs = battr.conf.detach().cpu().numpy() if getattr(battr, "conf", None) is not None else np.zeros((xyxy.shape[0],), dtype=np.float32)
    order = np.argsort(-confs)
    for i in order[: int(max(1, max_det))]:
        x0, y0, x1, y1 = [int(round(float(v))) for v in xyxy[i].tolist()]
        boxes.append([x0, y0, x1, y1])
        scores.append(float(confs[i]))
    return boxes, scores


def process_video(path: Path, args: argparse.Namespace, out_root: Path) -> Dict[str, object]:
    # Device
    dev = base.select_device(args.device)
    dtype = base.torch.float32

    # Load models
    from ultralytics import YOLO  # type: ignore
    yolo = YOLO(args.yolo_model)
    sam2_model = base.Sam2VideoModel.from_pretrained(args.sam2_weights).to(device=dev, dtype=dtype)
    sam2_model.eval()
    sam2_proc = base.Sam2VideoProcessor.from_pretrained(args.sam2_weights)

    # Decode frames
    frames, (tgt_w, tgt_h), fps = base.read_resize_frames(str(path), args.input_width)

    # Robust SAM-2 inference helper for multi-object outputs
    import time as _t
    def _infer_frame_multi(frame_idx: int):
        H, W = tgt_h, tgt_w
        start = _t.perf_counter()
        output = sam2_model(inference_session=session, frame_idx=frame_idx)
        post = sam2_proc.post_process_masks([output.pred_masks], original_sizes=[[H, W]], binarize=False)[0]
        if hasattr(post, "detach"):
            try:
                post = post.detach().cpu().numpy()
            except Exception:
                post = np.array(post)
        arr = post
        if arr is None:
            prob = np.zeros((H, W), dtype=np.float32)
        else:
            if arr.ndim == 2:
                prob = arr
            elif arr.ndim == 3:
                if arr.shape[0] != H and arr.shape[-2:] == (H, W):
                    prob = np.max(arr, axis=0)
                elif arr.shape[0:2] == (H, W):
                    prob = np.max(arr, axis=2)
                else:
                    while arr.ndim > 2:
                        arr = np.max(arr, axis=0)
                    prob = arr
            else:
                while arr.ndim > 2:
                    arr = np.max(arr, axis=0)
                prob = arr
        runtime_ms = (_t.perf_counter() - start) * 1000.0
        mask = base.prob_to_mask(prob, threshold=base.MASK_THRESHOLD, shape=(H, W))
        return mask, runtime_ms

    # Shots
    if args.shot_mode == "auto":
        try:
            shots = detect_shots(path, total_frames=len(frames), fps=fps, method="adaptive", min_shot_len_s=1.0, adaptive_sensitivity=3)
            shot_bounds = [(int(s.start), int(s.end)) for s in shots]
        except Exception:
            shot_bounds = [(0, int(len(frames)))]
    else:
        shot_bounds = [(0, int(len(frames)))]

    # Output (timestamped per clip)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    clip_dir = out_root / f"{path.stem}_{ts}"
    ensure_dir(clip_dir)
    overlay_path = clip_dir / f"overlay_{path.stem}_sam2.mp4"
    writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (tgt_w, tgt_h))

    # Allocate masks, metrics, and logging
    masks_full: List[np.ndarray] = [np.zeros((tgt_h, tgt_w), dtype=np.uint8) for _ in range(len(frames))]
    frame_runtimes: Dict[int, float] = {}
    stride = max(1, int(args.stride))
    log_lines: List[str] = []
    def _log(m: str) -> None:
        print(m)
        log_lines.append(m)
    re_prompt_rows: List[Dict[str, object]] = []

    _log(f"Processing {path}")
    _log(f"  Device: {dev}")
    _log(f"  Frames decoded: {len(frames)} @ {fps:.2f} fps -> {tgt_w}x{tgt_h}")
    _log(f"Autoprompt[YOLO] model={args.yolo_model} conf={args.yolo_conf}")
    _log(f"Shots detected: {len(shot_bounds)}")

    with base.torch.inference_mode():
        for i, (s, e) in enumerate(shot_bounds, start=1):
            s = int(max(0, min(s, len(frames) - 1)))
            e = int(max(s + 1, min(e, len(frames))))
            if e - s <= 0:
                continue
            # YOLO on shot start
            try:
                bgr0 = bgr_from_frame(frames[s])
            except Exception:
                bgr0 = None
            boxes, scores = ([], [])
            if bgr0 is not None:
                boxes, scores = yolo_boxes(yolo, bgr0, conf=args.yolo_conf, imgsz=args.input_width, device=args.device or dev, max_det=args.yolo_max_objects)
            if not boxes:
                _log(f"Shot {i}/{len(shot_bounds)}: YOLO NONE - skip frames {s}-{e-1}")
                re_prompt_rows.append({"shot_idx": i, "start": int(s), "end": int(e), "mode": "yolo-none", "score": "", "box_json": "[]"})
                continue
            re_prompt_rows.append({"shot_idx": i, "start": int(s), "end": int(e), "mode": "yolo", "score": (f"{scores[0]:.3f}" if scores else ""), "box_json": json.dumps(boxes)})
            # SAM-2 session for this shot
            session = sam2_proc.init_video_session(video=frames[s:e], inference_device=dev, processing_device=dev, video_storage_device=dev, dtype=dtype)
            seed_boxes = [boxes]
            obj_ids = list(range(1, len(boxes) + 1))
            sam2_proc.add_inputs_to_inference_session(inference_session=session, frame_idx=0, obj_ids=obj_ids, input_boxes=seed_boxes, input_points=None, input_labels=None)
            # Save YOLO seed annotated JPG in clip folder
            try:
                img_seed = bgr0.copy() if bgr0 is not None else None
                if img_seed is not None:
                    for (x0,y0,x1,y1), sc in zip(boxes, scores):
                        cv2.rectangle(img_seed, (int(x0),int(y0)), (int(x1),int(y1)), (0,200,0), 2)
                        tag = f"billboard {sc:.2f}"
                        cv2.putText(img_seed, tag, (int(x0), max(0, int(y0)-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                        cv2.putText(img_seed, tag, (int(x0), max(0, int(y0)-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
                    cv2.imwrite(str(clip_dir / f"shot_{i:03d}_seed.jpg"), img_seed)
            except Exception:
                pass
            _log("Seeded {} boxes at shot {} with scores=".format(len(boxes), s) + ",".join([f"{x:.2f}" for x in scores]))
            # Per-frame segmentation
            for off in range(0, e - s):
                mask, runtime_ms = _infer_frame_multi(off)
                masks_full[s + off] = mask
                if ((s + off) - s) % stride == 0:
                    frame_runtimes[s + off] = frame_runtimes.get(s + off, 0.0) + runtime_ms

    # Write overlay + annotated frames and compute metrics if GT available
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gt_dir = Path(args.gt_root) / path.stem if args.gt_root else None

    frame_rows: List[Dict[str, object]] = []
    prev_metric_centroid = (math.nan, math.nan)
    for idx, (frame, mask) in enumerate(zip(frames, masks_full)):
        rgb = np.array(frame)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        bgr = base.overlay_mask(rgb, cleaned)
        writer.write(bgr)
        if gt_dir is not None:
            gt_json = gt_dir / f"frame_{idx:06d}.json"
            if gt_json.exists():
                try:
                    src_w, src_h, polys = base.load_labelme_polys(gt_json)
                except (KeyError, ValueError, Exception):
                    continue
                if polys:
                    gt_mask = base.polys_to_mask(polys, src_w, src_h, tgt_w, tgt_h)
                    pred_mask = cleaned > 0
                    gt_mask_bool = gt_mask > 0
                    iou = base.compute_iou(pred_mask, gt_mask_bool)
                    biou = base.compute_iou(base.band_mask(pred_mask), base.band_mask(gt_mask_bool))
                    area = float(pred_mask.sum())
                    cx, cy = base.centroid_from_mask(pred_mask)
                    shift_px = math.nan
                    shift_norm = math.nan
                    if idx in frame_runtimes and not math.isnan(cx) and not math.isnan(prev_metric_centroid[0]):
                        dx = cx - prev_metric_centroid[0]
                        dy = cy - prev_metric_centroid[1]
                        shift_px = (dx * dx + dy * dy) ** 0.5
                        shift_norm = (shift_px / max(1.0, tgt_w)) * 100.0
                    if idx in frame_runtimes:
                        prev_metric_centroid = (cx, cy) if not math.isnan(cx) else (math.nan, math.nan)
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
    writer.release()

    # Per-frame metrics CSV
    per_frame_path = clip_dir / f"per_frame_{path.stem}_sam2.csv"
    if frame_rows:
        fields = ["clip_id","frame_no","iou","biou","area_px","centroid_x","centroid_y","centroid_shift_px","centroid_shift_norm","is_empty","runtime_ms","roi_w","roi_h"]
        with per_frame_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in frame_rows:
                w.writerow(r)

    # Summary metrics
    def _nan_percentiles(arr: List[float], qs: List[int]):
        xs = [x for x in arr if isinstance(x, float) and not math.isnan(x)]
        if not xs:
            return tuple([math.nan for _ in qs])
        xs = sorted(xs)
        out = []
        for q in qs:
            k = int(round((q / 100.0) * (len(xs) - 1)))
            out.append(float(xs[k]))
        return tuple(out)

    ious = [r["iou"] for r in frame_rows] if frame_rows else []
    biou_vals = [r["biou"] for r in frame_rows] if frame_rows else []
    iou_p25, iou_med, iou_p75 = _nan_percentiles(ious, [25, 50, 75]) if frame_rows else (math.nan, math.nan, math.nan)
    biou_p25, biou_med, biou_p75 = _nan_percentiles(biou_vals, [25, 50, 75]) if frame_rows else (math.nan, math.nan, math.nan)

    # Run log and params
    log_path = clip_dir / f"pilot_{path.stem}_sam2.log"
    with log_path.open("w", encoding="utf-8") as f:
        for line in log_lines:
            f.write(line + "\n")

    params_path = clip_dir / f"params_{path.stem}_sam2.json"
    params_payload = {
        "clip_id": path.stem,
        "clip_path": str(path),
        "input_width": args.input_width,
        "device": base.select_device(args.device),
        "yolo": {"model": args.yolo_model, "conf": args.yolo_conf, "max_objects": args.yolo_max_objects},
        "sam2": {"weights": args.sam2_weights},
        "shots": shot_bounds,
    }
    with params_path.open("w") as f:
        json.dump(params_payload, f, indent=2)

    # Re-prompt events CSV
    re_csv = clip_dir / f"re_prompts_{path.stem}_sam2.csv"
    if re_prompt_rows:
        with re_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["shot_idx","start","end","mode","score","box_json"])
            w.writeheader()
            for r in re_prompt_rows:
                w.writerow(r)

        # Write summary CSV (fix2-like format)
    summary_path = clip_dir / "summary_fix2.csv"
    # Jitter percentiles
    jitter_list = [r["centroid_shift_norm"] for r in frame_rows if isinstance(r.get("centroid_shift_norm"), float) and not math.isnan(r["centroid_shift_norm"])]
    def _np_percentile(vals, q):
        if not vals:
            return math.nan
        arr = np.array(vals, dtype=np.float32)
        try:
            return float(np.percentile(arr, q))
        except Exception:
            return math.nan
    jitter_med = _np_percentile(jitter_list, 50)
    jitter_p95 = _np_percentile(jitter_list, 95)
    # Area CV%%
    area_vals = [r["area_px"] for r in frame_rows] if frame_rows else []
    area_cv_pct = math.nan
    if area_vals:
        arr_area = np.array(area_vals, dtype=np.float32)
        m = float(np.mean(arr_area))
        if m > 0:
            area_cv_pct = float(np.std(arr_area) / m * 100.0)
    # Empty frames
    empty_frames_pct = (sum(1 for r in frame_rows if r["is_empty"] == 1) / len(frame_rows) * 100.0) if frame_rows else math.nan
    # FPS
    processed_count = len(frame_runtimes)
    total_time_s = sum(frame_runtimes.values()) / 1000.0 if frame_runtimes else 0.0
    proc_fps_measured = processed_count / total_time_s if total_time_s > 0 else 0.0
    proc_fps_theoretical = fps / stride
    # Write summary CSV with fix2 fields
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
    args = parse_args()
    out_root = Path(args.out_dir)
    ensure_dir(out_root)
    src = Path(args.data)
    sources = list_sources(src if src.exists() else src)
    if not sources:
        raise SystemExit(f"No sources found under {args.data}")

    results: List[Dict[str, object]] = []
    for p in sources:
        print(f"Processing video: {p}")
        res = process_video(p, args, out_root)
        print(json.dumps(res, indent=2))
        results.append(res)

    summary = out_root / "summary.csv"
    with summary.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary written to {summary}")


if __name__ == "__main__":
    main()




