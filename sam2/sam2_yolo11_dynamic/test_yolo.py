import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test a trained YOLO model on videos and save overlays/CSV")
    p.add_argument("--model", required=True, help="Path to YOLO .pt weights (e.g., runs/detect/trainN/weights/best.pt)")
    p.add_argument("--source", required=True, help="Video file or directory containing videos/images (recurses)")
    p.add_argument("--device", default="0", help="YOLO device spec, e.g. '0' for CUDA:0")
    p.add_argument("--conf", type=float, default=0.20, help="Confidence threshold for detections")
    p.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    p.add_argument("--max-det", type=int, default=3, help="Max detections per frame")
    p.add_argument("--stride", type=int, default=1, help="Process every Nth frame (speed-up)")
    p.add_argument("--out-dir", default="runs/yolo_test", help="Output directory for overlays/CSVs")
    p.add_argument("--save-txt", action="store_true", help="Also save YOLO-format txt per frame (optional)")
    return p.parse_args()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_sources(src: Path) -> List[Path]:
    if src.is_file():
        return [src]
    patterns = ["*.mp4", "*.mkv", "*.mov", "*.avi", "*.m4v", "*.mpg", "*.mpeg", "*.wmv", "*.ts", "*.webm",
                "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"]
    out: List[Path] = []
    for pat in patterns:
        out.extend(src.rglob(pat))
    return sorted(out)


def draw_boxes(bgr: np.ndarray, boxes: List[Tuple[int, int, int, int]], scores: List[float]) -> np.ndarray:
    img = bgr.copy()
    for (x0, y0, x1, y1), sc in zip(boxes, scores):
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 0), 2)
        label = f"billboard {sc:.2f}"
        cv2.putText(img, label, (x0, max(0, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (x0, max(0, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def run_yolo_on_frame(model, frame_bgr: np.ndarray, conf: float, imgsz: int, device: str, max_det: int) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
    res = model.predict(source=frame_bgr, imgsz=imgsz, conf=conf, device=device, verbose=False, max_det=max_det)
    boxes: List[Tuple[int, int, int, int]] = []
    scores: List[float] = []
    if not res:
        return boxes, scores
    r = res[0]
    battr = getattr(r, "boxes", None)
    if battr is None or getattr(battr, "xyxy", None) is None:
        return boxes, scores
    xyxy = battr.xyxy.detach().cpu().numpy()
    confs = battr.conf.detach().cpu().numpy() if getattr(battr, "conf", None) is not None else np.zeros((xyxy.shape[0],), dtype=np.float32)
    # order by score desc
    order = np.argsort(-confs)
    for i in order[: int(max(1, max_det))]:
        x0, y0, x1, y1 = [int(round(float(v))) for v in xyxy[i].tolist()]
        boxes.append((x0, y0, x1, y1))
        scores.append(float(confs[i]))
    return boxes, scores


def process_video(model, src_path: Path, out_dir: Path, conf: float, imgsz: int, device: str, max_det: int, stride: int, save_txt: bool) -> Dict[str, object]:
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {src_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    base = src_path.stem
    vid_out = out_dir / f"overlay_{base}_yolo.mp4"
    writer = cv2.VideoWriter(str(vid_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    csv_rows: List[Dict[str, object]] = []
    frame_idx = 0
    processed = 0
    t0 = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if stride > 1 and (frame_idx % stride != 0):
                writer.write(frame)
                frame_idx += 1
                continue

            boxes, scores = run_yolo_on_frame(model, frame, conf=conf, imgsz=imgsz, device=device, max_det=max_det)
            overlay = draw_boxes(frame, boxes, scores)
            writer.write(overlay)

            row = {
                "source": str(src_path),
                "frame": frame_idx,
                "n_boxes": len(boxes),
                "boxes": json.dumps(boxes),
                "scores": ",".join(f"{s:.3f}" for s in scores),
            }
            csv_rows.append(row)
            if save_txt:
                txt_dir = out_dir / "labels" / base
                ensure_dir(txt_dir)
                txt_path = txt_dir / f"frame_{frame_idx:06d}.txt"
                with txt_path.open("w") as f:
                    for (x0, y0, x1, y1) in boxes:
                        cx = (x0 + x1) / (2.0 * W)
                        cy = (y0 + y1) / (2.0 * H)
                        bw = (x1 - x0) / float(W)
                        bh = (y1 - y0) / float(H)
                        f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            processed += 1
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - t0
    fps_meas = processed / elapsed if elapsed > 0 else 0.0

    import csv
    csv_path = out_dir / f"per_frame_{base}_yolo.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = ["source", "frame", "n_boxes", "boxes", "scores"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)

    frames_with_det = sum(1 for r in csv_rows if r["n_boxes"] > 0)
    return {
        "video": str(src_path),
        "frames": frame_idx,
        "processed": processed,
        "frames_with_detections": frames_with_det,
        "detection_rate_pct": (frames_with_det / processed * 100.0) if processed else 0.0,
        "fps": fps_meas,
        "overlay_path": str(vid_out),
        "csv_path": str(csv_path),
    }


def process_image(model, img_path: Path, out_dir: Path, conf: float, imgsz: int, device: str, max_det: int) -> Dict[str, object]:
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"Could not read image: {img_path}")
    H, W = bgr.shape[:2]
    boxes, scores = run_yolo_on_frame(model, bgr, conf=conf, imgsz=imgsz, device=device, max_det=max_det)
    overlay = draw_boxes(bgr, boxes, scores)
    base = img_path.stem
    out_img = out_dir / f"pred_{base}.jpg"
    cv2.imwrite(str(out_img), overlay)
    return {
        "image": str(img_path),
        "n_boxes": len(boxes),
        "boxes": boxes,
        "scores": scores,
        "overlay_path": str(out_img),
    }


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception:
        raise SystemExit("Missing dependency: ultralytics. Install with: pip install ultralytics")

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    model = YOLO(args.model)

    src = Path(args.source)
    sources = list_sources(src if src.exists() else src)
    if not sources:
        raise SystemExit(f"No sources found under {args.source}")

    results: List[Dict[str, object]] = []
    for item in sources:
        if item.suffix.lower() in {".mp4", ".mkv", ".mov", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv", ".ts", ".webm"}:
            print(f"Processing video: {item}")
            res = process_video(model, item, out_dir, conf=args.conf, imgsz=args.imgsz, device=args.device, max_det=args.max_det, stride=args.stride, save_txt=args.save_txt)
            print(json.dumps(res, indent=2))
            results.append(res)
        else:
            print(f"Processing image: {item}")
            res = process_image(model, item, out_dir, conf=args.conf, imgsz=args.imgsz, device=args.device, max_det=args.max_det)
            print(json.dumps(res, indent=2))
            results.append(res)

    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
