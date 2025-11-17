import json
from pathlib import Path


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def convert_frame(json_path: Path, out_dir: Path, negatives: bool = True) -> bool:
    """Convert a single COCO-like per-frame JSON into a YOLO txt (single union box per frame).
    Returns True if a label file was written.
    """
    try:
        j = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    image = j.get("image", {})
    W = int(image.get("width", 0))
    H = int(image.get("height", 0))
    if W <= 0 or H <= 0:
        return False
    anns = j.get("annotations", []) or []
    x0u = y0u = None
    x1u = y1u = None
    for ann in anns:
        bbox = ann.get("bbox", None)
        if not bbox or len(bbox) < 4:
            continue
        x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        if w <= 0.0 or h <= 0.0:
            continue
        x0, y0, x1, y1 = x, y, x + w, y + h
        x0u = x0 if x0u is None else min(x0u, x0)
        y0u = y0 if y0u is None else min(y0u, y0)
        x1u = x1 if x1u is None else max(x1u, x1)
        y1u = y1 if y1u is None else max(y1u, y1)

    ensure_dir(out_dir)
    out_txt = out_dir / (json_path.stem + ".txt")
    if x0u is not None and y0u is not None and x1u is not None and y1u is not None:
        x0u = clamp(x0u, 0.0, float(W - 1))
        y0u = clamp(y0u, 0.0, float(H - 1))
        x1u = clamp(x1u, 0.0, float(W - 1))
        y1u = clamp(y1u, 0.0, float(H - 1))
        w = max(0.0, x1u - x0u)
        h = max(0.0, y1u - y0u)
        if w > 0.0 and h > 0.0:
            cx = (x0u + w * 0.5) / float(W)
            cy = (y0u + h * 0.5) / float(H)
            nw = w / float(W)
            nh = h / float(H)
            out_txt.write_text(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
            return True
    if negatives:
        out_txt.write_text("")
        return True
    return False
