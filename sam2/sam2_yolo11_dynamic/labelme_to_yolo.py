"""
Convert LabelMe JSON polygons to YOLO txt labels (single class 'billboard').

Usage examples:
  - Single clip folder -> specific output folder:
      python labelme_to_yolo.py --src "D:\\Billboard_Project\\sam2\\data\\gt_frames\\clip_basketball" \
                                --out "D:\\billboard_yolo\\labels\\train\\clip_basketball"

  - Root containing many clips -> mirror subfolders under --out:
      python labelme_to_yolo.py --src "D:\\Billboard_Project\\sam2\\data\\gt_frames" \
                                --out "D:\\billboard_yolo\\labels\\train"
"""

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np


def poly_bbox(xy):
    xy = np.array(xy, dtype=float)
    x0, y0 = xy.min(0)
    x1, y1 = xy.max(0)
    return x0, y0, x1, y1


def is_clip_dir(p: Path) -> bool:
    return p.is_dir() and any(p.glob("frame_*.json"))


def iter_clip_dirs(root: Path) -> Iterable[Path]:
    """Yield clip directories that contain frame_*.json files.
    Handles the case where root itself is a clip dir, or contains many clip dirs.
    """
    if is_clip_dir(root):
        yield root
        return
    for child in sorted(root.iterdir()):
        if is_clip_dir(child):
            yield child


def convert_clip(clip_dir: Path, out_base: Path, class_name: str = "billboard", negatives: bool = True) -> int:
    """Convert all LabelMe JSONs in clip_dir into YOLO txts under out_base/clip_dir.name.
    Returns number of frames written (including negatives)."""
    out_dir = out_base / clip_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for jp in sorted(clip_dir.glob("frame_*.json")):
        try:
            with jp.open("r", encoding="utf-8") as f:
                j = json.load(f)
        except Exception:
            continue
        W, H = int(j.get("imageWidth", 0)), int(j.get("imageHeight", 0))
        if W <= 0 or H <= 0:
            continue
        # Collect only billboard polygons by label (case-insensitive substring match)
        boxes = []
        for s in j.get("shapes", []):
            lbl = str(s.get("label", ""))
            if class_name.lower() not in lbl.lower():
                continue
            pts = s.get("points")
            if pts:
                x0, y0, x1, y1 = poly_bbox(pts)
                boxes.append([x0, y0, x1, y1])
        out_txt = out_dir / (jp.stem.replace("frame_", "") + ".txt")
        if not boxes:
            if negatives:
                out_txt.write_text("")
                count += 1
            continue
        # Union all polygons -> 1 box
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[2] for b in boxes)
        y1 = max(b[3] for b in boxes)
        # Clamp
        x0 = max(0.0, min(float(W - 1), x0)); x1 = max(0.0, min(float(W - 1), x1))
        y0 = max(0.0, min(float(H - 1), y0)); y1 = max(0.0, min(float(H - 1), y1))
        # Convert to YOLO normalized cx,cy,w,h
        cx = ((x0 + x1) / 2.0) / W
        cy = ((y0 + y1) / 2.0) / H
        w = max(1.0, (x1 - x0)) / W
        h = max(1.0, (y1 - y0)) / H
        out_txt.write_text(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert LabelMe JSON to YOLO labels (single class)")
    ap.add_argument("--src", required=True, help="Path to a clip dir (with frame_*.json) or a root containing multiple clips")
    ap.add_argument("--out", required=True, help="Output root for YOLO labels; clip subfolders are created here")
    ap.add_argument("--class-name", default="billboard", help="Class name filter (substring match, default 'billboard')")
    ap.add_argument("--negatives", action="store_true", help="Write empty txt for frames without objects")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    total = 0
    for clip_dir in iter_clip_dirs(src):
        n = convert_clip(clip_dir, out, class_name=args.class_name, negatives=args.negatives)
        print(f"Converted {n} frames in {clip_dir} -> {out/clip_dir.name}")
        total += n
    if total == 0:
        print("No labels written: check --src path (should be a clip folder with frame_*.json or a root containing such folders).")


if __name__ == "__main__":
    main()
