import sys
from pathlib import Path
from tkinter import FALSE

# Ensure project root (parent of experiments/) is importable so sam2_smoke/sam2_pilot resolve
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Add current directory for local imports
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# Import the original Fix-2 script unchanged
import sam2_base as base


def parse_args() -> base.argparse.Namespace:
    """CLI with CUDA enabled; mirrors Fix-2 defaults."""
    parser = base.argparse.ArgumentParser(description="SAM-2 pilot runner for Fix-2 (CUDA-enabled)")
    parser.add_argument("--data-root", dest="data_root", default="..\sam2\data")
    parser.add_argument("--weights", default="facebook/sam2.1-hiera-tiny")
    parser.add_argument("--runs-root", dest="runs_root", default="runs")
    parser.add_argument("--clips", nargs="*", help="Optional subset of clip IDs to process")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--reseed", action="store_true", default=False)
    return parser.parse_args()


def select_device(preferred: str | None) -> str:
    """CUDA-aware device selection compatible with Fix-2 semantics."""
    if preferred:
        if preferred == "cuda" and not base.torch.cuda.is_available():
            print("Requested cuda but unavailable; falling back to cpu")
            return "cpu"
        if preferred == "mps" and not base.torch.backends.mps.is_available():
            print("Requested mps but unavailable; falling back to cpu")
            return "cpu"
        return preferred
    if base.torch.cuda.is_available():
        return "cuda"
    return "mps" if base.torch.backends.mps.is_available() else "cpu"

from typing import List, Dict

def _default_clip_cfg(cid: str) -> Dict:
    # Fast-like default config mirroring Fix-2 fast profile
    trig = base.RUN_SPEC["clips"][1]["reseed"]["triggers"] if len(base.RUN_SPEC.get("clips", [])) > 1 else base.RUN_SPEC["clips"][0]["reseed"]["triggers"]
    return {
        "id": cid,
        "input_width": 1280,
        "stride": 1,
        "full_frame": True,
        "seed": {
            "mode": "first_labeled_frame",
            "from_gt_bbox": True,
            "bbox_pad_px": 6,
            "negatives": {"mode": "edge_fence", "count": 4, "offset_px": 6},
        },
        "reseed": {
            "enabled": False,
            "triggers": trig,
            "action": "reseed_with_box_plus_neg",
            "cooldown_frames": 0,
            "max_events": 0,
        },
    }

def _ensure_requested_in_runspec(requested: List[str], data_root: Path) -> None:
    if not requested:
        return
    existing = {cfg.get("id") for cfg in base.RUN_SPEC.get("clips", [])}
    for cid in requested:
        if cid in existing:
            continue
        mp4 = data_root / "clips" / f"{cid}.mp4"
        gt = data_root / "gt_frames" / cid
        if not mp4.exists() or not gt.exists():
            print(f'Skipping unknown clip {cid!r}: inputs not found under {data_root}')
            continue
        base.RUN_SPEC["clips"].append(_default_clip_cfg(cid))

if __name__ == "__main__":
    # Extend RUN_SPEC with any requested clip IDs not preconfigured
    _args_preview = parse_args()
    req = _args_preview.clips or []
        # Unify all preconfigured clip configs to a single template so every clip is treated the same
    try:
        _clips = base.RUN_SPEC.get("clips", [])
        _tmpl = None
        for _cfg in _clips:
            try:
                if int(_cfg.get("stride", 0)) == 1:
                    _tmpl = _cfg
                    break
            except Exception:
                pass
        if _tmpl is None and _clips:
            _tmpl = _clips[0]
        if _tmpl is not None:
            for _cfg in _clips:
                try:
                    _cfg["input_width"] = int(_tmpl.get("input_width", _cfg.get("input_width", 1280)))
                    _cfg["stride"] = int(_tmpl.get("stride", _cfg.get("stride", 1)))
                    _tseed = dict(_tmpl.get("seed", {}))
                    _cseed = dict(_cfg.get("seed", {}))
                    for _k in ("mode", "from_gt_bbox", "bbox_pad_px", "negatives"):
                        if _k in _tseed:
                            _cseed[_k] = _tseed[_k]
                    _cfg["seed"] = _cseed
                    _trs = dict(_tmpl.get("reseed", {}))
                    _crs = dict(_cfg.get("reseed", {}))
                    for _k in ("enabled", "triggers", "action", "cooldown_frames", "max_events"):
                        if _k in _trs:
                            _crs[_k] = _trs[_k]
                    _cfg["reseed"] = _crs
                except Exception:
                    pass
    except Exception:
        pass
    # Monkey-patch only the CLI and device selection; leave everything else identical
    base.parse_args = parse_args  # type: ignore[attr-defined]
    base.select_device = select_device  # type: ignore[attr-defined]
    base.main()


