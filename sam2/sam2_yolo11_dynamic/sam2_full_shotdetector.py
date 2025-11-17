import sys
from pathlib import Path

# Ensure project root (parent of experiments/) is importable so sam2_smoke/sam2_pilot resolve
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the original Fix-2 script unchanged
import sam2_pilot_fix2 as base
from shot_detection import detect_shots, ShotSegment


def parse_args() -> base.argparse.Namespace:
    """CLI with CUDA enabled; mirrors Fix-2 defaults."""
    parser = base.argparse.ArgumentParser(description="SAM-2 pilot runner for Fix-2 (CUDA-enabled)")
    parser.add_argument("--data-root", dest="data_root", default="data")
    parser.add_argument("--weights", default="models/sam2.1-hiera-tiny")
    parser.add_argument("--runs-root", dest="runs_root", default="runs")
    parser.add_argument("--clips", nargs="*", help="Optional subset of clip IDs to process")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
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
            "enabled": True,
            "triggers": trig,
            "action": "reseed_with_box_plus_neg",
            "cooldown_frames": 3,
            "max_events": 100,
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


def _install_shot_detector_hook() -> None:
    """Monkey-patch base.load_frames_and_seed to compute shot boundaries.
    Keeps Fix-2 core behavior intact; only attaches shot_bounds to clip_cfg.
    """
    orig = base.load_frames_and_seed
    def wrapped(clip_cfg, clip_path, gt_dir, target_width):
        frames, size, fps, seed_idx, seed_json, box, points, labels, gt_boxes = orig(clip_cfg, clip_path, gt_dir, target_width)
        try:
            shots = detect_shots(
                clip_path,
                total_frames=len(frames),
                fps=fps,
                method="adaptive",
                min_shot_len_s=1,
                adaptive_sensitivity=3,
            )
            clip_cfg["shot_bounds"] = [(int(s.start), int(s.end)) for s in shots]
        except Exception:
            clip_cfg["shot_bounds"] = [(0, int(len(frames)))]
        return frames, size, fps, seed_idx, seed_json, box, points, labels, gt_boxes
    base.load_frames_and_seed = wrapped  # type: ignore[attr-defined]

if __name__ == "__main__":
    # Extend RUN_SPEC with any requested clip IDs not preconfigured
    _args_preview = parse_args()
    req = _args_preview.clips or []
    _install_shot_detector_hook()
    _ensure_requested_in_runspec(req, Path(_args_preview.data_root))
    # Monkey-patch only the CLI and device selection; leave everything else identical
    base.parse_args = parse_args  # type: ignore[attr-defined]
    base.select_device = select_device  # type: ignore[attr-defined]
    base.main()



