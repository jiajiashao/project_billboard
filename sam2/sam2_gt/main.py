# Standard library imports for path manipulation and system path management
import sys
from pathlib import Path
from tkinter import FALSE

# Path setup block: Ensures the project root directory is in Python's module search path
# This allows imports from sam2_smoke and sam2_pilot modules to resolve correctly
# ROOT points to the parent directory of the current file (sam2/)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Add current directory for local imports
# THIS_DIR points to the directory containing this file (sam2_gt/)
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# Import the original Fix-2 script unchanged
# This imports the base SAM-2 implementation that contains the core functionality
import sam2_base as base


def parse_args() -> base.argparse.Namespace:
    """CLI with CUDA enabled; mirrors Fix-2 defaults."""
    # Command-line argument parser setup
    # Creates an argument parser for the SAM-2 pilot runner with CUDA support
    parser = base.argparse.ArgumentParser(description="SAM-2 pilot runner for Fix-2 (CUDA-enabled)")
    # Data root directory: where input clips and ground truth frames are stored
    parser.add_argument("--data-root", dest="data_root", default="./../data")
    # Model weights: specifies which SAM-2 model variant to use (default is tiny hiera model)
    parser.add_argument("--weights", default="facebook/sam2.1-hiera-tiny")
    # Runs root directory: where output results will be saved
    parser.add_argument("--runs-root", dest="runs_root", default="runs")
    # Clips: optional list of specific clip IDs to process (if not provided, processes all)
    parser.add_argument("--clips", nargs="*", help="Optional subset of clip IDs to process")
    # Device selection: choose between CPU, CUDA (GPU), or MPS (Apple Silicon)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda")
    # Reseed flag: enables reseeding functionality during tracking
    parser.add_argument("--reseed", action="store_true", default=False)
    return parser.parse_args()


def select_device(preferred: str | None) -> str:
    """CUDA-aware device selection compatible with Fix-2 semantics."""
    # Device selection logic: handles user preference and fallback scenarios
    if preferred:
        # If user specified a device, validate it's available before using it
        # Check if CUDA was requested but not available - fallback to CPU
        if preferred == "cuda" and not base.torch.cuda.is_available():
            print("Requested cuda but unavailable; falling back to cpu")
            return "cpu"
        # Check if MPS (Apple Silicon) was requested but not available - fallback to CPU
        if preferred == "mps" and not base.torch.backends.mps.is_available():
            print("Requested mps but unavailable; falling back to cpu")
            return "cpu"
        # If preferred device is available, use it
        return preferred
    # Auto-detection: if no preference specified, choose best available device
    # Priority: CUDA > MPS > CPU
    if base.torch.cuda.is_available():
        return "cuda"
    return "mps" if base.torch.backends.mps.is_available() else "cpu"

# Type hints for function parameters and return values
from typing import List, Dict

def _default_clip_cfg(cid: str) -> Dict:
    # Fast-like default config mirroring Fix-2 fast profile
    # Creates a default configuration dictionary for a clip that wasn't preconfigured
    # Extracts reseed triggers from existing clip configs (prefers second clip if available)
    trig = base.RUN_SPEC["clips"][1]["reseed"]["triggers"] if len(base.RUN_SPEC.get("clips", [])) > 1 else base.RUN_SPEC["clips"][0]["reseed"]["triggers"]
    return {
        "id": cid,  # Clip identifier
        "input_width": 1280,  # Frame width for processing
        "stride": 1,  # Process every frame (stride of 1 means no frame skipping)
        "full_frame": True,  # Process the entire frame
        "seed": {
            # Seed configuration: how to initialize tracking on the first frame
            "mode": "first_labeled_frame",  # Use the first frame with ground truth labels
            "from_gt_bbox": True,  # Initialize from ground truth bounding box
            "bbox_pad_px": 6,  # Padding around bounding box in pixels
            "negatives": {"mode": "edge_fence", "count": 4, "offset_px": 6},  # Negative point sampling strategy
        },
        "reseed": {
            # Reseed configuration: how to reinitialize tracking when it fails
            "enabled": False,  # Reseeding is disabled by default
            "triggers": trig,  # Conditions that trigger reseeding (from existing config)
            "action": "reseed_with_box_plus_neg",  # Action to take when reseeding
            "cooldown_frames": 0,  # Minimum frames between reseed events
            "max_events": 0,  # Maximum number of reseed events allowed
        },
    }

def _ensure_requested_in_runspec(requested: List[str], data_root: Path) -> None:
    # Ensures that all requested clip IDs are present in RUN_SPEC configuration
    # If a clip is requested but not preconfigured, adds it with default settings
    if not requested:
        return  # No clips requested, nothing to do
    # Get set of existing clip IDs that are already configured
    existing = {cfg.get("id") for cfg in base.RUN_SPEC.get("clips", [])}
    # Process each requested clip ID
    for cid in requested:
        if cid in existing:
            continue  # Clip already configured, skip
        # Check if required input files exist for this clip
        mp4 = data_root / "clips" / f"{cid}.mp4"  # Video file path
        gt = data_root / "gt_frames" / cid  # Ground truth frames directory
        # Validate that both video and ground truth data exist
        if not mp4.exists() or not gt.exists():
            print(f'Skipping unknown clip {cid!r}: inputs not found under {data_root}')
            continue
        # Add the clip to RUN_SPEC with default configuration
        base.RUN_SPEC["clips"].append(_default_clip_cfg(cid))

if __name__ == "__main__":
    # Main execution block: sets up configuration and runs SAM-2 tracking
    
    # Step 1: Parse command-line arguments to get user preferences
    _args_preview = parse_args()
    req = _args_preview.clips or []  # Extract requested clip IDs (empty list if none specified)
    
    # Step 2: Unify all preconfigured clip configs to a single template so every clip is treated the same
    # This ensures consistent processing parameters across all clips
    try:
        _clips = base.RUN_SPEC.get("clips", [])  # Get all configured clips
        _tmpl = None  # Template configuration to use for unification
        
        # Find a template clip: prefer one with stride=1 (processes every frame)
        for _cfg in _clips:
            try:
                if int(_cfg.get("stride", 0)) == 1:
                    _tmpl = _cfg
                    break
            except Exception:
                pass
        
        # Fallback: if no stride=1 clip found, use the first clip as template
        if _tmpl is None and _clips:
            _tmpl = _clips[0]
        
        # Step 3: Apply template settings to all clip configurations
        # This standardizes input_width, stride, seed, and reseed settings across all clips
        if _tmpl is not None:
            for _cfg in _clips:
                try:
                    # Unify input_width: use template's width or keep existing/default
                    _cfg["input_width"] = int(_tmpl.get("input_width", _cfg.get("input_width", 1280)))
                    # Unify stride: use template's stride or keep existing/default
                    _cfg["stride"] = int(_tmpl.get("stride", _cfg.get("stride", 1)))
                    
                    # Unify seed configuration: copy seed settings from template
                    _tseed = dict(_tmpl.get("seed", {}))  # Template seed config
                    _cseed = dict(_cfg.get("seed", {}))  # Current clip seed config
                    # Copy specific seed keys from template to current clip
                    for _k in ("mode", "from_gt_bbox", "bbox_pad_px", "negatives"):
                        if _k in _tseed:
                            _cseed[_k] = _tseed[_k]
                    _cfg["seed"] = _cseed
                    
                    # Unify reseed configuration: copy reseed settings from template
                    _trs = dict(_tmpl.get("reseed", {}))  # Template reseed config
                    _crs = dict(_cfg.get("reseed", {}))  # Current clip reseed config
                    # Copy specific reseed keys from template to current clip
                    for _k in ("enabled", "triggers", "action", "cooldown_frames", "max_events"):
                        if _k in _trs:
                            _crs[_k] = _trs[_k]
                    _cfg["reseed"] = _crs
                except Exception:
                    pass  # Skip clips that cause errors during unification
    except Exception:
        pass  # Continue even if unification fails
    
    # Step 4: Monkey-patch only the CLI and device selection functions
    # Replace the base module's functions with our custom versions
    # This allows us to customize argument parsing and device selection without modifying base code
    base.parse_args = parse_args  # type: ignore[attr-defined]
    base.select_device = select_device  # type: ignore[attr-defined]
    
    # Step 5: Call the main function from base module to start processing
    # This executes the actual SAM-2 tracking pipeline with our custom configurations
    base.main()


