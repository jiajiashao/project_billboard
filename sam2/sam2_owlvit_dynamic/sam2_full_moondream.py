import sys
from pathlib import Path

# Ensure project root (parent of experiments/) is importable so sam2_smoke/sam2_pilot resolve
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the original Fix-2 script unchanged
import sam2_pilot_fix2 as base
from shot_detection import detect_shots, ShotSegment
from autoprompt_moondream import MoondreamBoxPromptor, PROMPT_TERMS


def parse_args() -> base.argparse.Namespace:
    """CLI with CUDA enabled; mirrors Fix-2 defaults."""
    parser = base.argparse.ArgumentParser(description="SAM-2 pilot runner for Fix-2 (CUDA-enabled)")
    parser.add_argument("--data-root", dest="data_root", default="data")
    parser.add_argument("--weights", default="models/sam2.1-hiera-tiny")
    parser.add_argument("--runs-root", dest="runs_root", default="runs")
    parser.add_argument("--clips", nargs="*", help="Optional subset of clip IDs to process")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--auto-prompt", action="store_true", default=False)
    parser.add_argument("--moondream-model", default="vikhyatk/moondream2")
    parser.add_argument("--moondream-device", default=None)
    parser.add_argument("--moondream-threshold", type=float, default=0.10)
    parser.add_argument("--autoprompt-fallback", default="skip")
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
                min_shot_len_s=1.0,
                adaptive_sensitivity=3,
            )
            clip_cfg["shot_bounds"] = [(int(s.start), int(s.end)) for s in shots]
        except Exception:
            clip_cfg["shot_bounds"] = [(0, int(len(frames)))]
        return frames, size, fps, seed_idx, seed_json, box, points, labels, gt_boxes
    base.load_frames_and_seed = wrapped  # type: ignore[attr-defined]

# Internal shared state for autoprompt hooks
_AP_STATE = {"args": None, "promptor": None, "sessions": {}, "last_clip_ctx": None, "log": None}

def _install_autoprompt_hooks(args) -> None:
    """Install hooks to disable reseeding and inject Moondream seeds per shot."""
    _AP_STATE["args"] = args
    promptor = MoondreamBoxPromptor(
        model_id=args.moondream_model,
        device=args.moondream_device or None,
        threshold=float(args.moondream_threshold),
        prompts=PROMPT_TERMS,
    ) if args.auto_prompt else None
    _AP_STATE["promptor"] = promptor

    # Patch logger to capture log function for this clip
    orig_create_logger = base.create_logger
    def create_logger_wrapper():
        lines, log = orig_create_logger()
        _AP_STATE["log"] = log
        return lines, log
    base.create_logger = create_logger_wrapper  # type: ignore[attr-defined]

    # Patch load_frames_and_seed to stash context and disable reseeding
    orig_lfs = base.load_frames_and_seed
    def load_frames_and_seed_wrapper(clip_cfg, clip_path, gt_dir, target_width):
        frames, size, fps, seed_idx, seed_json, box, points, labels, gt_boxes = orig_lfs(clip_cfg, clip_path, gt_dir, target_width)
        try:
            clip_cfg.setdefault("reseed", {})["enabled"] = False
        except Exception:
            pass
        _AP_STATE["last_clip_ctx"] = {
            "clip_cfg": clip_cfg,
            "frames": frames,
            "fps": fps,
            "seed_idx": seed_idx,
            "seed_points": points,
            "seed_labels": labels,
        }
        return frames, size, fps, seed_idx, seed_json, box, points, labels, gt_boxes
    base.load_frames_and_seed = load_frames_and_seed_wrapper  # type: ignore[attr-defined]

    # Patch processor methods to attach per-session context and inject seeds
    Proc = base.Sam2VideoProcessor
    orig_init_sess = Proc.init_video_session
    def init_video_session_wrapper(self, *args, **kwargs):
        session = orig_init_sess(self, *args, **kwargs)
        ctx = _AP_STATE.get("last_clip_ctx")
        if ctx is not None:
            _AP_STATE["sessions"][id(session)] = dict(ctx)
            _AP_STATE["sessions"][id(session)]["autoprompt_done"] = False
        return session
    Proc.init_video_session = init_video_session_wrapper  # type: ignore[attr-defined]

    orig_add_inputs = Proc.add_inputs_to_inference_session
    def add_inputs_wrapper(self, *, inference_session, frame_idx, obj_ids, input_boxes, input_points=None, input_labels=None):
        res = orig_add_inputs(self, inference_session=inference_session, frame_idx=frame_idx, obj_ids=obj_ids, input_boxes=input_boxes, input_points=input_points, input_labels=input_labels)
        sess_id = id(inference_session)
        ctx = _AP_STATE["sessions"].get(sess_id)
        args_local = _AP_STATE.get("args")
        promptor_local = _AP_STATE.get("promptor")
        log_fn = _AP_STATE.get("log")
        if ctx and args_local and args_local.auto_prompt and promptor_local and not ctx.get("autoprompt_done") and frame_idx == ctx.get("seed_idx"):
            clip_cfg = ctx.get("clip_cfg") or {}
            shots = clip_cfg.get("shot_bounds") or []
            frames = ctx.get("frames") or []
            seed_points = ctx.get("seed_points")
            seed_labels = ctx.get("seed_labels")
            try:
                import numpy as np
                import cv2
            except Exception:
                ctx["autoprompt_done"] = True
                return res
            total = len(shots)
            for i, (s, e) in enumerate(shots, start=1):
                frame = None
                if isinstance(frames, list) and 0 <= s < len(frames):
                    fr = frames[s]
                    try:
                        arr = np.asarray(fr)
                        if arr.ndim == 3 and arr.shape[2] == 3:
                            bgr = cv2.cvtColor(arr.astype(np.uint8, copy=False), cv2.COLOR_RGB2BGR)
                            frame = bgr
                    except Exception:
                        frame = None
                pred = None
                if frame is not None:
                    try:
                        pred = promptor_local(frame)
                    except Exception:
                        pred = None
                if pred is None:
                    if callable(log_fn):
                        log_fn(f"Shot {i}/{total}: autoprompt NONE - fallback={args_local.autoprompt_fallback} - skipping frames {s}-{max(s, e-1)}")
                    continue
                x0, y0, x1, y1 = pred.as_int_tuple()
                new_boxes = [[[int(x0), int(y0), int(x1), int(y1)]]]
                try:
                    orig_add_inputs(self, inference_session=inference_session, frame_idx=int(s), obj_ids=obj_ids, input_boxes=new_boxes, input_points=seed_points, input_labels=seed_labels)
                except Exception:
                    pass
                if callable(log_fn):
                    log_fn(f"Shot {i}/{total}: autoprompt ok (label={pred.label}, score={pred.score:.2f}) - seeding @ frame {s}")
            ctx["autoprompt_done"] = True
        return res
    Proc.add_inputs_to_inference_session = add_inputs_wrapper  # type: ignore[attr-defined]
if __name__ == "__main__":
    # Extend RUN_SPEC with any requested clip IDs not preconfigured
    _args_preview = parse_args()
    req = _args_preview.clips or []
    if _args_preview.auto_prompt:
        _install_autoprompt_hooks(_args_preview)
    _install_shot_detector_hook()
    _ensure_requested_in_runspec(req, Path(_args_preview.data_root))
    # Monkey-patch only the CLI and device selection; leave everything else identical
    base.parse_args = parse_args  # type: ignore[attr-defined]
    base.select_device = select_device  # type: ignore[attr-defined]
    base.main()









