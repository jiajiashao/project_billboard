from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Any

# Minimal, dependency-light helper for Moondream auto-prompting.
# - No printing and no file I/O.
# - Assumes public hub id; caller ensures transformers/torch availability.

PROMPT_TERMS: List[str] = [
    "billboard",
    "perimeter billboard",
    "LED ad board",
    "stadium advertising board",
    "sideline banner",
]


@dataclass
class BoxPrediction:
    label: str
    score: float
    x0: float
    y0: float
    x1: float
    y1: float

    def as_int_tuple(self) -> Tuple[int, int, int, int]:
        return int(round(self.x0)), int(round(self.y0)), int(round(self.x1)), int(round(self.y1))


class MoondreamBoxPromptor:
    def __init__(
        self,
        model_id: str = "vikhyatk/moondream2",
        device: Optional[str] = None,           # "cuda" / "cpu" / None->auto
        threshold: float = 0.10,                # keep boxes with score >= threshold
        prompts: Optional[List[str]] = None,    # default defined above
        enable_encode_cache: bool = True,       # reuse encode_image() if available
    ) -> None:
        from transformers import AutoModelForCausalLM  # lazy import
        import torch  # type: ignore

        if device is None:
            device = "cuda" if getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"
        self.device = device
        self.threshold = float(threshold)
        self.prompts = list(prompts) if prompts else list(PROMPT_TERMS)
        self.enable_encode_cache = bool(enable_encode_cache)

        # trust_remote_code is required by Moondream.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map={"": device},
        ).eval()

    def __call__(self, frame_bgr) -> Optional[BoxPrediction]:
        """Accepts a single OpenCV BGR frame (numpy HxWx3, uint8).
        Returns best BoxPrediction or None if nothing passes filters.
        """
        try:
            import numpy as np  # type: ignore
            import cv2  # type: ignore
            from PIL import Image  # type: ignore
        except Exception:
            # If required deps are missing, treat as no prediction.
            return None

        if frame_bgr is None:
            return None
        arr = np.asarray(frame_bgr)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return None
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)

        # Convert BGR -> RGB for model consumption.
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        W, H = pil_img.size
        if W <= 0 or H <= 0:
            return None

        model = self.model
        best: Optional[BoxPrediction] = None

        # Optional encode cache for efficiency.
        enc: Any = None
        if self.enable_encode_cache and hasattr(model, "encode_image"):
            try:
                enc = model.encode_image(pil_img)
            except Exception:
                enc = None

        for term in self.prompts:
            try:
                if enc is not None and hasattr(model, "detect"):
                    res = model.detect(enc, term)
                else:
                    res = model.detect(pil_img, term) if hasattr(model, "detect") else None
            except Exception:
                res = None
            if not res or not isinstance(res, dict):
                continue
            objects = res.get("objects")
            if not isinstance(objects, list):
                continue
            for obj in objects:
                try:
                    x0n = float(obj.get("x_min", 0.0))
                    y0n = float(obj.get("y_min", 0.0))
                    x1n = float(obj.get("x_max", 0.0))
                    y1n = float(obj.get("y_max", 0.0))
                    score = float(obj.get("score", 0.5))
                except Exception:
                    continue
                # Clamp normalized coords to [0,1].
                x0n = max(0.0, min(1.0, x0n))
                y0n = max(0.0, min(1.0, y0n))
                x1n = max(0.0, min(1.0, x1n))
                y1n = max(0.0, min(1.0, y1n))
                # Convert to absolute pixels.
                x0 = x0n * W
                y0 = y0n * H
                x1 = x1n * W
                y1 = y1n * H
                # Basic sanity checks.
                if not (x1 > x0 and y1 > y0):
                    continue
                bw = x1 - x0
                bh = y1 - y0
                if not (0.01 * W <= bw <= 0.95 * W and 0.01 * H <= bh <= 0.95 * H):
                    continue
                if score < self.threshold:
                    continue
                cand = BoxPrediction(label=term, score=score, x0=x0, y0=y0, x1=x1, y1=y1)
                if best is None or cand.score > best.score:
                    best = cand
        return best
