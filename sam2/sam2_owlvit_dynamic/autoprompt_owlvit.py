from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

# OWL-ViT/OWLv2 auto-prompt helper. No prints, no file I/O.
# Exposes PROMPT_TERMS, BoxPrediction, OwlVitBoxPromptor.

PROMPT_TERMS: List[str] = [
    "billboard",
    "perimeter billboard",
    "LED ad board",
    "stadium advertising board",
    "sideline banner",
]


def _is_box_valid_xyxy(x0: int, y0: int, x1: int, y1: int, W: int, H: int) -> bool:
    # area + aspect/coverage sanity checks
    w = max(0, int(x1) - int(x0))
    h = max(0, int(y1) - int(y0))
    if w < 6 or h < 6:
        return False
    area = (w * h) / float(max(1.0, W * H))
    if area < 0.002 or area > 0.70:  # reject tiny and giant boxes
        return False
    ar = w / float(h + 1e-6)
    if ar < 0.1 or ar > 15.0:  # reject extreme aspect ratios
        return False
    # keep box within frame bounds
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        return False
    return True


@dataclass
class BoxPrediction:
    x0: float
    y0: float
    x1: float
    y1: float
    score: float
    label: str

    def as_int_tuple(self, width: Optional[int] = None, height: Optional[int] = None) -> Tuple[int, int, int, int]:
        x0, y0, x1, y1 = self.x0, self.y0, self.x1, self.y1
        if width is not None and height is not None:
            # Clamp to valid image bounds
            x0 = max(0.0, min(float(width - 1), x0))
            y0 = max(0.0, min(float(height - 1), y0))
            x1 = max(0.0, min(float(width - 1), x1))
            y1 = max(0.0, min(float(height - 1), y1))
        return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


class OwlVitBoxPromptor:
    def __init__(
        self,
        model_id: str = "google/owlv2-base-patch16-ensemble",
        device: Optional[str] = None,
        prompts: Optional[List[str]] = None,
        score_thr: float = 0.30,
        nms_iou: float = 0.5,
        min_box_area_frac: float = 0.002,
    ) -> None:
        """Construct an OWL-ViT promptor using Transformers only.

        - model_id: Hugging Face model ID.
        - device: one of {"cuda","cpu","mps"} or None to infer.
        - prompts: list of class strings; defaults to PROMPT_TERMS.
        - score_thr: filter threshold for candidate boxes.
        - nms_iou: IoU threshold for optional torchvision.ops.nms; if torchvision not available, falls back to top-1 per label.
        - min_box_area_frac: extra minimum area fraction gate.
        """
        try:
            import torch  # type: ignore
            try:
                from transformers import Owlv2Processor, Owlv2ForObjectDetection  # type: ignore
                _backend = 'owlv2'
            except Exception:
                from transformers import OwlViTProcessor, OwlViTForObjectDetection  # type: ignore
                _backend = 'owlvit'
        except Exception as e:
            raise RuntimeError("Missing dependency: transformers is required for OWL-ViT/OwlViT") from e

        self._backend = _backend
        self.prompts = prompts if (prompts and len(prompts) > 0) else PROMPT_TERMS
        self.score_thr = float(score_thr)
        self.nms_iou = float(nms_iou)
        self.min_box_area_frac = float(min_box_area_frac)

        # Resolve device
        if device in ("cuda", "cpu", "mps"):
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"        # Cache processor and model on instance
        self._torch = torch
        if self._backend == "owlv2":
            _mdl_id = model_id
            self._processor = Owlv2Processor.from_pretrained(_mdl_id)
            self._model = Owlv2ForObjectDetection.from_pretrained(_mdl_id)
        else:
            _mdl_id = model_id
            if isinstance(_mdl_id, str) and ('owlv2' in _mdl_id.lower()):
                _mdl_id = 'google/owlvit-base-patch16'
            self._processor = OwlViTProcessor.from_pretrained(_mdl_id)
            self._model = OwlViTForObjectDetection.from_pretrained(_mdl_id)
        try:
            self._model.to(self.device)
        except Exception:
            # Some backends might not support move; ignore
            pass
        self._model.eval()

        # Optional torchvision NMS
        try:
            import torchvision  # type: ignore
            self._tv_nms = getattr(torchvision.ops, "nms", None)
        except Exception:
            self._tv_nms = None

    def _prep_texts(self) -> list:
        return list(self.prompts) if getattr(self, "_backend", "owlv2") == "owlv2" else [list(self.prompts)]

    def __call__(self, frame_bgr) -> Optional[BoxPrediction]:
        """Run OWL-ViT on a BGR uint8 image and return the best BoxPrediction or None."""
        try:
            import numpy as np  # type: ignore
            from PIL import Image  # type: ignore
            import cv2  # type: ignore
        except Exception:
            return None

        if frame_bgr is None:
            return None
        arr = np.asarray(frame_bgr)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return None
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        H, W = int(arr.shape[0]), int(arr.shape[1])
        if H <= 1 or W <= 1:
            return None

        try:
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        except Exception:
            rgb = arr[..., ::-1].copy()
        image = Image.fromarray(rgb)

        # Prepare inputs: OWLv2 accepts a list[str] for single image
        texts = self._prep_texts()
        device = self.device
        try:
            inputs = self._processor(text=texts, images=image, return_tensors="pt")
            inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            with self._torch.no_grad():
                outputs = self._model(**inputs)
            target_sizes = self._torch.tensor([[H, W]], device=device)
            results = self._processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=self.score_thr)
        except Exception:
            return None

        if not results:
            return None
        res = results[0]
        boxes = res.get("boxes")
        scores = res.get("scores")
        labels = res.get("labels")
        if boxes is None or scores is None or labels is None:
            return None

        # Move to CPU for processing
        try:
            boxes = boxes.detach().cpu()
            scores = scores.detach().cpu()
            labels = labels.detach().cpu()
        except Exception:
            return None

        # Filter by threshold and strict geometry gates
        keep_idx: List[int] = []
        for i in range(min(int(boxes.shape[0]), int(scores.shape[0]))):
            s = float(scores[i].item())
            if s < self.score_thr:
                continue
            x0, y0, x1, y1 = [float(v) for v in boxes[i].tolist()]
            x0 = max(0.0, min(float(W - 1), x0))
            y0 = max(0.0, min(float(H - 1), y0))
            x1 = max(0.0, min(float(W - 1), x1))
            y1 = max(0.0, min(float(H - 1), y1))
            if not _is_box_valid_xyxy(int(x0), int(y0), int(x1), int(y1), W, H):
                continue
            area_frac = ((x1 - x0) * (y1 - y0)) / float(max(1.0, W * H))
            if area_frac < self.min_box_area_frac:
                continue
            keep_idx.append(i)
        if not keep_idx:
            return None

        boxes_f = boxes[keep_idx]
        scores_f = scores[keep_idx]
        labels_f = labels[keep_idx]

        # Optional NMS across all candidates
        kept: List[int]
        if callable(getattr(self, "_tv_nms", None)):
            try:
                kept_idx = self._tv_nms(boxes_f, scores_f, float(self.nms_iou))
                kept = [int(i) for i in kept_idx.detach().cpu().tolist()]
            except Exception:
                kept = list(range(boxes_f.shape[0]))
        else:
            # Keep top-1 per label
            best_per_label = {}
            for i in range(int(boxes_f.shape[0])):
                lbl = int(labels_f[i].item())
                sc = float(scores_f[i].item())
                if lbl not in best_per_label or sc > best_per_label[lbl][0]:
                    best_per_label[lbl] = (sc, i)
            kept = [idx for (_, idx) in best_per_label.values()]

        if not kept:
            return None

        # Select single best overall
        best_idx = max(kept, key=lambda i: float(scores_f[i].item()))
        bx = boxes_f[best_idx].tolist()
        sc = float(scores_f[best_idx].item())
        lb_idx = int(labels_f[best_idx].item())
        label = self.prompts[lb_idx] if 0 <= lb_idx < len(self.prompts) else str(lb_idx)
        x0, y0, x1, y1 = float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])
        return BoxPrediction(x0=x0, y0=y0, x1=x1, y1=y1, score=sc, label=label)

    def predict_topk(self, frame_bgr, k: int = 2) -> List[BoxPrediction]:
        try:
            import numpy as np  # type: ignore
            from PIL import Image  # type: ignore
            import cv2  # type: ignore
        except Exception:
            return []
        if frame_bgr is None:
            return []
        arr = np.asarray(frame_bgr)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return []
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        H, W = int(arr.shape[0]), int(arr.shape[1])
        if H <= 1 or W <= 1:
            return []
        try:
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        except Exception:
            rgb = arr[..., ::-1].copy()
        image = Image.fromarray(rgb)
        texts = self._prep_texts()
        device = self.device
        try:
            inputs = self._processor(text=texts, images=image, return_tensors="pt")
            inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            with self._torch.no_grad():
                outputs = self._model(**inputs)
            target_sizes = self._torch.tensor([[H, W]], device=device)
            results = self._processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=self.score_thr)
        except Exception:
            return []
        if not results:
            return []
        res = results[0]
        boxes = res.get("boxes")
        scores = res.get("scores")
        labels = res.get("labels")
        if boxes is None or scores is None or labels is None:
            return []
        try:
            boxes = boxes.detach().cpu()
            scores = scores.detach().cpu()
            labels = labels.detach().cpu()
        except Exception:
            return []
        keep_idx: List[int] = []
        for i in range(min(int(boxes.shape[0]), int(scores.shape[0]))):
            s = float(scores[i].item())
            if s < self.score_thr:
                continue
            x0, y0, x1, y1 = [float(v) for v in boxes[i].tolist()]
            x0 = max(0.0, min(float(W - 1), x0))
            y0 = max(0.0, min(float(H - 1), y0))
            x1 = max(0.0, min(float(W - 1), x1))
            y1 = max(0.0, min(float(H - 1), y1))
            if not _is_box_valid_xyxy(int(x0), int(y0), int(x1), int(y1), W, H):
                continue
            area_frac = ((x1 - x0) * (y1 - y0)) / float(max(1.0, W * H))
            if area_frac < self.min_box_area_frac:
                continue
            keep_idx.append(i)
        if not keep_idx:
            return []
        boxes_f = boxes[keep_idx]
        scores_f = scores[keep_idx]
        labels_f = labels[keep_idx]
        kept: List[int]
        if callable(getattr(self, "_tv_nms", None)):
            try:
                kept_idx = self._tv_nms(boxes_f, scores_f, float(self.nms_iou))
                kept = [int(i) for i in kept_idx.detach().cpu().tolist()]
            except Exception:
                kept = list(range(boxes_f.shape[0]))
        else:
            best_per_label = {}
            for i in range(int(boxes_f.shape[0])):
                lbl = int(labels_f[i].item())
                sc = float(scores_f[i].item())
                if lbl not in best_per_label or sc > best_per_label[lbl][0]:
                    best_per_label[lbl] = (sc, i)
            kept = [idx for (_, idx) in best_per_label.values()]
        if not kept:
            return []
        # Sort by score desc and take top-k
        scored = [(float(scores_f[i].item()), i) for i in kept]
        scored.sort(key=lambda t: -t[0])
        out: List[BoxPrediction] = []
        for _, ii in scored[: max(1, int(k))]:
            bx = boxes_f[ii].tolist()
            sc = float(scores_f[ii].item())
            lb_idx = int(labels_f[ii].item())
            label = self.prompts[lb_idx] if 0 <= lb_idx < len(self.prompts) else str(lb_idx)
            x0, y0, x1, y1 = float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])
            out.append(BoxPrediction(x0=x0, y0=y0, x1=x1, y1=y1, score=sc, label=label))
        return out




