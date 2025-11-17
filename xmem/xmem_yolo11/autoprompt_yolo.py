from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class BoxPrediction:
    x0: float
    y0: float
    x1: float
    y1: float
    score: float
    label: str

    def as_int_tuple(self, width: Optional[int] = None, height: Optional[int] = None) -> Tuple[int, int, int, int]:
        x0, y0, x1, y1 = float(self.x0), float(self.y0), float(self.x1), float(self.y1)
        if width is not None and height is not None:
            x0 = max(0.0, min(float(width - 1), x0))
            y0 = max(0.0, min(float(height - 1), y0))
            x1 = max(0.0, min(float(width - 1), x1))
            y1 = max(0.0, min(float(height - 1), y1))
        return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


class YoloBoxPromptor:
    def __init__(self, model_path: str, device: str = "0", conf: float = 0.20, imgsz: int = 1280) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            raise RuntimeError("Missing dependency: ultralytics is required for YOLO autoprompt") from e

        self._YOLO = YOLO
        self.model = YOLO(model_path)
        self.device = str(device)
        self.conf = float(conf)
        self.imgsz = int(imgsz)

    def __call__(self, frame_bgr, max_det: int = 3) -> List[BoxPrediction]:
        try:
            import numpy as np  # type: ignore
        except Exception:
            return []
        if frame_bgr is None:
            return []
        arr = frame_bgr
        if not hasattr(arr, "ndim"):
            try:
                import numpy as np  # type: ignore
                arr = np.asarray(frame_bgr)
            except Exception:
                return []
        if arr.ndim != 3 or arr.shape[2] != 3:
            return []

        try:
            res = self.model.predict(
                source=arr,
                imgsz=self.imgsz,
                conf=self.conf,
                device=self.device,
                verbose=False,
                max_det=int(max(1, max_det)),
            )
        except Exception:
            return []
        if not res:
            return []
        r = res[0]
        boxes_attr = getattr(r, "boxes", None)
        if boxes_attr is None or getattr(boxes_attr, "xyxy", None) is None:
            return []

        try:
            import numpy as np  # type: ignore
            xyxy = boxes_attr.xyxy.detach().cpu().numpy()
            confs = boxes_attr.conf.detach().cpu().numpy() if getattr(boxes_attr, "conf", None) is not None else np.zeros((xyxy.shape[0],), dtype=np.float32)
            out: List[BoxPrediction] = []
            order = np.argsort(-confs)
            for i in order[: int(max(1, max_det))]:
                x0, y0, x1, y1 = [float(v) for v in xyxy[i].tolist()]
                sc = float(confs[i])
                out.append(BoxPrediction(x0=x0, y0=y0, x1=x1, y1=y1, score=sc, label="billboard"))
            return out
        except Exception:
            return []
