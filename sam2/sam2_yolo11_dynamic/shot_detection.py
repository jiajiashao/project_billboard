"""
Shot detection helper.

Optional dependency: install PySceneDetect via:
    pip install pyscenedetect

This module will automatically fall back to an OpenCV-based detector if
PySceneDetect is not available. No files are written and no output is printed.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional


@dataclass
class ShotSegment:
    start: int  # inclusive frame index
    end: int  # exclusive frame index


def _probe_fps_and_frames(path_str: str) -> tuple[Optional[float], Optional[int]]:
    """Best-effort probe of (fps, total_frames) using OpenCV.

    Returns (fps, total_frames), each may be None if probing fails.
    """
    try:
        import cv2  # type: ignore
    except Exception:
        return None, None

    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        cap.release()
        return None, None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    fps_out = float(fps) if fps and fps > 0 else None
    total_out = int(total) if total and total > 0 else None
    return fps_out, total_out


def _opencv_fallback_boundaries(
    path_str: str,
    max_merge_gap_frames: int,
) -> tuple[List[int], int]:
    """Detect shot boundaries using simple frame differencing.

    Returns (boundaries, total_frames). Boundaries exclude 0 and last frame.
    Uses only cv2 and numpy. Deterministic.
    """
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover - environment dependent
        raise RuntimeError("OpenCV fallback unavailable (cv2/numpy import failed)") from e

    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Failed to open video with OpenCV")

    boundaries: List[int] = []
    ret, prev = cap.read()
    frame_idx = 0
    if not ret:
        cap.release()
        raise RuntimeError("Could not read first frame")

    # Preprocess: small grayscale to stabilize noise and reduce cost.
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.resize(prev_gray, (160, max(1, int(prev_gray.shape[0] * 160 / max(prev_gray.shape[1], 1)))), interpolation=cv2.INTER_AREA)

    diffs: List[float] = []
    last_kept_boundary = -10**9

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (prev_gray.shape[1], prev_gray.shape[0]), interpolation=cv2.INTER_AREA)
        diff = cv2.absdiff(gray, prev_gray)
        score = float(diff.mean())
        diffs.append(score)

        prev_gray = gray

        # We will choose thresholds after we have statistics; here we only collect.

    total_frames = frame_idx + 1  # include the very first frame at index 0
    cap.release()

    if not diffs:
        # Single-frame video: no boundaries
        return [], total_frames

    # Robust, deterministic threshold: median + 3 * MAD (approx robust std).
    arr = np.asarray(diffs, dtype=np.float32)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    robust_std = 1.4826 * mad
    thresh = med + 3.0 * robust_std

    # As a safeguard, ensure threshold is at least a small epsilon above median.
    import numpy as _np  # ensure np.isfinite is available if np alias changed above
    if not _np.isfinite(thresh) or thresh <= med:
        thresh = med + 1.0

    for i, s in enumerate(arr, start=1):  # boundary after frame i-1 before frame i
        if s > thresh:
            # prevent emitting boundaries clustered within max_merge_gap_frames inline
            if i - last_kept_boundary > max_merge_gap_frames:
                boundaries.append(i)
                last_kept_boundary = i

    # Boundaries must be strictly within (0, total_frames)
    boundaries = [b for b in boundaries if 0 < b < total_frames]

    return boundaries, total_frames


def _merge_close_boundaries(boundaries: List[int], max_gap: int) -> List[int]:
    if not boundaries:
        return []
    sorted_b = sorted(set(boundaries))
    merged: List[int] = []
    last = None
    for b in sorted_b:
        if last is None:
            merged.append(b)
            last = b
        else:
            if b - last <= max_gap:
                # De-jitter: keep the earlier boundary by default
                # (deterministic and avoids creating micro-shots).
                continue
            merged.append(b)
            last = b
    return merged


def _segments_from_boundaries(boundaries: List[int], total_end: int) -> List[ShotSegment]:
    # boundaries expected sorted, unique, and within (0, total_end)
    cuts = [0] + boundaries + [total_end]
    out: List[ShotSegment] = []
    for i in range(len(cuts) - 1):
        s, e = cuts[i], cuts[i + 1]
        if e > s:
            out.append(ShotSegment(start=int(s), end=int(e)))
    return out


def _merge_micro_shots(segments: List[ShotSegment], min_len_frames: int) -> List[ShotSegment]:
    if not segments or min_len_frames <= 0:
        return segments
    out: List[ShotSegment] = []
    for seg in segments:
        if not out:
            out.append(seg)
            continue
        if seg.end - seg.start < min_len_frames:
            # Merge with previous by extending previous end to this end.
            prev = out[-1]
            out[-1] = ShotSegment(start=prev.start, end=seg.end)
        else:
            out.append(seg)
    return out


def detect_shots(
    video_path: Path | str,
    total_frames: Optional[int] = None,
    fps: Optional[float] = None,
    method: Literal["adaptive", "content"] = "adaptive",
    min_shot_len_s: float = 0.5,
    content_threshold: float = 27.0,
    adaptive_sensitivity: int = 3,
    max_merge_gap_frames: int = 2,
) -> List[ShotSegment]:
    """Return a sorted, non-overlapping list of [start,end) frame segments.

    Must be deterministic. If detection fails, raises a clear Exception with a short message.
    """

    # Basic validation
    if method not in ("adaptive", "content"):
        raise ValueError("method must be 'adaptive' or 'content'")
    if min_shot_len_s < 0:
        raise ValueError("min_shot_len_s must be >= 0")
    if max_merge_gap_frames < 0:
        raise ValueError("max_merge_gap_frames must be >= 0")

    path_str = str(Path(video_path))

    # Determine fps to use for time->frame conversion (when needed for merging logic).
    fps_to_use: Optional[float] = fps if (fps is not None and fps > 0) else None
    probed_fps, probed_frames = _probe_fps_and_frames(path_str)
    if fps_to_use is None:
        fps_to_use = probed_fps if (probed_fps is not None and probed_fps > 0) else 30.0

    # Use PySceneDetect if available, otherwise fallback to OpenCV differencing.
    boundaries: List[int] = []
    final_total_frames: Optional[int] = total_frames if (total_frames is not None and total_frames > 0) else None

    used_pyscenedetect = False
    try:
        # Try new API first (scenedetect >= 0.6)
        from scenedetect import SceneManager  # type: ignore
        try:
            from scenedetect import open_video  # type: ignore
            has_open_video = True
        except Exception:
            has_open_video = False
        try:
            from scenedetect.detectors import ContentDetector, AdaptiveDetector  # type: ignore
        except Exception:
            from scenedetect.detectors import ContentDetector  # type: ignore
            AdaptiveDetector = None  # type: ignore

        used_pyscenedetect = True

        if has_open_video:
            video = open_video(path_str)
            sm = SceneManager()
            if method == "adaptive" and 'AdaptiveDetector' in locals() and locals()['AdaptiveDetector'] is not None:
                sm.add_detector(AdaptiveDetector(sensitivity=int(adaptive_sensitivity)))  # type: ignore
            else:
                sm.add_detector(ContentDetector(threshold=float(content_threshold)))
            sm.detect_scenes(video, show_progress=False)
            scene_list = sm.get_scene_list()

            # Derive boundaries from scene start times (exclude first 0 start).
            starts = [t[0].get_frames() for t in scene_list]
            if starts and starts[0] == 0:
                starts = starts[1:]
            boundaries = list(sorted(set(int(s) for s in starts if s > 0)))

            # Determine total frames from last scene end if not provided.
            if final_total_frames is None:
                if scene_list:
                    last_end_inc = scene_list[-1][1].get_frames()
                    final_total_frames = int(last_end_inc + 1)
                else:
                    final_total_frames = probed_frames if probed_frames is not None else None
        else:
            # Older API pathway
            from scenedetect import VideoManager, StatsManager  # type: ignore

            video_manager = VideoManager([path_str])
            stats_manager = StatsManager()
            sm = SceneManager(stats_manager)
            # AdaptiveDetector may not exist; try and fall back.
            try:
                from scenedetect.detectors import AdaptiveDetector as _AD  # type: ignore
                AdaptiveDetector = _AD  # type: ignore
            except Exception:
                from scenedetect.detectors import ContentDetector as _CD  # type: ignore
                AdaptiveDetector = None  # type: ignore
                ContentDetector = _CD  # type: ignore

            if method == "adaptive" and AdaptiveDetector is not None:
                sm.add_detector(AdaptiveDetector(sensitivity=int(adaptive_sensitivity)))
            else:
                sm.add_detector(ContentDetector(threshold=float(content_threshold)))

            video_manager.start()
            try:
                sm.detect_scenes(frame_source=video_manager)
                scene_list = sm.get_scene_list(video_manager)
            finally:
                video_manager.release()

            starts = [t[0].get_frames() for t in scene_list]
            if starts and starts[0] == 0:
                starts = starts[1:]
            boundaries = list(sorted(set(int(s) for s in starts if s > 0)))
            if final_total_frames is None:
                if scene_list:
                    last_end_inc = scene_list[-1][1].get_frames()
                    final_total_frames = int(last_end_inc + 1)
                else:
                    final_total_frames = probed_frames if probed_frames is not None else None

    except Exception:
        used_pyscenedetect = False
        # Fall back to OpenCV-based detector.
        boundaries, detected_total = _opencv_fallback_boundaries(path_str, max_merge_gap_frames=max_merge_gap_frames)
        if final_total_frames is None:
            final_total_frames = detected_total

    # Ensure we have a total frame count to bound the output if known/provided.
    if final_total_frames is None:
        # As an ultimate fallback, try to use probed_frames, else raise.
        if probed_frames is not None and probed_frames > 0:
            final_total_frames = int(probed_frames)
        else:
            # We can still return segments based on boundaries, but to guarantee coverage we need length.
            # In absence of total length, if no boundaries were detected, we cannot produce segments.
            if not boundaries:
                raise RuntimeError("Shot detection failed to determine total frame count")
            # If we have boundaries but not total length, we cannot guarantee coverage; use last boundary + 1.
            final_total_frames = int(max(boundaries) + 1)

    # De-jitter boundaries and clamp to range.
    boundaries = [b for b in boundaries if 0 < b < int(final_total_frames)]
    boundaries = _merge_close_boundaries(boundaries, max_merge_gap_frames)

    # Build segments.
    segments = _segments_from_boundaries(boundaries, int(final_total_frames))

    # Merge micro-shots shorter than threshold.
    min_len_frames = int(round(max(0.0, min_shot_len_s) * float(fps_to_use))) if fps_to_use else 0
    if min_len_frames > 0:
        segments = _merge_micro_shots(segments, min_len_frames)

    # Guarantee coverage when total_frames is known: [0, total_frames)
    if total_frames is not None and total_frames > 0:
        if not segments:
            segments = [ShotSegment(0, int(total_frames))]
        else:
            # Adjust first and last
            first = segments[0]
            if first.start != 0:
                segments[0] = ShotSegment(0, first.end)
            last = segments[-1]
            if last.end != int(total_frames):
                segments[-1] = ShotSegment(last.start, int(total_frames))

    # Ensure sorted, non-overlapping, and within bounds.
    cleaned: List[ShotSegment] = []
    for seg in segments:
        s = max(0, int(seg.start))
        e = max(s, int(seg.end))
        if total_frames is not None:
            e = min(e, int(total_frames))
        if cleaned and s < cleaned[-1].end:
            # enforce non-overlap by snapping start
            s = cleaned[-1].end
        if e > s:
            cleaned.append(ShotSegment(s, e))

    # Final determinism: collapse any accidental duplicates and sort.
    cleaned = sorted({(seg.start, seg.end) for seg in cleaned})
    cleaned = [ShotSegment(s, e) for s, e in cleaned]

    if not cleaned:
        # If everything collapsed, still guarantee at least one segment when total known.
        if total_frames is not None and total_frames > 0:
            cleaned = [ShotSegment(0, int(total_frames))]
        else:
            raise RuntimeError("Shot detection produced no segments")

    return cleaned


"""
# Example (not executed here):
from shot_detection import detect_shots, ShotSegment
from pathlib import Path

shots = detect_shots(
    Path("data/clips/clip_corner.mp4"),
    total_frames=1234,        # optional
    fps=25.0,                 # optional
    method="adaptive",
    min_shot_len_s=0.5,
    adaptive_sensitivity=3
)
# shots -> [ShotSegment(start=0,end=132), ShotSegment(132,301), ...]

# In sam2_pilot_fix2_cuda.py, at the start of each clip processing:
# from shot_detection import detect_shots
# shots = detect_shots(clip_path, total_frames=len(frames), fps=fps, method="adaptive", min_shot_len_s=0.5)
# for shot in shots:
#     # later I'll reseed at shot.start using my existing GT/auto-prompt logic
#     pass
"""

