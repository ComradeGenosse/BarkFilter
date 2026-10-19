"""Reusable offline buffering pipeline, crossfades, and traceable processing metrics."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from time import perf_counter
import numpy as np
from .audio import to_float_audio, fit_length
from .detection import AllDetector, Interval, merge_intervals, intersects


@dataclass
class WindowTrace:
    start_seconds: float
    end_seconds: float
    separated: bool
    inference_seconds: float
    input_seconds: float


@dataclass
class PipelineResult:
    audio: np.ndarray
    spans: list[Interval]
    trace: list[WindowTrace]
    wall_seconds: float


def plan_windows(frames: int, window_frames: int, hop_frames: int) -> list[tuple[int, int]]:
    if frames < 0 or window_frames < 1 or not 0 < hop_frames <= window_frames:
        raise ValueError("Invalid window parameters")
    if frames == 0:
        return []
    # For final window, slide it backward to avoid tiny tail calls.
    starts = list(range(0, max(frames - window_frames, 0) + 1, hop_frames))
    last_start = max(0, frames - window_frames)
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return [(s, min(frames, s + window_frames)) for s in sorted(set(starts))]


def process_audio(audio: np.ndarray, sample_rate: int, separator, *, detector=None,
                  window_seconds: float = 8.0, overlap_seconds: float = 2.0,
                  mode: str = "hybrid", event_margin_seconds: float = .5) -> PipelineResult:
    x = to_float_audio(audio)
    if sample_rate < 1 or window_seconds <= 0 or not 0 <= overlap_seconds < window_seconds:
        raise ValueError("Invalid sample rate/window/overlap")
    if mode not in ("continuous", "hybrid"):
        raise ValueError("mode must be continuous or hybrid")
    started = perf_counter()
    if len(x) == 0:
        return PipelineResult(x.copy(), [], [], perf_counter() - started)
    det = AllDetector() if mode == "continuous" else detector
    if det is None:
        raise ValueError("Hybrid mode requires explicit detector. No detector = no safe bark gating.")
    spans = merge_intervals(det.detect(x, sample_rate), duration=len(x)/sample_rate,
                            margin=event_margin_seconds if mode == "hybrid" else 0)
    if mode == "hybrid" and not spans:
        return PipelineResult(x.copy(), [], [], perf_counter()-started)
    window_frames = max(1, round(window_seconds * sample_rate))
    hop_frames = max(1, round((window_seconds-overlap_seconds)*sample_rate))
    windows = plan_windows(len(x), window_frames, hop_frames)
    # All windows are blended, including ones not separated. This ensures
    # identical pass-through when no window is transformed.
    accum = np.zeros_like(x, dtype=np.float64)
    weights = np.zeros((len(x), 1), dtype=np.float64)
    trace: list[WindowTrace] = []
    for i, (a, b) in enumerate(windows):
        should_separate = mode == "continuous" or intersects(a/sample_rate, b/sample_rate, spans)
        tick = perf_counter()
        if should_separate:
            processed = to_float_audio(separator.separate(x[a:b].copy(), sample_rate))
            if processed.shape[1] != x.shape[1]:
                raise ValueError("Separator changed the number of channels")
            # Reject substantial length changes instead of hiding backend errors.
            if abs(len(processed) - (b-a)) > max(2, round(sample_rate * 0.05)):
                raise ValueError("Separator duration drift exceeds 50 ms")
            processed = fit_length(processed, b-a)
        else:
            processed = x[a:b]
        seconds = perf_counter()-tick
        size = b-a
        w = np.ones(size, dtype=np.float64)
        if i > 0:
            prev_end = windows[i-1][1]
            fade = min(max(0, prev_end-a), size)
            if fade:
                # cosine overlap-add; the summed normalization handles arbitrary overlaps
                ramp = .5 - .5 * np.cos(np.linspace(0, np.pi, fade, endpoint=True))
                w[:fade] *= ramp
        if i < len(windows)-1:
            next_start = windows[i+1][0]
            fade = min(max(0, b-next_start), size)
            if fade:
                ramp = .5 + .5 * np.cos(np.linspace(0, np.pi, fade, endpoint=True))
                w[-fade:] *= ramp
        # Floor weights prevents degenerate zero at multi-overlap boundaries.
        w = np.maximum(w, 1.e-7)
        accum[a:b] += processed * w[:, None]
        weights[a:b] += w[:, None]
        trace.append(WindowTrace(a/sample_rate, b/sample_rate, should_separate,
                                 seconds if should_separate else 0.0, size/sample_rate))
    y = (accum / np.maximum(weights, 1e-12)).astype(np.float32)
    if not np.isfinite(y).all():
        raise RuntimeError("Separation produced nonfinite samples")
    return PipelineResult(y, spans, trace, perf_counter()-started)
