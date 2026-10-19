"""Pluggable dog-bark detection. No AI model is bundled by this project."""
from __future__ import annotations
from dataclasses import dataclass
import csv
import json
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
from .audio import to_float_audio


@dataclass(frozen=True)
class Interval:
    start: float
    end: float


def merge_intervals(spans: list[Interval], *, duration: float, margin: float = 0.0) -> list[Interval]:
    if duration < 0 or margin < 0:
        raise ValueError("duration and margin must be nonnegative")
    expanded = []
    for x in spans:
        if not (np.isfinite(x.start) and np.isfinite(x.end)) or x.start > x.end:
            raise ValueError("Invalid interval")
        a, b = max(0.0, x.start - margin), min(duration, x.end + margin)
        if a < b:
            expanded.append(Interval(a, b))
    expanded.sort(key=lambda x: x.start)
    joined: list[Interval] = []
    for x in expanded:
        if joined and x.start <= joined[-1].end + 1e-9:
            joined[-1] = Interval(joined[-1].start, max(x.end, joined[-1].end))
        else:
            joined.append(x)
    return joined


def intersects(start: float, end: float, spans: list[Interval]) -> bool:
    return any(s.start < end and s.end > start for s in spans)


class AllDetector:
    def detect(self, audio: np.ndarray, sample_rate: int) -> list[Interval]:
        return [Interval(0, len(audio) / sample_rate)] if len(audio) else []


class ManualDetector:
    """Ground-truth/hand-labeled event spans; NOT automatic bark detection."""
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def detect(self, audio: np.ndarray, sample_rate: int) -> list[Interval]:
        record = json.loads(self.path.read_text(encoding="utf-8"))
        spans = [Interval(float(a), float(b)) for a, b in record["barks"]]
        return merge_intervals(spans, duration=len(audio) / sample_rate)


class YAMNetDetector:
    """Optional CPU TensorFlow Hub YAMNet; requires locally cached/extracted TF Hub model.

    Uses class `Bark`, not general class `Dog` to limit false positives. YAMNet is
    an event classifier and cannot produce an audio residual.
    """
    def __init__(self, model_dir: str, threshold: float = 0.14):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be within [0, 1]")
        try:
            import tensorflow_hub as hub
        except ImportError as exc:
            raise RuntimeError("Install optional detector: pip install '.[detector]'") from exc
        self.model = hub.load(model_dir)
        classmap = self.model.class_map_path().numpy().decode("utf-8")
        with open(classmap, newline="", encoding="utf-8") as f:
            records = list(csv.DictReader(f))
        match = [i for i, row in enumerate(records) if row.get("display_name", "").casefold() == "bark"]
        if len(match) != 1:
            raise RuntimeError("YAMNet class map missing unambiguous 'Bark' class")
        self.bark_class = match[0]
        self.threshold = threshold

    def detect(self, audio: np.ndarray, sample_rate: int) -> list[Interval]:
        import tensorflow as tf
        audio = to_float_audio(audio)
        mono = audio.mean(axis=1)
        from math import gcd
        g = gcd(sample_rate, 16000)
        waveform = resample_poly(mono, 16000 // g, sample_rate // g).astype(np.float32)
        if len(waveform) < int(0.975 * 16000):
            waveform = np.pad(waveform, (0, int(0.975 * 16000) - len(waveform)))
        scores, _, _ = self.model(tf.convert_to_tensor(waveform))
        values = scores.numpy()[:, self.bark_class]
        # YAMNet window ~0.96 s, hop ~0.48 s. These are approximate spans.
        spans = [Interval(i * .48, i * .48 + .96)
                 for i, value in enumerate(values) if float(value) >= self.threshold]
        return merge_intervals(spans, duration=len(audio) / sample_rate)


class YAMNetOnnxDetector:
    """CPU ONNX YAMNet (~16 MB), easier to deploy on Windows than TensorFlow.

    Source converter: https://huggingface.co/audiomagic/yamnet-onnx
    This does bark event CLASSIFICATION only, not sound separation.
    """
    def __init__(self, onnx_model: str, class_map: str, threshold: float = .14):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be within [0,1]")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Install optional ONNX detector: pip install '.[onnx-detector]'") from exc
        self.session = ort.InferenceSession(str(onnx_model), providers=["CPUExecutionProvider"])
        with open(class_map, newline="", encoding="utf-8") as f:
            records = list(csv.DictReader(f))
        match = [i for i, row in enumerate(records) if row.get("display_name", "").casefold() == "bark"]
        if len(match) != 1:
            raise RuntimeError("YAMNet ONNX class map missing unambiguous 'Bark' class")
        self.bark_class = match[0]
        self.threshold = threshold
        self.input_key = self.session.get_inputs()[0].name

    def detect(self, audio: np.ndarray, sample_rate: int) -> list[Interval]:
        from math import gcd
        mono = to_float_audio(audio).mean(axis=1)
        g = gcd(sample_rate, 16000)
        waveform = resample_poly(mono, 16000//g, sample_rate//g).astype(np.float32)
        if len(waveform) < 16000:
            waveform = np.pad(waveform, (0, 16000-len(waveform)))
        scores = self.session.run(None, {self.input_key:waveform})[0]
        matches = [Interval(i*.48, i*.48+.96)
                   for i, score in enumerate(scores[:,self.bark_class])
                   if float(score)>=self.threshold]
        return merge_intervals(matches, duration=len(audio)/sample_rate)
