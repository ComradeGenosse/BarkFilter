"""Audio I/O. FFmpeg is needed for video and compressed input; WAV uses soundfile."""
from __future__ import annotations
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

DEFAULT_SR = 48000


def to_float_audio(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2 or a.shape[1] not in (1, 2):
        raise ValueError("Expected mono or stereo audio (frames, channels)")
    if not np.isfinite(a).all():
        raise ValueError("Input audio contains NaN/Infinity")
    return a


def resample(x: np.ndarray, source_sr: int, dest_sr: int) -> np.ndarray:
    x = to_float_audio(x)
    if source_sr <= 0 or dest_sr <= 0:
        raise ValueError("Sample rates must be positive")
    if source_sr == dest_sr:
        return x.copy()
    common = gcd(source_sr, dest_sr)
    y = resample_poly(x, dest_sr // common, source_sr // common, axis=0).astype(np.float32)
    exact_len = round(len(x) * dest_sr / source_sr)
    return fit_length(y, exact_len)


def fit_length(x: np.ndarray, frames: int) -> np.ndarray:
    x = to_float_audio(x)
    if frames < 0:
        raise ValueError("Negative frame count")
    if len(x) >= frames:
        return x[:frames].copy()
    return np.pad(x, ((0, frames - len(x)), (0, 0)), mode="constant")


def read_audio(path: str | Path, *, sample_rate: int = DEFAULT_SR, channels: int = 0,
               ffmpeg: str = "ffmpeg") -> tuple[np.ndarray, int]:
    """Decode file/video. channels=0 preserves 1/2 channels, >0 forces that channel count."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in (".wav", ".flac"):
        data, source_sr = sf.read(path, dtype="float32", always_2d=True)
        if data.shape[1] <= 2:
            data = to_float_audio(data)
            if channels == 1 and data.shape[1] == 2:
                data = data.mean(axis=1, keepdims=True)
            elif channels == 2 and data.shape[1] == 1:
                data = np.repeat(data, 2, axis=1)
            return resample(data, source_sr, sample_rate), sample_rate
        # For surround-channel WAV/FLAC, use FFmpeg's channel-layout-aware stereo downmix.
    if not shutil.which(ffmpeg) and not Path(ffmpeg).is_file():
        raise RuntimeError("FFmpeg not found; use WAV/FLAC or install FFmpeg")
    with tempfile.TemporaryDirectory(prefix="barkfilter-decode-") as tmp:
        wav = Path(tmp) / "decoded.wav"
        cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
               "-map", "0:a:0", "-vn", "-ar", str(sample_rate), "-c:a", "pcm_f32le"]
        # Avoid unsupported >2 channel material; 5.1 and 7.1 are downmixed.
        cmd += ["-ac", str(channels or 2)]
        cmd += [str(wav)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"FFmpeg decode failed: {result.stderr[-800:]}")
        data, sr = sf.read(wav, dtype="float32", always_2d=True)
        return to_float_audio(data), sr


def write_audio(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), to_float_audio(audio), sample_rate, subtype="PCM_24")
