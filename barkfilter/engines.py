"""Separation backends. The GGUF adapter shells out; official SAM stays loaded."""
from __future__ import annotations
from pathlib import Path
import subprocess
import tempfile
import shutil
import time
import numpy as np
import soundfile as sf
from .audio import to_float_audio, fit_length, resample, write_audio


class IdentitySeparator:
    """Tests/diagnostics only. Never claims to remove barking."""
    name = "identity-NO-BARK-REMOVAL"

    def separate(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        return to_float_audio(audio).copy()


class SamGGUFSeparator:
    """SAM Audio via audio.cpp CLI. Each call starts a NEW MODEL PROCESS.

    This deliberately makes cold-start costs visible. It is NOT a persistent
    session / streaming adapter. Use official SAM adapter for warm-call tests.
    """
    name = "sam-audio-gguf"

    def __init__(self, cli: str, model: str, *, backend: str = "cuda",
                 prompt: str = "dog barking", timeout: int = 900):
        self.cli = str(cli)
        self.model = str(model)
        self.backend = backend
        self.prompt = prompt
        self.timeout = timeout
        if not Path(self.cli).is_file() and not shutil.which(self.cli):
            raise FileNotFoundError(f"audio.cpp binary not found: {self.cli}")
        if not Path(self.model).is_file():
            raise FileNotFoundError(f"GGUF model not found: {self.model}")

    def separate(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        x = to_float_audio(audio)
        channels = []
        # The GGUF implementation returns MONO 48k WAV. To retain stereo,
        # run each channel independently (and pay the extra startup cost).
        for ch in range(x.shape[1]):
            with tempfile.TemporaryDirectory(prefix="barkfilter-samgguf-") as tmp:
                input_path = Path(tmp) / "input.wav"
                output_dir = Path(tmp) / "outputs"
                write_audio(input_path, x[:, ch:ch+1], sample_rate)
                cmd = [self.cli, "--task", "s2s", "--family", "sam_audio",
                       "--model", self.model, "--backend", self.backend,
                       "--audio", str(input_path), "--text", self.prompt,
                       "--seed", "-1", "--out-dir", str(output_dir)]
                try:
                    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, timeout=self.timeout)
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(f"audio.cpp timed out after {self.timeout}s") from exc
                if p.returncode != 0:
                    raise RuntimeError(f"audio.cpp failed (exit {p.returncode}): {p.stderr[-1800:]}")
                residual = output_dir / "residual.wav"
                if not residual.is_file():
                    raise RuntimeError(f"SAM output missing residual.wav; stdout: {p.stdout[-1200:]}")
                y, sr = sf.read(str(residual), dtype="float32", always_2d=True)
                if y.shape[1] != 1:
                    raise RuntimeError("Expected mono SAM GGUF output")
                channels.append(fit_length(resample(y, sr, sample_rate), len(x))[:, 0])
        return np.stack(channels, axis=1)


class SamOfficialSeparator:
    """Official SAM Audio Python interface with persistent GPU model instance.

    Dependencies and accepted/gated checkpoint must be installed separately.
    """
    name = "sam-audio-official"

    def __init__(self, model: str = "facebook/sam-audio-small", *, prompt: str = "dog barking",
                 device: str = "cuda"):
        try:
            from sam_audio import SAMAudio, SAMAudioProcessor
            import torch
        except ImportError as exc:
            raise RuntimeError("Install upstream facebookresearch/sam-audio first") from exc
        self.torch = torch
        self.prompt = prompt
        self.device = device
        self.processor = SAMAudioProcessor.from_pretrained(model)
        self.model = SAMAudio.from_pretrained(model).eval().to(device)

    def separate(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        x = to_float_audio(audio)
        channels = []
        for ch in range(x.shape[1]):
            with tempfile.TemporaryDirectory(prefix="barkfilter-sam-official-") as tmp:
                wav = Path(tmp) / "audio.wav"
                write_audio(wav, x[:, ch:ch + 1], sample_rate)
                batch = self.processor(audios=[str(wav)], descriptions=[self.prompt]).to(self.device)
                with self.torch.inference_mode():
                    result = self.model.separate(batch, predict_spans=False, reranking_candidates=1)
                y = result.residual.detach().cpu().float().numpy().squeeze()
                if y.ndim != 1:
                    raise RuntimeError(f"Unexpected official SAM residual shape: {y.shape}")
                sr = self.processor.audio_sampling_rate
                channels.append(fit_length(resample(y, sr, sample_rate), len(x))[:, 0])
        return np.stack(channels, axis=1)
