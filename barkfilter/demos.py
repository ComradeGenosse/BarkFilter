"""Create openly redistributable *synthetic* test fixtures.

WARNING: The clean references are produced by a known signal mixer, NOT by AI.
The ducked outputs are a deliberately poor non-AI comparison baseline.
These are pipeline test fixtures, NOT evidence of dog-bark removal quality.
"""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import soundfile as sf
from .audio import resample, write_audio

RATE = 24000
DURATION = 6.0
N = int(RATE * DURATION)


def speech() -> np.ndarray:
    if shutil.which("espeak"):
        p = subprocess.run(["espeak", "--stdout", "-s", "150", "-a", "120",
                            "The little dog ran across the street. Did you hear that?"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode == 0 and p.stdout[:4] == b"RIFF":
            try:
                x, sr = sf.read(io.BytesIO(p.stdout), dtype="float32")
                x = resample(np.asarray(x).reshape(-1, 1), sr, RATE)[:, 0]
                out = np.zeros(N)
                length = min(N-len(out)//8, len(x))
                out[N//8:N//8+length] = x[:length]
                return .65*out
            except Exception:
                pass
    # Procedural voiced formant-like proxy when eSpeak isn't installed.
    t = np.arange(N) / RATE
    envelope = .5*(1 + np.sin(2*np.pi*3.2*t)) ** 2
    return .065 * envelope * (np.sin(2*np.pi*145*t)+.25*np.sin(2*np.pi*290*t))


def music() -> np.ndarray:
    t = np.arange(N) / RATE
    notes = [220., 261.626, 329.628, 392., 329.628, 261.626, 246.942, 220.]
    out = np.zeros(N)
    for n in range(int(DURATION * 4)):
        a = round(n * .25 * RATE)
        b = min(N, a + round(.38 * RATE))
        if b <= a: continue
        local = np.arange(b-a) / RATE
        f = notes[n%len(notes)]
        env = np.exp(-5*local)
        out[a:b] += .13*env*(np.sin(2*np.pi*f*local)+.25*np.sin(4*np.pi*f*local))
    out += .04*np.sin(2*np.pi*110*t)
    return out


def sfx(seed=44) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros(N)
    for centre, level in [(1.1,.12),(2.4,.2),(4.1,.17)]:
        a = round(centre*RATE)
        z = min(N-a, round(.4*RATE))
        out[a:a+z] += level*rng.standard_normal(z)*np.exp(-14*np.arange(z)/RATE)
    return out


def birds() -> np.ndarray:
    t = np.arange(N)/RATE
    chirp = np.sin(2*np.pi*(1800*t + 700*np.sin(2*np.pi*8*t)/(2*np.pi*8)))
    gate = ((t%1.2) < .18).astype(float)
    return .065*chirp*gate


def bark_proxy(onsets, *, pitch=180, level=.3, seed=19, duration=.21) -> tuple[np.ndarray, list[list[float]]]:
    rng = np.random.default_rng(seed)
    out = np.zeros(N)
    spans = []
    for onset in onsets:
        a = round(onset*RATE)
        count = min(round(duration*RATE), N-a)
        if count <= 0: continue
        t = np.arange(count)/RATE
        # Construct noisy pitched impulse, not an actual recorded dog.
        sweep = pitch*(1.35-.45*np.minimum(t/.2, 1))
        phase = 2*np.pi*np.cumsum(sweep)/RATE
        noise = rng.standard_normal(count)
        env = np.minimum(1, t/.008)*np.exp(-11*t)
        out[a:a+count] += level*env*(.65*np.sin(phase)+.35*noise)
        spans.append([round(onset, 3), round(onset+count/RATE, 3)])
    return out, spans


def case_config():
    return {
        "speech_overlap": (speech(), [1.7, 2.45, 3.55], .24, .24),
        "music_overlap": (music(), [1.4, 3.1, 4.45], .22, .26),
        "action_sfx": (.8*music()+sfx(), [1.95, 3.42], .21, .24),
        "other_animals": (birds()+.6*speech(), [1.5, 3.3], .23, .23),
        "multiple_dogs": (.5*speech()+music(), [1.4, 2.1, 2.2, 3.0], .22, .21),
        "brief_bark": (music(), [2.0], .34, .085),
        "quiet_bark": (.8*speech()+.6*music(), [2.1, 4.0], .065, .22),
        "overlapping_words_HARD": (speech(), [1.2, 1.8, 2.25, 2.6, 3.1], .38, .25),
        "NO_BARK": (speech()+music(), [], 0, .20),
        "false_positive_percussion": (sfx()+music(), [], 0, .20),
    }


def generate(outdir: str|Path):
    directory = Path(outdir)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"description":"SYNTHETIC source stems; not canine recordings; NOT AI outputs",
                "rate": RATE, "duration_sec": DURATION,
                "speech": "espeak synthetic TTS if available, else voiced-tone fallback",
                "music": "original procedural notes", "bark": "original procedural bark-LIKE waveform",
                "baseline": "ground-truth-span waveform ducking; reduces EVERYTHING during spans",
                "examples":{}}
    for name, (clean, positions, gain, length) in case_config().items():
        bark, spans = bark_proxy(positions, level=gain, duration=length,
                                 pitch=170 if name != "multiple_dogs" else 130,
                                 seed=3)
        clean = np.asarray(clean, dtype=np.float32)
        mixed = clean + bark.astype(np.float32)
        mix_peak = max(np.max(np.abs(mixed)), .001)
        if mix_peak > .97:
            clean *= .97/mix_peak
            mixed *= .97/mix_peak
        # Demonstrate basic ducking failure even with perfect event boundaries.
        ducked = mixed.copy()
        for a,b in spans:
            start,end = round(a*RATE),round(b*RATE)
            fade = min(round(.012*RATE), max(1,(end-start)//4))
            window = np.full(end-start, .12)
            if fade:
                down = np.linspace(1, .12, fade, endpoint=True)
                up = np.linspace(.12, 1, fade, endpoint=True)
                window[:fade] = down
                window[-fade:] = up
            ducked[start:end] *= window
        files = {}
        for kind, arr in (("input",mixed),("baseline_ducked",ducked),("oracle_clean_NOT_MODEL",clean)):
            filename = f"{name}_{kind}.wav"
            write_audio(directory/filename, arr[:,None], RATE)
            files[kind] = filename
        json_path = directory/f"{name}_spans.json"
        json_path.write_text(json.dumps({"barks":spans},indent=2),encoding="utf-8")
        manifest["examples"][name] = {"files":files, "spans":json_path.name,
                                      "simulated_events":len(spans)}
    (directory/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (directory/"ABOUT_DEMOS.txt").write_text(
        "IMPORTANT: These are synthetic fixtures. 'baseline_ducked' simply turns down EVERY sound "
        "during a known simulated-bark interval (including dialogue/music). The 'oracle_clean' reference "
        "is from the original unmixed signal, NOT an AI result. None of these WAVs demonstrate AI bark removal.\n"
        "Run barkfilter with a downloaded SAM model to produce genuine AI outputs, then evaluate those outputs.\n",
        encoding="utf-8")
    print(f"Generated {len(manifest['examples'])} synthetic fixture sets in {directory}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir",default="demos")
    args = ap.parse_args(argv)
    generate(args.output_dir)


if __name__ == "__main__":
    main()
