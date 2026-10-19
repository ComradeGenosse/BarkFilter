# Project BarkFilter — Phase 0A

**Status: Offline pipeline / model adapters implemented and unit-tested. The target
RTX 5090, pretrained SAM weights, real-dog evaluation corpus and inference quality
are not available in the build environment. This is NOT a validated AI bark remover.**

The initial deliverable accepts an audio file or FFmpeg-readable video, routes it
through either Meta's SAM Audio official Python model or the `audio.cpp` native
GGUF model and writes the **residual** (“everything except the text-prompted
`dog barking` target”). Audio without detected barks can bypass separation.
A lightweight optional YAMNet event classifier is used only to *detect* barking;
YAMNet does NOT remove anything.

### Quick start (Windows 11)

In PowerShell, from this project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[test]"
python -m pytest -q
python -m barkfilter.demos --output-dir demos
```

Install FFmpeg separately and add its `bin` folder to your `PATH` for MP4,
MKV, MP3 and other compressed media. WAV and FLAC use the soundfile library.
The project requires Python 3.11+ and does not itself download any checkpoint.

### Recommended first AI experiment: SAM Audio Small GGUF (local CUDA)

1. Download a compatible native Windows CUDA `audiocpp_cli.exe` from
   [audio.cpp releases](https://github.com/0xShug0/audio.cpp/releases).
   Follow the [Windows CUDA runtime DLL packaging instructions](https://github.com/0xShug0/audio.cpp/blob/main/docs/build/windows.md).
   **Important:** the CUDA package may need the separate CUDA runtime archive
   extracted alongside the binary. Use a current build with `sam_audio` support.
   Alternately, [build with MSVC and CUDA](https://github.com/0xShug0/audio.cpp#build-yourself):
   `.\scripts\build_windows.ps1 -Preset windows-cuda-release -Target audiocpp_cli`.
2. Download *one exact GGUF file* from
   [audio-cpp/SAM-Audio-GGUF](https://huggingface.co/audio-cpp/SAM-Audio-GGUF),
   initially **`sam-audio-small-q8_0.gguf`** (~2.38 GB on disk). You may use
   `huggingface-cli download audio-cpp/SAM-Audio-GGUF sam-audio-small-q8_0.gguf
   --local-dir .\models` after installing `huggingface_hub` and satisfying any
   upstream terms/access requirements; manually downloading the exact file is
   also fine. The GGUF carries SAM-derived weights, so review the
   [SAM License](https://github.com/facebookresearch/sam-audio/blob/main/LICENSE).
3. Use actual clips with documented ownership/permission. **Start continuous:**

```powershell
python -m barkfilter.cli "samples\dogs-and-dialogue.mp4" `
  --output "results\dialogue_processed.wav" --backend sam-gguf `
  --cli "C:\audio-cpp\audiocpp_cli.exe" `
  --model "models\sam-audio-small-q8_0.gguf" --device cuda `
  --mode continuous --window 8 --overlap 2 --mono `
  --trace "results\dialogue_trace.json"
```

Omit `--mono` to retain two channels, but the current GGUF wrapper separates each
channel independently, roughly doubling model invocations and potentially
altering stereo imaging. Input with >2 channels is downmixed to stereo. The GGUF
backend **restarts the whole CLI process and reloads the model on every call**.
This is acceptable for offline feasibility, NOT proof of continuous-stream
throughput. The relevant audio.cpp outputs are named `target.wav` (bark) and
`residual.wav` (other sounds); the program uses the `residual.wav` file.

### Second AI experiment: official SAM Audio (persistent model in Python)

For a warm-session model that remains in GPU memory between chunks, use Meta's
official Python implementation. It requires gated Hugging Face model approval,
`hf auth login`, and the upstream's substantial Torch/CUDA dependencies.
Use a separate environment if necessary:

```powershell
git clone https://github.com/facebookresearch/sam-audio.git
cd sam-audio
pip install .
cd ..
python -m barkfilter.cli "samples\dogs-and-dialogue.wav" `
  --output "results\official_processed.wav" --backend sam-official `
  --model facebook/sam-audio-small --device cuda --mode continuous `
  --window 8 --overlap 2 --mono
```

See [upstream README](https://github.com/facebookresearch/sam-audio/blob/main/README.md).
The official stack's Windows support is **not verified** here. Its dependency
chain includes DAC-VAE, ImageBind, PE-AV, torchaudio/torchcodec, and other libraries.
SAM produces both target and residual; this wrapper uses `predict_spans=False`
and `reranking_candidates=1` initially for a speed-oriented baseline. A higher
quality setting might need longer inference. Benchmark before selecting it.

### Strategy B/C: detector-triggered / hybrid, without building streaming code

An *automatic* detector is optional. **On Windows, first try the lightweight
[YAMNet ONNX export](https://huggingface.co/audiomagic/yamnet-onnx) (~16.1 MB):**

```powershell
pip install -e ".[onnx-detector]"
pip install huggingface_hub
hf download audiomagic/yamnet-onnx yamnet.onnx yamnet_class_map.csv --local-dir "models\yamnet-onnx"
python -m barkfilter.cli "samples\dogs-and-dialogue.wav" `
  --output "results\hybrid_onnx.wav" --backend sam-gguf `
  --cli "C:\audio-cpp\audiocpp_cli.exe" --model "models\sam-audio-small-q8_0.gguf" `
  --device cuda --mode hybrid --detector yamnet-onnx `
  --yamnet-onnx "models\yamnet-onnx\yamnet.onnx" `
  --yamnet-classes "models\yamnet-onnx\yamnet_class_map.csv" `
  --yamnet-threshold 0.14 --window 8 --overlap 2 --margin 0.5
```

ONNX detection runs on CPU and avoids requiring TensorFlow for detection. This
third-party conversion is not independently benchmarked here; verify detector
scores on real barking and visually inspect predicted spans. The ONNX model is
Apache-2.0; its AudioSet class ontology/labels are CC BY 4.0, per the model
card. The package redistributes neither model nor labels.

Alternatively install and cache the original
[YAMNet TF Hub model](https://tfhub.dev/google/yamnet/1) **once** while online.
Example, with the optional dependencies installed:

```powershell
pip install -e ".[detector]"
python -c "import tensorflow_hub as hub; hub.load('https://tfhub.dev/google/yamnet/1')"
```

The download is cached locally under TensorFlow Hub's cache (see `TFHUB_CACHE_DIR`).
For reliable offline use, save or unpack the model to a stable directory
and pass that model directory through `--yamnet-dir`; do not rely on an
online URL in the offline run. Note that on native Windows TensorFlow/YAMNet
may run on CPU; GPU capability for YAMNet is not required.

```powershell
python -m barkfilter.cli "samples\dogs-and-dialogue.wav" `
  --output "results\hybrid.wav" --backend sam-gguf `
  --cli "C:\audio-cpp\audiocpp_cli.exe" --model "models\sam-audio-small-q8_0.gguf" `
  --device cuda --mode hybrid --detector yamnet --yamnet-dir "models\yamnet" `
  --yamnet-threshold 0.14 --window 8 --overlap 2 --margin 0.5
```

YAMNet's `Bark` class is chosen as the target; it uses ~0.96s frames and ~0.48s
hops, so brief events can be missed. The threshold **0.14 is only an unvalidated
starting point**, not a calibrated bark probability. Verify false negatives.
For manually annotated ground-truth event windows, use `--detector manual
--spans demos\speech_overlap_spans.json`. Manual spans are NOT AI detection.
Never label manually gated accuracy as automatic accuracy.

### Reproducible performance benchmarks on the actual RTX 5090

```powershell
.\scripts\run_windows_benchmarks.ps1 `
  -AudioCpp "C:\audio-cpp\audiocpp_cli.exe" `
  -Model "models\sam-audio-small-q8_0.gguf" `
  -Source "samples\real_barks_2_minutes.wav"
# For persistent official SAM, in the separately installed Python environment:
.\scripts\run_windows_official_benchmark.ps1 -Source "samples\real_barks_2_minutes.wav"
```

The benchmark tracks first model construction (official), per-window wall time,
end-to-end wall time, throughput / RTF, audio-window and overlap lengths,
per-window inference deadlines, sampled process+child RSS/CPU and NVIDIA VRAM
used via `nvidia-smi` when available. Its GPU number is a **total-card reading**;
`gpu_peak_delta_mb` is a noisy differential, not precise allocated model memory.
Usage samples can miss very fast peaks, and external applications affect VRAM.
`model_call_seconds` for GGUF includes every subprocess startup/model reload.
Warm official model calls reuse one Python instance. Clip shorter than benchmark
duration is tiled and prominently flagged in JSON: use authentic long clips for
sustained-throughput claims.

**Important: 3–10 seconds of buffering feasibility requires BOTH** serialized
steady-state RTF below 1 for long continuous material (ideally headroom) **and**
every window's compute taking less than its hop/deadline, with initial usable
output available within the allowable playback buffer. Detecting a bark after
it happens cannot erase audio already played; initial prebuffer or full
pre-analysis is essential. Native audio.cpp CLI per-chunk cold starts cannot
substitute for a persistent inference benchmark. The benchmark's simple
`window + max_call <= 10` criterion is a conservative budget approximation, not
a streaming latency guarantee.

### Synthetic BEFORE/AFTER listening fixtures (NOT AI demonstrations)

`demos/` contains 10 procedurally generated scenarios with `*_input.wav`,
`*_baseline_ducked.wav` and `*_oracle_clean_NOT_MODEL.wav` (plus JSON annotations).
They exercise the pipeline, allow independent review of expected results and
show how naive muting damages overlapped speech or music. They do NOT contain
real dog recordings, and the “clean” files come from the synthesizer's separate
signal stems — no model generated them. The sound labeled “bark” is a synthetic
percussive proxy, so passing these scenarios alone cannot establish model quality.
`espeak` is used for the synthetic dialogue if installed; otherwise an original
voiced-tone fallback is used. The generation script and provenance are in the
package. For real before/after AI demonstrations, run actual SAM weights on
appropriately licensed real dog/speech/music sound stems and save the outputs.

Evaluate separated results **only** with a real reference mixture where the
unmixed source tracks are licensed and retained:

```powershell
python -m barkfilter.evaluate --original demos\music_overlap_input.wav `
  --processed results\music_processed.wav `
  --clean demos\music_overlap_oracle_clean_NOT_MODEL.wav `
  --spans demos\music_overlap_spans.json
```

The evaluator reports improvement in residual-vs-clean error within annotated
bark spans and alteration outside them. The overlapping-bark error includes all
artifacts, so it is NOT a stand-alone “dB of bark reduction” measure. Conduct
blind A/B listening on true canine recordings, with speech intelligibility,
music timbre, transients, stereo coherence and no-bark null cases.

### Model comparison (status on 2026-10-19)

| Candidate | Bark **separation**? | Windows/availability | License | Performance data in THIS project |
|---|---|---|---|---|
| Meta SAM Audio Small (official) | Yes, target & residual | Python 3.11+, CUDA recommended; gated HF access; Windows dependency compatibility untested | Meta SAM License 2025 | Not run; no 5090 measurements |
| SAM Audio Small Q8_0 GGUF via audio.cpp | Yes, target & residual | Native Windows CUDA package; ~2.38 GB downloadable weight; runtime requires CUDA DLLs | SAM-derived weight: SAM License; audio.cpp runtime Apache 2.0; T5 component Apache 2.0 | Not run; third-party measurements not treated as tests |
| SAM Audio Base/Large GGUF | Yes | Windows CUDA; much larger weights, potential higher demands | As above | Not run |
| AudioSep | Yes, text-conditioned separated source (subtract from mixture requires testing) | Open-source Python, project includes Windows Conda environment example, checkpoint separately downloaded | GitHub code MIT; verify checkpoint provenance/license separately | Not run |
| YAMNet ONNX (~16 MB) | **NO**; detects an AudioSet `Bark` event | Native Windows ONNX Runtime CPU; local model+CSV | Google model Apache 2.0; ontology labels CC BY 4.0; ONNX export Apache 2.0 | Not run; detector adapter mock-tested |
| YAMNet TF Hub | **NO**; detects an AudioSet `Bark` event | TensorFlow CPU on native Windows; local cache | Apache 2.0 | Not run |
| Spectral gate / voice-noise suppressor | Not reliably selective; can damage speech/music | Wide hardware support | Depends on specific implementation | Not run |

Sources: [Meta official](https://github.com/facebookresearch/sam-audio),
[Meta license](https://github.com/facebookresearch/sam-audio/blob/main/LICENSE),
[GGUF model card](https://huggingface.co/audio-cpp/SAM-Audio-GGUF),
[audio.cpp runtime](https://github.com/0xShug0/audio.cpp),
[AudioSep](https://github.com/Audio-AGI/AudioSep),
[YAMNet](https://www.tensorflow.org/hub/tutorials/yamnet),
[YAMNet ONNX](https://huggingface.co/audiomagic/yamnet-onnx).
Please review current upstream licenses/terms before redistribution. We don't
redistribute third-party weights, executable backends, or external media.

### Feasibility and next phase decision

| Strategy | Accuracy/quality expectation | Compute | Main failure risk | Assessment |
|---|---|---|---|---|
| A. Continuous SAM separation | Potentially catches every bark | Highest | Changes no-bark scenes, persistent throughput | Necessary baseline first |
| B. Detector-triggered | Avoids model when no bark | Lower average | Missed barking, abrupt edits at detection spans | Needs detector calibration |
| C. Hybrid context + overlap | Avoids no-bark alteration; smoother edits | Variable | Bark leaked across context/overlap, boundary artifacts | Best **candidate** for Phase 0B pending measured quality/latency |

**GO**: Additional local offline SAM model evaluation on the user's RTX 5090.

**NO-GO (not yet verified)**: Claiming universal live 3–10s buffered streaming
playback is achievable. This is an evidence-gate decision, not proof that
real-time separation is impossible. The missing evidence is (1) source-separated
real dog samples, (2) blind preservation scores, (3) sustained RTX 5090 warm
inference including overlap and detector overhead, and (4) no-bark false positives.

For Phase 0B, if quality and throughput pass: persistent native worker (rather
than per-window CLI subprocess), rolling 6–10s context, detector with pre-roll and
post-roll, overlap-add, and exact passthrough for confident no-bark sections.
Only then consider playback integration. Investigate stereo-channel coherence,
DRM/audio-capture constraints and TV routing **after** separation quality is proven.

### Files

- `barkfilter/engines.py`: Meta SAM and GGUF backends; identity backend for tests.
- `barkfilter/detection.py`: YAMNet (TensorFlow and ONNX), manual ground-truth and all-audio selectors.
- `barkfilter/pipeline.py`: overlap blending, passthrough, traces.
- `barkfilter/audio.py`: FFmpeg decode, sample-rate conversion and WAV output.
- `barkfilter/benchmark.py`: separate sampled CPU/RAM/VRAM and real throughput.
- `barkfilter/demos.py`: synthetic, reproducible fixtures, including hard cases.
- `barkfilter/evaluate.py`: compares outputs against clean reference signals.
- `scripts/`: Windows PowerShell benchmark entrypoints.
- `tests/`: audio I/O, command errors, no-bark identity, overlap, metrics, video.
- `results/VALIDATION_ONLY_identity_pipeline_benchmark.json`: **non-AI wrapper overhead**, not an RTX 5090 result.

**Constraints:** The project is offline-only; no streaming interface, no browser
extension, no television integration and no cloud inference. No other project or
repository was modified.

### Git repository audio policy

Generated synthetic demo WAVs are intentionally omitted from Git history. Create all ten original/baseline/oracle comparison sets using `python -m barkfilter.demos --output-dir demos`. The oracle-clean files are synthetic ground truth, **not** AI predictions. No pretrained weights are committed.

## Unattended tests and benchmarks (Phase 0A automation)

Workflow: [BarkFilter unattended benchmarks](https://github.com/ComradeGenosse/BarkFilter/actions/workflows/autonomous-benchmarks.yml).

- **Automatically on relevant pushes and every Monday at 07:17 UTC:** GitHub-hosted **Windows and Linux** runners install the package, run all Python tests, regenerate the ten synthetic mixtures, measure identity/hybrid pipeline behavior, exercise the **actual ONNX YAMNet classifier** (public ~16 MB download), and benchmark CPU wrapper/overlap overhead. Each job uploads its self-contained JSON/WAV result bundle and Markdown summary (14-day retention). These results are NOT dog bark source separation. The YAMNet test exercises real inference on artificial sounds; it does not establish detector precision/recall on authentic dogs.
- **Optional real GPU run, after one-time PC setup:** GitHub Actions → `BarkFilter unattended benchmarks` → Run workflow → `run_gpu=true`. This targets a **self-hosted Windows x64 runner** with custom `barkfilter-gpu` label. It requires a registered runner with the RTX 5090 available, NVIDIA drivers, the CUDA-enabled `audiocpp_cli.exe`, the GGUF model stored locally, and repo variables `BARKFILTER_AUDIOCPP_PATH` and `BARKFILTER_GGUF_MODEL_PATH` pointing to the respective absolute Windows paths. Follow GitHub's [self-hosted runner instructions](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners). Do not run untrusted pull requests on a personal GPU runner. Only workflow-dispatch jobs target it.
- **Run the same full GPU test without GitHub Actions (local Windows/remote desktop):** `python -m pip install -e ".[test]"`, then `$env:BARKFILTER_AUDIOCPP="C:\\path\\audiocpp_cli.exe"; $env:BARKFILTER_MODEL="C:\\path\\sam-audio-small-q8_0.gguf"; ./scripts/run_autonomous_gpu.ps1 -RequireRtx5090`.

GPU output includes a preflight file, six-minute synthetic-case breakdown (four selected 6-second cases), residual WAV outputs, detailed per-window process durations and CPU/RAM/VRAM samples, plus 4/8/16-second GGUF benchmarks. This intentionally **reloads the GGUF model on each separate chunk**, so it is **not** a sustained warm streaming measurement. Long-duration real canine samples and subjective speech/music review remain required before a live streaming GO decision. No approved/gated model is downloaded automatically or redistributed. Never treat identity results, procedural barks, or generated oracle stems as model quality evidence.

To run the CPU suite yourself without a GPU:

```powershell
python -m pip install -e ".[test]"
python -m pytest -q
python -m barkfilter.autonomous --backend identity --mode hybrid --mono --window 4 --overlap 1 --output-dir results/autonomous
python -m pip install -e ".[onnx-detector]" huggingface_hub
python -m barkfilter.detector_benchmark --download --output-dir results/autonomous
```

The `report.json` preserves the details of every attempted case and setup errors, and returns a nonzero exit code for failures. The `.github/workflows/tests.yml` unit-test workflow remains independent.
