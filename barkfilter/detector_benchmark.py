"""Unattended REAL YAMNet ONNX CPU model smoke/latency benchmark (no separation).

Downloads openly licensed model weights once through Hugging Face Hub if requested.
Results on synthesized bark-LIKE waveforms cannot establish real-dog accuracy.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time
from .audio import read_audio
from .demos import generate
from .detection import YAMNetOnnxDetector


MODEL_REPO = "audiomagic/yamnet-onnx"


def run(output_dir: Path, *, model_path: str | None = None, class_map: str | None = None,
        download: bool = False, threshold: float = .14) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "model_repository": MODEL_REPO, "model_kind": "YAMNet ONNX classifier, NOT separation",
              "synthetic_bark_accuracy_warning": "Synthetic bark-like proxy; precision/recall on real dogs NOT measured",
              "cpu": platform.processor(), "status": "error", "cases": []}
    path = output_dir / "detector_report.json"
    def save():
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        if download:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(MODEL_REPO, "yamnet.onnx")
            class_map = hf_hub_download(MODEL_REPO, "yamnet_class_map.csv")
        if not model_path or not class_map:
            raise ValueError("Provide --model and --classes or use --download")
        start = time.perf_counter()
        detector = YAMNetOnnxDetector(model_path, class_map, threshold=threshold)
        report["model_load_seconds"] = time.perf_counter() - start
        samples = output_dir / "detector_synthetic_inputs"
        generate(samples)
        manifest = json.loads((samples / "manifest.json").read_text(encoding="utf-8"))
        for case, entry in manifest["examples"].items():
            row = {"case": case, "spans": [], "timing_seconds": None}
            report["cases"].append(row)
            x, sr = read_audio(samples / entry["files"]["input"], channels=1)
            tick = time.perf_counter()
            spans = detector.detect(x, sr)
            row["timing_seconds"] = time.perf_counter() - tick
            row["spans"] = [{"start": s.start, "end": s.end} for s in spans]
            row["is_synthetic_no_bark_control"] = entry["simulated_events"] == 0
        report["status"] = "ok"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        save()
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="results/autonomous")
    p.add_argument("--model")
    p.add_argument("--classes")
    p.add_argument("--threshold", type=float, default=.14)
    p.add_argument("--download", action="store_true")
    a = p.parse_args(argv)
    result = run(Path(a.output_dir), model_path=a.model, class_map=a.classes,
                 download=a.download, threshold=a.threshold)
    print(f"YAMNet model test: {result['status']}; {len(result['cases'])} synthetic cases")
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
