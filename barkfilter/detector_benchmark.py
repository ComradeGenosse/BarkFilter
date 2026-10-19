"""Unattended REAL YAMNet ONNX CPU model smoke/latency benchmark (no separation).

Downloads openly licensed model weights once through Hugging Face Hub if requested.
Results on synthesized bark-LIKE waveforms cannot establish real-dog accuracy.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import tempfile
from urllib.request import urlopen
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time
from .audio import read_audio
from .demos import generate
from .detection import YAMNetOnnxDetector


MODEL_REPO = "audiomagic/yamnet-onnx"
ESC50_RAW = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master"
ESC10_CATEGORIES = ("dog", "rooster", "crying_baby", "chainsaw", "clock_tick")


def choose_esc10_samples(csv_text: str, *, per_category: int = 3) -> list[dict]:
    """Deterministic small, balanced non-commercially-safe CC BY ESC-10 subset."""
    if per_category < 1 or per_category > 10:
        raise ValueError("per_category must be between 1 and 10")
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    chosen = []
    for category in ESC10_CATEGORIES:
        candidates = sorted((r for r in rows if r["category"] == category and
                             r["esc10"].casefold() == "true"),
                            key=lambda r: r["filename"])
        if len(candidates) < per_category:
            raise ValueError(f"ESC-10 missing {category} sample count")
        chosen.extend(candidates[:per_category])
    return chosen


def benchmark_esc10(detector, report: dict, *, per_category: int = 3) -> None:
    """Fetch original CC BY files into temporary disk; emit metadata, never media."""
    with urlopen(ESC50_RAW + "/meta/esc50.csv", timeout=30) as response:
        meta = response.read(500_000).decode("utf-8-sig")
    chosen = choose_esc10_samples(meta, per_category=per_category)
    records = []
    with tempfile.TemporaryDirectory(prefix="barkfilter-esc10-") as d:
        for row in chosen:
            name = row["filename"]
            if Path(name).name != name:
                raise ValueError("Unexpected unsafe ESC-10 filename")
            url = ESC50_RAW + "/audio/" + name
            with urlopen(url, timeout=30) as response:
                body = response.read(2_000_000)
            if len(body) < 1000 or len(body) >= 2_000_000:
                raise RuntimeError("Invalid or unusually large ESC-10 sample")
            dest = Path(d) / name
            dest.write_bytes(body)
            x, sr = read_audio(dest, channels=1)
            tick = time.perf_counter()
            spans = detector.detect(x, sr)
            records.append({"filename": name, "category": row["category"],
                            "source_url": url, "sha256": hashlib.sha256(body).hexdigest(),
                            "duration_seconds": len(x) / sr, "detection_seconds": time.perf_counter() - tick,
                            "bark_detected": bool(spans), "detected_spans": [
                                {"start": s.start, "end": s.end} for s in spans]})
    positives = [x for x in records if x["category"] == "dog"]
    negatives = [x for x in records if x["category"] != "dog"]
    report["real_esc10"] = {
        "source": "ESC-10 subset of ESC-50; K. J. Piczak (2015)",
        "license": "CC BY 3.0 (ESC-10 only); original file attribution in upstream LICENSE",
        "meaning": "Clip-level category discrimination, NOT bark timestamp accuracy",
        "positive_class": "dog", "negative_classes": [v for v in ESC10_CATEGORIES if v != "dog"],
        "true_positive_clips": sum(r["bark_detected"] for r in positives),
        "positive_clips": len(positives),
        "false_positive_clips": sum(r["bark_detected"] for r in negatives),
        "negative_clips": len(negatives),
        "examples": records,
    }


def run(output_dir: Path, *, model_path: str | None = None, class_map: str | None = None,
        download: bool = False, threshold: float = .14, esc10: bool = False) -> dict:
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
        if esc10:
            benchmark_esc10(detector, report)
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
    p.add_argument("--esc10", action="store_true", help="Run a small REAL CC BY ESC-10 bark/nonbark classifier benchmark")
    a = p.parse_args(argv)
    result = run(Path(a.output_dir), model_path=a.model, class_map=a.classes,
                 download=a.download, threshold=a.threshold, esc10=a.esc10)
    print(f"YAMNet model test: {result['status']}; {len(result['cases'])} synthetic cases")
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
