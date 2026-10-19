"""Non-interactive offline evaluation. Synthetic evidence is NEVER a real-bark quality claim.

Generate fixtures, exercise all cases, preserve output/metrics when individual cases fail,
and create an honest machine-readable summary for Actions and local Windows GPU runs.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import time
import traceback

import numpy as np

from .audio import read_audio, write_audio
from .cli import add_shared_arguments, create_separator
from .demos import generate
from .detection import AllDetector, ManualDetector
from .evaluate import evaluate
from .pipeline import process_audio
from .benchmark import query_gpu_name, UsageMonitor

DEFAULT_CASES = "speech_overlap,music_overlap,NO_BARK,overlapping_words_HARD,brief_bark,quiet_bark"


def run_suite(*, output_dir: Path, backend: str = "identity", cli: str | None = None,
              model: str | None = None, device: str = "cuda", prompt: str = "dog barking",
              window: float = 8., overlap: float = 2., mono: bool = True,
              mode: str = "continuous", cases: str = DEFAULT_CASES,
              require_gpu: bool = False) -> dict:
    """Run the selected cases. Always persist an explicit report even on model errors."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_dir = output_dir / "synthetic_inputs"
    generate(case_dir)
    names = [x.strip() for x in cases.split(",") if x.strip()]
    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    if not names or len(names) != len(set(names)) or any(n not in manifest["examples"] for n in names):
        raise ValueError("--cases must be unique names present in the synthetic manifest")
    if mode not in ("continuous", "hybrid"):
        raise ValueError("mode must be continuous or hybrid")
    if window <= 0 or overlap < 0 or overlap >= window:
        raise ValueError("Invalid window / overlap")

    results = []
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {"platform": platform.platform(), "gpu_name": query_gpu_name()},
        "backend": backend,
        "mode": mode,
        "window_seconds": window,
        "overlap_seconds": overlap,
        "input_provenance": "procedural sound proxies ONLY; no authentic dog recordings",
        "quality_verification": "NOT VERIFIED with real barking",
        "streaming_go_no_go": "UNDETERMINED; warm sustained GPU measurements and real media required",
        "cases": results,
        "model_initialization_seconds": None,
        "success": False,
    }
    report_path = output_dir / "report.json"
    def save():
        report["success"] = bool(results) and len(results) == len(names) and all(v["status"] == "ok" for v in results)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    if backend == "identity":
        report["quality_verification"] = "NO MODEL: infrastructure/passthrough validation only"
    elif backend not in ("sam-gguf", "sam-official"):
        raise ValueError("Unknown backend")
    save()
    try:
        if require_gpu and not report["hardware"]["gpu_name"]:
            raise RuntimeError("--require-gpu was set but nvidia-smi found no NVIDIA GPU")
        opts = argparse.Namespace(backend=backend, cli=cli, model=model, device=device, prompt=prompt)
        t0 = time.perf_counter()
        separator = create_separator(opts)
        report["model_initialization_seconds"] = time.perf_counter() - t0
    except Exception as exc:
        report["setup_error"] = f"{type(exc).__name__}: {exc}"
        save()
        return report

    for name in names:
        entry = manifest["examples"][name]
        record = {"case": name, "status": "error", "synthetic": True,
                  "ground_truth_spans_used": mode == "hybrid"}
        results.append(record)
        try:
            x, sr = read_audio(case_dir / entry["files"]["input"], channels=1 if mono else 0)
            clean, clean_sr = read_audio(case_dir / entry["files"]["oracle_clean_NOT_MODEL"],
                                         channels=1 if mono else 0)
            if clean_sr != sr or x.shape != clean.shape:
                raise RuntimeError("Fixture mismatch")
            detector = (AllDetector() if mode == "continuous"
                        else ManualDetector(case_dir / entry["spans"]))
            with UsageMonitor() as monitor:
                result = process_audio(x, sr, separator, detector=detector, mode=mode,
                                       window_seconds=window, overlap_seconds=overlap,
                                       event_margin_seconds=.5)
            out = output_dir / "processed" / f"{name}_{backend}.wav"
            write_audio(out, result.audio, sr)
            truth = ManualDetector(case_dir / entry["spans"]).detect(x, sr)
            metrics = evaluate(x, result.audio, clean, sr, truth)
            record.update({
                "status": "ok", "input_seconds": len(x) / sr,
                "channels": int(x.shape[1]), "wall_seconds": result.wall_seconds,
                "rtf": result.wall_seconds / (len(x) / sr),
                "processed_wav": out.relative_to(output_dir).as_posix(),
                "model_calls": sum(t.separated for t in result.trace),
                "window_calls": [asdict(t) for t in result.trace],
                "no_bark_bit_identical_before_wav_roundtrip": bool(np.array_equal(result.audio, x)) if not truth else None,
                "reference_metrics": metrics,
                "resource_usage": monitor.metrics(),
            })
        except Exception as exc:
            record.update({"error": f"{type(exc).__name__}: {exc}",
                           "traceback": traceback.format_exc(limit=5)})
        finally:
            save()
        print(f"{name}: {record['status']} ({record.get('wall_seconds', 'no timing')} seconds)", flush=True)

    return report


def write_summary(report: dict, output_dir: Path) -> None:
    tested = sum(r.get("status") == "ok" for r in report["cases"])
    content = ["# BarkFilter unattended benchmark", "",
               f"- Engine: `{report['backend']}` / mode `{report['mode']}`",
               f"- Host GPU: `{report['hardware']['gpu_name'] or 'unavailable'}`",
               f"- Synthetic cases passed: **{tested}/{len(report['cases'])}**",
               f"- Quality: **{report['quality_verification']}**",
               f"- Streaming feasibility: **{report['streaming_go_no_go']}**",
               "- Detailed timings, resource samples and case errors: `report.json`", "",
               "**Synthetic sound proxies and identity timing are NOT real bark removal or GPU feasibility evidence.**"]
    if report.get("setup_error"):
        content.append(f"- Setup error: `{report['setup_error']}`")
    (output_dir / "SUMMARY.md").write_text("\n".join(content) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/autonomous")
    parser.add_argument("--cases", default=DEFAULT_CASES,
                        help="Comma-delimited synthetic cases from generated fixture manifest")
    parser.add_argument("--require-gpu", action="store_true",
                        help="Fail explicitly if nvidia-smi cannot find an NVIDIA GPU")
    add_shared_arguments(parser)
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    report = run_suite(output_dir=output, backend=args.backend, cli=args.cli,
                       model=args.model, device=args.device, prompt=args.prompt,
                       window=args.window, overlap=args.overlap, mono=args.mono,
                       mode=args.mode, cases=args.cases, require_gpu=args.require_gpu)
    write_summary(report, output)
    print(f"Report: {output / 'report.json'}; success={report['success']}")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
