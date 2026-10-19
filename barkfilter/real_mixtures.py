"""Reproducible real-dog / synthetic speech+music separation challenge.

Uses authentic CC BY ESC-10 dog recordings at runtime; does NOT redistribute samples
in Git. Clean source is synthetic and known, so reference error is measurable.
ESC-10 recordings may include unrelated background audio: error reduction here is
NOT a pure isolated-dog-bark SNR measurement. Listening is still mandatory.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import time
from urllib.request import urlopen
import numpy as np

from .audio import read_audio, write_audio
from .cli import add_shared_arguments, create_separator
from .demos import generate
from .detection import ManualDetector, AllDetector
from .evaluate import evaluate
from .pipeline import process_audio
from .benchmark import UsageMonitor, query_gpu_name
from .detector_benchmark import choose_esc10_samples, ESC50_RAW


def build_mixtures(outdir: Path, *, fetch=None) -> dict:
    """Create 2 authentic dog overlays + 1 clean control and attribution. Injectable fetch for tests."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fetch = fetch or (lambda url: urlopen(url, timeout=30).read(2_000_000))
    source_dir = outdir / "procedural_clean"
    generate(source_dir)
    meta = fetch(ESC50_RAW + "/meta/esc50.csv").decode("utf-8-sig")
    rows = [x for x in choose_esc10_samples(meta, per_category=2) if x["category"] == "dog"]
    LICENSE_URL = ESC50_RAW + "/LICENSE"
    (outdir / "ESC50_UPSTREAM_LICENSE.txt").write_bytes(fetch(LICENSE_URL))
    specs = [("speech_real_dog", "speech_overlap", rows[0]),
             ("music_real_dog", "music_overlap", rows[1]),
             ("NO_BARK_clean_control", "NO_BARK", None)]
    manifest = {"dataset": "ESC-10 (subset of ESC-50)",
                "dataset_license": "CC BY 3.0; see ESC50_UPSTREAM_LICENSE.txt",
                "dataset_url": "https://github.com/karolpiczak/ESC-50",
                "method": "Authentic field-recorded dog clip mixed with procedural clean speech/music",
                "caveat": "ESC dog recordings may contain environmental audio, not isolated barks",
                "cases": {}}
    for name, clean_case, row in specs:
        clean_path = source_dir / f"{clean_case}_oracle_clean_NOT_MODEL.wav"
        clean, sr = read_audio(clean_path, channels=1)
        bark_span = []
        credit = None
        mixed = clean.copy()
        if row is not None:
            url = ESC50_RAW + "/audio/" + row["filename"]
            body = fetch(url)
            if len(body) >= 2_000_000 or len(body) < 1000:
                raise RuntimeError("Unexpected ESC-10 WAV content length")
            with tempfile.TemporaryDirectory(prefix="barkfilter-real-dog-") as t:
                file = Path(t) / row["filename"]
                file.write_bytes(body)
                dog, dog_sr = read_audio(file, channels=1)
            if dog_sr != sr:
                raise RuntimeError("Sample rate mismatch")
            start = round(.5 * sr)
            count = min(len(dog), len(clean)-start)
            dog = dog[:count]
            if not count or np.max(np.abs(dog)) < 1e-5:
                raise RuntimeError("Empty/silent dog sample")
            dog = dog * (0.38 / np.max(np.abs(dog)))
            mixed[start:start+count] += dog
            # Preserve relative mixture proportions without individual-source clipping.
            peak = float(np.max(np.abs(mixed)))
            if peak > .97:
                mixed *= .97/peak
                clean *= .97/peak
            bark_span = [[start/sr, (start+count)/sr]]
            credit = {"filename": row["filename"], "url": url,
                      "sha256": hashlib.sha256(body).hexdigest(),
                      "source_filename": row["src_file"], "license": "ESC-10 CC BY 3.0"}
        paths = {"original": f"{name}_original.wav", "clean_reference": f"{name}_clean.wav",
                 "spans": f"{name}_overlay_span.json"}
        write_audio(outdir / paths["original"], mixed, sr)
        write_audio(outdir / paths["clean_reference"], clean, sr)
        (outdir / paths["spans"]).write_text(json.dumps({"barks": bark_span},indent=2),encoding="utf-8")
        manifest["cases"][name] = {"files": paths, "dog_credit": credit, "dog_overlay_spans": bark_span}
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run(output_dir: Path, args) -> dict:
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report={"schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "backend": args.backend, "gpu": query_gpu_name(),
            "quality_caveat": "Real dog sources but only synthetic speech/music; no blind listening or bark-only stem",
            "feasibility": "UNDETERMINED", "cases": [], "success": False}
    dest=output_dir/"real_dog_report.json"
    def save():
        dest.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    save()
    try:
        manifest=build_mixtures(output_dir/"mixtures")
        t=time.perf_counter()
        separator=create_separator(args)
        report["model_initialization_seconds"]=time.perf_counter()-t
        for name, record in manifest["cases"].items():
            row={"case":name,"status":"error","dog_credit":record["dog_credit"]}
            report["cases"].append(row)
            try:
                root=output_dir/"mixtures"
                src,sr=read_audio(root/record["files"]["original"],channels=1)
                clean,_=read_audio(root/record["files"]["clean_reference"],channels=1)
                annotated=ManualDetector(root/record["files"]["spans"])
                detector=AllDetector() if args.mode=="continuous" else annotated
                with UsageMonitor() as monitor:
                    r=process_audio(src,sr,separator,detector=detector,mode=args.mode,
                                    window_seconds=args.window,overlap_seconds=args.overlap,
                                    event_margin_seconds=args.margin)
                out=output_dir / f"{name}_{args.backend}_processed.wav"
                write_audio(out,r.audio,sr)
                row.update({"status":"ok","processed_wav":out.name,
                            "wall_seconds":r.wall_seconds,"model_calls":sum(w.separated for w in r.trace),
                            "per_window": [asdict(w) for w in r.trace],
                            "metrics":evaluate(src,r.audio,clean,sr,annotated.detect(src,sr)),
                            "resources":monitor.metrics()})
            except Exception as exc:
                row["error"]=f"{type(exc).__name__}: {exc}"
            finally:
                save()
        report["success"]=len(report["cases"])==3 and all(r["status"]=="ok" for r in report["cases"])
    except Exception as exc:
        report["setup_error"]=f"{type(exc).__name__}: {exc}"
    save()
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",default="results/real_dog")
    add_shared_arguments(p)
    a=p.parse_args(argv)
    r=run(Path(a.output_dir),a)
    print(f"Real dog source mixtures: {'SUCCESS' if r['success'] else 'FAILED'}, "
          f"{len(r['cases'])} cases, backend={a.backend}")
    return 0 if r["success"] else 1


if __name__=="__main__":
    raise SystemExit(main())
