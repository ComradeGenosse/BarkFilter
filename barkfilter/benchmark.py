"""Repeatable end-to-end benchmarking; captures process memory/CPU/GPU when available."""
from __future__ import annotations
import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import threading
import time
import statistics
import psutil
from .audio import read_audio, fit_length
from .cli import add_shared_arguments, create_detector, create_separator
from .pipeline import process_audio


def gpu_vram_mb():
    if not shutil.which("nvidia-smi"):
        return None
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=3)
        if p.returncode == 0:
            return sum(float(x.strip()) for x in p.stdout.splitlines())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return None


class UsageMonitor:
    def __init__(self, interval: float = .20):
        self.interval = interval
        self.stop = threading.Event()
        self.rss_peaks = []
        self.cpu_peaks = []
        self.gpu_peaks = []
        self.parent = psutil.Process()
        self.thread = threading.Thread(target=self.worker, daemon=True)

    def worker(self):
        while not self.stop.is_set():
            try:
                processes = [self.parent] + self.parent.children(recursive=True)
                rss = sum(p.memory_info().rss for p in processes if p.is_running()) / (1024**2)
                cpu = sum(p.cpu_percent(interval=None) for p in processes if p.is_running())
                self.rss_peaks.append(rss)
                self.cpu_peaks.append(cpu)
                gpu = gpu_vram_mb()
                if gpu is not None:
                    self.gpu_peaks.append(gpu)
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                pass
            self.stop.wait(self.interval)

    def __enter__(self):
        self.gpu_start = gpu_vram_mb()
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=2)

    def metrics(self):
        return {"peak_process_tree_rss_mb": max(self.rss_peaks, default=None),
                "peak_process_tree_cpu_percent": max(self.cpu_peaks, default=None),
                "peak_total_gpu_vram_mb": max(self.gpu_peaks, default=None),
                "gpu_start_vram_mb": self.gpu_start,
                "gpu_peak_delta_mb": (max(self.gpu_peaks)-self.gpu_start
                                      if self.gpu_peaks and self.gpu_start is not None else None)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--durations", default="4,8,16,32", help="Clip lengths (seconds)")
    parser.add_argument("--windows", default="4,8,12", help="Window lengths (seconds)")
    parser.add_argument("--repeats", type=int, default=2)
    add_shared_arguments(parser)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be at least one")
    if args.backend == "identity":
        print("WARNING: identity backend measures wrapper overhead ONLY, never model feasibility")
    x, sr = read_audio(args.input, channels=1 if args.mono else 0)
    if not len(x):
        parser.error("Empty input")
    durations = [float(i) for i in args.durations.split(",")]
    windows = [float(i) for i in args.windows.split(",")]
    # Cold start includes model construction; repeated calls reuse object.
    t0 = time.perf_counter()
    engine = create_separator(args)
    load_seconds = time.perf_counter()-t0
    detector = create_detector(args)
    records = []
    for dur in durations:
        if dur <= 0:
            parser.error("duration must be positive")
        n = round(dur*sr)
        # Repetition is visible in the report: use real long sample for credible quality.
        repeated = len(x) < n
        if repeated:
            import numpy as np
            chunk = np.tile(x, (int((n+len(x)-1)//len(x)), 1))[:n]
        else:
            chunk = x[:n]
        for window in windows:
            if window <= args.overlap:
                parser.error("every window must be greater than overlap")
            for trial in range(args.repeats):
                t = time.perf_counter()
                with UsageMonitor() as monitor:
                    result = process_audio(chunk, sr, engine, detector=detector,
                                           window_seconds=window, overlap_seconds=args.overlap,
                                           mode=args.mode, event_margin_seconds=args.margin)
                elapsed = time.perf_counter()-t
                separated = [v for v in result.trace if v.separated]
                hop_seconds = window-args.overlap
                max_call = max((v.inference_seconds for v in separated), default=0)
                records.append({"duration_seconds":len(chunk)/sr, "window_seconds":window,
                                "overlap_seconds":args.overlap,"trial":trial,
                                "sample_repeated_to_fill_duration":repeated,
                                "wall_seconds":elapsed, "rtf": elapsed/(len(chunk)/sr),
                                "throughput_audio_sec_per_wall_sec":(len(chunk)/sr)/elapsed if elapsed else None,
                                "model_calls":len(separated), "model_call_seconds": [v.inference_seconds for v in separated],
                                "max_model_call_seconds":max_call,
                                "hop_deadline_seconds":hop_seconds,
                                "meets_hop_deadline_serial":max_call < hop_seconds,
                                "meets_10sec_startup_budget_one_window":window + max_call <= 10,
                                **monitor.metrics()})
                print(f"{dur:5.1f}s clip / {window:4.1f}s window / trial {trial}: "
                      f"{elapsed:.3f}s, RTF={elapsed/(len(chunk)/sr):.3f}, "
                      f"{len(separated)} calls")
    payload = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "host":platform.platform(), "cpu":platform.processor(), "backend":engine.name,
               "model":args.model, "device":args.device, "input":args.input, "mode":args.mode,
               "detector":args.detector, "window_backend_process_reused":args.backend == "sam-official",
               "gguf_each_call_cold_process":args.backend == "sam-gguf",
               "model_load_seconds":load_seconds, "gpu_name":query_gpu_name(),
               "warning":("identity means no actual separation" if args.backend=="identity" else
                          "GGUF reloads model per window" if args.backend=="sam-gguf" else ""),
               "trials":records}
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {out}")


def query_gpu_name():
    if not shutil.which("nvidia-smi"):
        return None
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=3)
        return p.stdout.strip() if p.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


if __name__ == "__main__":
    main()
