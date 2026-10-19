"""Measured against a known clean reference; not a substitute for blind listening."""
from __future__ import annotations
import argparse
import json
import numpy as np
from .audio import read_audio
from .detection import ManualDetector


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x,dtype=np.float64)**2))) if np.size(x) else 0.0


def evaluate(original, processed, clean, sample_rate, spans):
    if original.shape != processed.shape or original.shape != clean.shape:
        raise ValueError("All signals must have same length and channels")
    n=len(original)
    mask = np.zeros(n, dtype=bool)
    for s in spans:
        mask[max(0,round(s.start*sample_rate)):min(n,round(s.end*sample_rate))] = True
    noise_before = rms(original[mask]-clean[mask])
    noise_after = rms(processed[mask]-clean[mask])
    outside_error = rms(processed[~mask]-clean[~mask])
    outside_signal = rms(clean[~mask])
    nonbark_error_db = 20*np.log10(max(outside_error,1e-12)/max(outside_signal,1e-12))
    improvement = 20*np.log10(max(noise_before,1e-12)/max(noise_after,1e-12))
    return {"inside_annotated_barks_error_improvement_db":float(improvement),
            "outside_barks_error_to_clean_db":float(nonbark_error_db),
            "outside_barks_sample_rmse": outside_error,
            "whole_file_rmse_vs_clean": rms(processed-clean),
            "clean_reference_used":True,
            "warning":"Inside-bark metric includes all processing artifacts, not isolated bark power"}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original",required=True)
    p.add_argument("--processed",required=True)
    p.add_argument("--clean",required=True)
    p.add_argument("--spans",required=True)
    p.add_argument("--output-json")
    a=p.parse_args(argv)
    original,sr=read_audio(a.original)
    processed,_=read_audio(a.processed)
    clean,_=read_audio(a.clean)
    spans=ManualDetector(a.spans).detect(original,sr)
    result=evaluate(original, processed, clean, sr, spans)
    print(json.dumps(result,indent=2))
    if a.output_json:
        from pathlib import Path
        Path(a.output_json).write_text(json.dumps(result,indent=2),encoding="utf-8")


if __name__=="__main__": main()
