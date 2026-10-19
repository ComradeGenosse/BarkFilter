"""Minimal offline CLI: WAV/FLAC/audio/video in, processed WAV out."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from .audio import read_audio, write_audio
from .detection import AllDetector, ManualDetector, YAMNetDetector, YAMNetOnnxDetector
from .engines import IdentitySeparator, SamGGUFSeparator, SamOfficialSeparator
from .pipeline import process_audio


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["sam-gguf", "sam-official", "identity"], required=True)
    parser.add_argument("--cli", help="Path to audiocpp_cli.exe (sam-gguf)")
    parser.add_argument("--model", help="GGUF path or official HF model ID")
    parser.add_argument("--device", default="cuda", help="GGUF backend (cuda/cpu/vulkan) or official torch device")
    parser.add_argument("--prompt", default="dog barking", help="Lowercase target sound description")
    parser.add_argument("--mode", choices=["continuous", "hybrid"], default="continuous")
    parser.add_argument("--detector", choices=["all", "manual", "yamnet", "yamnet-onnx"], default="all")
    parser.add_argument("--spans", help="JSON manual ground-truth bark span file")
    parser.add_argument("--yamnet-dir", help="Locally downloaded or cached YAMNet SavedModel directory")
    parser.add_argument("--yamnet-onnx", help="Path to locally downloaded yamnet.onnx")
    parser.add_argument("--yamnet-classes", help="Path to locally downloaded yamnet_class_map.csv")
    parser.add_argument("--yamnet-threshold", type=float, default=.14)
    parser.add_argument("--window", type=float, default=8.0, help="Chunk size, seconds")
    parser.add_argument("--overlap", type=float, default=2.0, help="Chunk overlap, seconds")
    parser.add_argument("--margin", type=float, default=.5, help="Event interval padding, seconds")
    parser.add_argument("--mono", action="store_true", help="Downmix source to mono for faster processing")


def create_separator(args):
    if args.backend == "identity":
        return IdentitySeparator()
    if args.backend == "sam-gguf":
        if not args.cli or not args.model:
            raise ValueError("sam-gguf requires --cli and --model")
        return SamGGUFSeparator(args.cli, args.model, backend=args.device, prompt=args.prompt)
    return SamOfficialSeparator(args.model or "facebook/sam-audio-small", prompt=args.prompt,
                                device=args.device)


def create_detector(args):
    if args.mode == "continuous":
        return AllDetector()
    if args.detector == "manual":
        if not args.spans:
            raise ValueError("--detector manual requires --spans")
        return ManualDetector(args.spans)
    if args.detector == "yamnet":
        if not args.yamnet_dir:
            raise ValueError("--detector yamnet requires --yamnet-dir")
        return YAMNetDetector(args.yamnet_dir, threshold=args.yamnet_threshold)
    if args.detector == "yamnet-onnx":
        if not args.yamnet_onnx or not args.yamnet_classes:
            raise ValueError("--detector yamnet-onnx requires --yamnet-onnx and --yamnet-classes")
        return YAMNetOnnxDetector(args.yamnet_onnx, args.yamnet_classes,
                                  threshold=args.yamnet_threshold)
    return AllDetector()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="WAV, FLAC or FFmpeg-readable media file")
    ap.add_argument("--output", required=True, help="Result WAV")
    ap.add_argument("--trace", help="Optional processing trace JSON")
    add_shared_arguments(ap)
    args = ap.parse_args(argv)
    if args.backend == "identity":
        print("WARNING: identity backend does not detect or remove barking; pipeline validation only")
    x, sr = read_audio(args.input, channels=1 if args.mono else 0)
    engine = create_separator(args)
    detector = create_detector(args)
    result = process_audio(x, sr, engine, detector=detector, mode=args.mode,
                           window_seconds=args.window, overlap_seconds=args.overlap,
                           event_margin_seconds=args.margin)
    write_audio(args.output, result.audio, sr)
    print(f"Output: {args.output}; {len(x)/sr:.2f}s input; {result.wall_seconds:.2f}s processing; "
          f"{sum(t.separated for t in result.trace)} model calls; channels={x.shape[1]}")
    if args.trace:
        payload = {"source": str(args.input), "backend": engine.name, "mode": args.mode,
                   "detector": args.detector, "sample_rate": sr,
                   "window_seconds": args.window, "overlap_seconds": args.overlap,
                   "wall_seconds": result.wall_seconds, "bark_spans": [asdict(x) for x in result.spans],
                   "trace": [asdict(x) for x in result.trace]}
        Path(args.trace).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
