"""Ensure unattended jobs never mistake pipeline health for model quality."""
import json
import numpy as np
import pytest
from barkfilter import autonomous
from barkfilter.detection import Interval


def test_autonomous_identity_suite_generates_artifacts(tmp_path):
    out = tmp_path / "run"
    report = autonomous.run_suite(output_dir=out, backend="identity",
                                  cases="speech_overlap,NO_BARK", mode="hybrid",
                                  window=4, overlap=1)
    autonomous.write_summary(report, out)
    assert report["success"]
    assert "NO MODEL" in report["quality_verification"]
    assert "UNDETERMINED" in report["streaming_go_no_go"]
    assert report["cases"][0]["model_calls"] > 0
    assert report["cases"][1]["model_calls"] == 0
    assert report["cases"][1]["no_bark_bit_identical_before_wav_roundtrip"] is True
    assert "NOT real bark" in (out / "SUMMARY.md").read_text()
    assert (out / "processed" / "NO_BARK_identity.wav").exists()
    assert json.loads((out / "report.json").read_text())["success"]


def test_missing_real_model_fails_and_persists_report(tmp_path):
    out = tmp_path / "failure"
    r = autonomous.run_suite(output_dir=out, backend="sam-gguf", model="missing.gguf",
                             cli="missing-exe", cases="NO_BARK")
    assert r["success"] is False
    assert "setup_error" in r
    assert json.loads((out / "report.json").read_text())["success"] is False


def test_require_gpu_fails_loudly_on_gpu_absence(tmp_path, monkeypatch):
    monkeypatch.setattr(autonomous, "query_gpu_name", lambda: None)
    out = tmp_path / "gpu"
    r = autonomous.run_suite(output_dir=out, backend="identity", cases="NO_BARK", require_gpu=True)
    assert r["success"] is False
    assert "nvidia-smi" in r["setup_error"]
    assert (out / "report.json").exists()


def test_reject_duplicate_or_unknown_cases(tmp_path):
    with pytest.raises(ValueError, match="unique names"):
        autonomous.run_suite(output_dir=tmp_path / "bad1", cases="NO_BARK,NO_BARK")
    with pytest.raises(ValueError, match="unique names"):
        autonomous.run_suite(output_dir=tmp_path / "bad2", cases="mystery")


def test_cli_nonzero_on_backend_failure(tmp_path):
    status = autonomous.main(["--backend", "sam-gguf", "--cli", "absent", "--model", "absent",
                              "--cases", "NO_BARK", "--output-dir", str(tmp_path)])
    assert status == 1
    assert (tmp_path / "SUMMARY.md").exists()


def test_yamnet_detector_benchmark_failure_report(tmp_path):
    from barkfilter.detector_benchmark import run
    result = run(tmp_path, model_path="not-present.onnx", class_map="not-present.csv")
    assert result["status"] == "error"
    assert "error" in json.loads((tmp_path / "detector_report.json").read_text())
