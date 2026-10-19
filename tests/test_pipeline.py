import numpy as np
import pytest
from barkfilter.pipeline import process_audio, plan_windows
from barkfilter.detection import Interval, merge_intervals
from barkfilter.engines import IdentitySeparator


class EventDetector:
    def __init__(self, spans): self.spans=spans
    def detect(self, x, sr): return self.spans


class Gain:
    def __init__(self, gain): self.gain=gain;self.calls=0
    def separate(self, x, sr): self.calls+=1;return x*self.gain


def audio(n=44100, channels=2):
    rng=np.random.default_rng(5)
    return rng.uniform(-.2,.2,size=(n,channels)).astype(np.float32)


def test_identity_overlap_roundtrip():
    x=audio(48001,2)
    r=process_audio(x,8000,IdentitySeparator(),window_seconds=1.2,
                    overlap_seconds=.4,mode='continuous')
    assert r.audio.shape == x.shape
    assert np.max(abs(r.audio-x)) < 2e-7
    assert len(r.trace)>3


def test_hybrid_no_barks_does_not_run_model_or_change_audio():
    x=audio()
    engine=Gain(.0)
    r=process_audio(x,8000,engine,detector=EventDetector([]),mode='hybrid')
    assert engine.calls == 0
    assert np.array_equal(r.audio,x)


def test_hybrid_selective_calls_and_crossfade():
    x=audio(n=48000,channels=1)
    eng=Gain(.0)
    r=process_audio(x,8000,eng,detector=EventDetector([Interval(1,1.5)]),
                    window_seconds=2.0,overlap_seconds=.5,mode='hybrid',event_margin_seconds=0)
    assert 0 < eng.calls < len(r.trace)
    assert np.isfinite(r.audio).all()
    assert len(r.audio)==len(x)


def test_continuous_bypasses_detector():
    x=audio(8000,1)
    eng=Gain(.2)
    r=process_audio(x,8000,eng,mode='continuous',window_seconds=2,overlap_seconds=.5)
    assert eng.calls==1
    assert np.allclose(r.audio, .2*x)


def test_invalid_windows():
    with pytest.raises(ValueError): plan_windows(5,3,0)
    with pytest.raises(ValueError): process_audio(audio(), 8000, Gain(1),
                                                   window_seconds=1, overlap_seconds=1)


def test_no_silent_fallback_on_backend_error():
    class Bad:
        def separate(self,x,sr): raise RuntimeError('bad model')
    with pytest.raises(RuntimeError,match='bad model'):
        process_audio(audio(), 8000, Bad(),mode='continuous')


def test_duration_drift_rejected():
    class Bad:
        def separate(self,x,sr): return x[:-1000]
    with pytest.raises(ValueError,match='duration drift'):
        process_audio(audio(8000,1), 8000,Bad(),mode='continuous')


def test_spans_combination():
    spans=[Interval(3,4),Interval(.2,.5),Interval(.4,.8)]
    assert merge_intervals(spans,duration=5,margin=.1)==[Interval(.1,.9),Interval(2.9,4.1)]


def test_input_validation():
    x=audio();x[0,0]=float('nan')
    with pytest.raises(ValueError,match='NaN'):
        process_audio(x,8000,IdentitySeparator(),mode='continuous')
