import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pytest
from barkfilter.audio import read_audio,write_audio
from barkfilter.cli import main
from barkfilter.detection import ManualDetector, Interval
from barkfilter.evaluate import evaluate
from barkfilter.pipeline import process_audio
from barkfilter.engines import IdentitySeparator, SamGGUFSeparator


def test_audio_round_trip(tmp_path):
    x=np.sin(np.linspace(0,100,5000,dtype=np.float32))[:,None]*.4
    wav=tmp_path/'test.wav'
    write_audio(wav,x,16000)
    y,sr=read_audio(wav,sample_rate=48000)
    assert y.shape==(15000,1)
    assert sr==48000


def test_manual_intervals_and_cli(tmp_path,capsys):
    x=np.ones((5000,1),np.float32)*.2
    inp=tmp_path/'in.wav';out=tmp_path/'out.wav'
    write_audio(inp,x,1000)
    # CLI's 48 kHz internal processing is intentional.
    (tmp_path/'spans.json').write_text(json.dumps({'barks':[[.3,.7]]}))
    main([str(inp),'--output',str(out),'--backend','identity','--mode','hybrid',
          '--detector','manual','--spans',str(tmp_path/'spans.json')])
    y,sr=read_audio(out)
    assert sr==48000
    assert np.max(abs(y[3000:-3000]-.2))<.001  # sinc resampling has edge transients
    assert 'WARNING' in capsys.readouterr().out


def test_no_detector_must_fail_in_hybrid():
    with pytest.raises(ValueError,match='detector'):
        process_audio(np.zeros((100,1),np.float32),16000,IdentitySeparator(),mode='hybrid')


def test_evaluator_not_fake_bark_score():
    clean=np.zeros((1000,1),np.float32)
    original=np.zeros_like(clean);original[200:400]=.3
    result=original.copy();result[200:400]=.15
    value=evaluate(original,result,clean,1000,[Interval(.2,.4)])
    assert 5.9<value['inside_annotated_barks_error_improvement_db']<6.1


def test_audio_cpp_exit_failure_raises(tmp_path):
    model=tmp_path/'fake.gguf';model.write_bytes(b'not a model')
    executable=shutil.which('false')
    if not executable:pytest.skip('false unavailable')
    backend=SamGGUFSeparator(executable,str(model))
    with pytest.raises(RuntimeError,match='audio.cpp failed'):
        backend.separate(np.zeros((500,1),np.float32),16000)


def test_ffmpeg_video_input(tmp_path):
    if not shutil.which('ffmpeg'):pytest.skip('ffmpeg missing')
    x=np.zeros((16000,1),np.float32)
    inp=tmp_path/'in.wav';write_audio(inp,x,16000)
    mp4=tmp_path/'video.mp4'
    cmd=['ffmpeg','-y','-hide_banner','-loglevel','error','-f','lavfi','-i',
         'color=c=black:s=320x240:r=24:d=1','-i',str(inp),'-shortest',
         '-c:v','mpeg4','-c:a','aac',str(mp4)]
    p=subprocess.run(cmd,capture_output=True)
    assert p.returncode==0,p.stderr
    decoded,sr=read_audio(mp4,sample_rate=48000)
    assert sr==48000 and .90<len(decoded)/sr<1.10


def test_yamnet_onnx_detector_with_fake_session(monkeypatch, tmp_path):
    import sys
    import types
    from barkfilter.detection import YAMNetOnnxDetector
    labels = tmp_path/'classes.csv'
    labels.write_text('index,mid,display_name\n0,/a,Speech\n1,/b,Bark\n')
    class FakeSession:
        def __init__(self, *_args, **_kwargs): pass
        def get_inputs(self): return [types.SimpleNamespace(name='waveform')]
        def run(self, *_args):
            scores = np.zeros((5,2),np.float32)
            scores[2,1] = .91
            return [scores]
    monkeypatch.setitem(sys.modules,'onnxruntime', types.SimpleNamespace(InferenceSession=FakeSession))
    det = YAMNetOnnxDetector('unused.onnx',str(labels),threshold=.2)
    spans = det.detect(np.zeros((48000*4,1),np.float32),48000)
    assert spans == [Interval(.96,1.92)]
