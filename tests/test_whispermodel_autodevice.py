import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from faster_whisper import WhisperModel

class DummyModel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.is_multilingual = True  # Für Tokenizer-Auswahl im WhisperModel

def test_auto_device_and_compute_type(monkeypatch):
    # Patch ctranslate2.models.Whisper to capture arguments
    import faster_whisper.transcribe as transcribe_mod
    monkeypatch.setattr(transcribe_mod, "ctranslate2", type("ctranslate2", (), {"models": type("models", (), {"Whisper": DummyModel})}))
    monkeypatch.setattr(transcribe_mod, "download_model", lambda *a, **kw: "dummy_path")
    monkeypatch.setattr(transcribe_mod, "get_logger", lambda: type("Logger", (), {"info": lambda self, msg: None})())

    # Patch detect_best_device_and_type, damit Automatik garantiert wird
    import faster_whisper.auto_device as auto_device_mod
    monkeypatch.setattr(auto_device_mod, "detect_best_device_and_type", lambda prefer_gpu=True: ("cpu", "int8"))

    # device and compute_type default (should auto-detect)
    model = WhisperModel("large-v3")
    # device and compute_type should be set in model.model.kwargs
    assert "device" in model.model.kwargs
    assert "compute_type" in model.model.kwargs
    assert model.model.kwargs["device"] == "cpu"
    assert model.model.kwargs["compute_type"] == "int8"

    # device explicit, compute_type default (should only auto-detect compute_type)
    model = WhisperModel("large-v3", device="cpu")
    assert model.model.kwargs["device"] == "cpu"
    assert model.model.kwargs["compute_type"] == "int8"

    # device and compute_type explicit (should take as is)
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")
    assert model.model.kwargs["device"] == "cpu"
    assert model.model.kwargs["compute_type"] == "int8"
