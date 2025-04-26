import sys
import platform
import importlib.util
import pytest

from faster_whisper.auto_device import detect_best_device_and_type

def test_detect_best_device_and_type_cpu(monkeypatch):
    # Simuliere kein CUDA, kein MPS
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    # torch nicht installiert oder keine GPU/MPS
    sys_modules_backup = sys.modules.copy()
    sys.modules["torch"] = None
    device, compute_type = detect_best_device_and_type(prefer_gpu=True)
    assert device == "cpu"
    assert compute_type == "int8"
    sys.modules = sys_modules_backup

def test_detect_best_device_and_type_cuda(monkeypatch):
    class DummyTorch:
        @staticmethod
        def cuda():
            class DummyCuda:
                @staticmethod
                def is_available():
                    return True
            return DummyCuda
        cuda = type("cuda", (), {"is_available": staticmethod(lambda: True)})
    sys.modules["torch"] = DummyTorch
    device, compute_type = detect_best_device_and_type(prefer_gpu=True)
    assert device == "cuda"
    assert compute_type == "float16"
    sys.modules.pop("torch", None)

def test_detect_best_device_and_type_mps(monkeypatch):
    # Simuliere macOS ARM mit MPS
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    class DummyMPS:
        @staticmethod
        def is_available():
            return True
    class DummyTorch:
        backends = type("backends", (), {"mps": DummyMPS})
    sys.modules["torch"] = DummyTorch
    device, compute_type = detect_best_device_and_type(prefer_gpu=True)
    assert device == "mps"
    assert compute_type == "float16"
    sys.modules.pop("torch", None)
