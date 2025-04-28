import sys

try:
    import torch
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
except ImportError:
    mps_available = False

print(f"PyTorch MPS verfügbar: {mps_available}")

from faster_whisper import WhisperModel

def test_model(device=None, compute_type=None):
    try:
        print(f"\nInitialisiere WhisperModel mit device={device}, compute_type={compute_type} ...")
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        if compute_type is not None:
            kwargs["compute_type"] = compute_type
        model = WhisperModel("tiny", **kwargs)
        used_device = getattr(model.model, "device", "unbekannt")
        used_compute_type = getattr(model.model, "compute_type", "unbekannt")
        print(f"[OK] Modell geladen auf Device: {used_device}, Compute-Type: {used_compute_type}")
    except Exception as e:
        print(f"[FEHLER] {e}")

# Test 1: Automatische Auswahl
test_model()
# Test 2: Explizit MPS + float16
test_model(device="mps", compute_type="float16")
# Test 3: Explizit CPU + float32
test_model(device="cpu", compute_type="float32")
# Test 4: Explizit CUDA + float16
test_model(device="cuda", compute_type="float16")
