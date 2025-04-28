from faster_whisper import WhisperModel

print("[TEST] Initialisiere WhisperModel mit device='mps', compute_type='float16' ...")
try:
    model = WhisperModel("tiny", device="mps", compute_type="float16")
    used_device = getattr(model.model, "device", None)
    used_compute_type = getattr(model.model, "compute_type", None)
    if used_device == "mps":
        print(f"[OK] Modell läuft auf MPS (Apple GPU)! Device: {used_device}, Compute-Type: {used_compute_type}")
    else:
        print(f"[FEHLER] Modell läuft NICHT auf MPS! Device: {used_device}, Compute-Type: {used_compute_type}")
except Exception as e:
    print(f"[FEHLER] Initialisierung auf MPS nicht möglich: {e}")
