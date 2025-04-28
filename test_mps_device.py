from faster_whisper import WhisperModel

print("Initialisiere WhisperModel mit automatischer Hardware-Erkennung...")
model = WhisperModel("tiny")

# Versuche, Device und Compute-Type direkt auszugeben
try:
    device = getattr(model.model, "device", None)
    compute_type = getattr(model.model, "compute_type", None)
    print(f"[INFO] Model loaded on device: {device}")
    print(f"[INFO] Compute type: {compute_type}")
except Exception as e:
    print(f"[ERROR] Konnte device/compute_type nicht auslesen: {e}")

# Optional: Einmal transkribieren (Stichprobe, falls Testfile vorhanden)
# from pathlib import Path
# test_audio = Path("tests/data/jfk.flac")
# if test_audio.exists():
#     segments, info = model.transcribe(str(test_audio))
#     print(f"[INFO] Transkription erfolgreich. Sprache: {info.language}")
# else:
#     print("[INFO] Kein Testaudio gefunden, nur Device-Test durchgeführt.")
