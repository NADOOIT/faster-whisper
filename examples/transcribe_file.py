"""
Beispiel: Transkribiere eine Audiodatei mit automatischer Device-Erkennung (CPU/MPS/CUDA)

Vorbereitung:
- venv aktivieren und Abhängigkeiten installieren (siehe README)
- Audiodatei (z.B. audio.wav) im Projektverzeichnis

Ausführen:
    .venv/bin/python examples/transcribe_file.py audio.wav
"""
import sys
from faster_whisper import WhisperModel

if len(sys.argv) < 2:
    print("Usage: python transcribe_file.py <audiofile>")
    sys.exit(1)

model = WhisperModel("tiny")
segments, info = model.transcribe(sys.argv[1])

print(f"Detected language: {info.language}\n")
for segment in segments:
    print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")
