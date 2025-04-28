"""
Beispiel: Stapelverarbeitung mehrerer Audiodateien mit schneller Whisper-Transkription

Vorbereitung:
- venv aktivieren und Abhängigkeiten installieren (siehe README)
- Audiodateien im Projektverzeichnis (z.B. audio1.wav, audio2.wav)

Ausführen:
    .venv/bin/python examples/batch_transcribe.py audio1.wav audio2.wav
"""
import sys
from faster_whisper import WhisperModel

if len(sys.argv) < 2:
    print("Usage: python batch_transcribe.py <audiofile1> <audiofile2> ...")
    sys.exit(1)

model = WhisperModel("tiny")

for audiofile in sys.argv[1:]:
    print(f"\n--- Transkription für: {audiofile} ---")
    segments, info = model.transcribe(audiofile)
    print(f"Detected language: {info.language}\n")
    for segment in segments:
        print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")
