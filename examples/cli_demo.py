"""
Beispiel: Einfache CLI für Whisper-Transkription mit Geräteauswahl

Vorbereitung:
- venv aktivieren, Abhängigkeiten installieren (siehe README)
- Audiodatei im Projektverzeichnis

Ausführen:
    .venv/bin/python examples/cli_demo.py --model tiny --device auto --file audio.wav

Optionen:
    --model   Modellgröße (tiny, base, small, medium, large)
    --device  Device (auto, cpu, mps, cuda)
    --file    Audiodatei
"""
import argparse
from faster_whisper import WhisperModel

parser = argparse.ArgumentParser(description="Whisper Transkription CLI")
parser.add_argument("--model", default="tiny", help="Modellgröße (tiny, base, small, medium, large)")
parser.add_argument("--device", default="auto", help="Device (auto, cpu, mps, cuda)")
parser.add_argument("--file", required=True, help="Pfad zur Audiodatei")
args = parser.parse_args()

model = WhisperModel(args.model, device=args.device)
segments, info = model.transcribe(args.file)

print(f"Detected language: {info.language}\n")
for segment in segments:
    print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")
