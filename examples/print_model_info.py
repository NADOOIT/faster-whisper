"""
Beispiel: Zeigt Informationen zu einem Whisper-Modell an (Pfad, Größe, unterstützte Sprachen)

Vorbereitung:
- venv aktivieren, Abhängigkeiten installieren (siehe README)

Ausführen:
    .venv/bin/python examples/print_model_info.py --model tiny
    .venv/bin/python examples/print_model_info.py --model /pfad/zum/modell

Hilfreich für Troubleshooting, Modellwahl und Setup!
"""
import argparse
from faster_whisper import WhisperModel

parser = argparse.ArgumentParser(description="Whisper Modell-Info anzeigen")
parser.add_argument("--model", default="tiny", help="Modellgröße oder Pfad (tiny, base, ... oder lokal)")
args = parser.parse_args()

try:
    model = WhisperModel(args.model)
    print(f"Modell: {args.model}")
    print(f"Unterstützte Sprachen: {getattr(model, 'supported_languages', 'unbekannt')}")
    print(f"Modellpfad: {getattr(model, 'model_path', 'unbekannt')}")
except Exception as e:
    print(f"Fehler beim Laden des Modells: {e}")
