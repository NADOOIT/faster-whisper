# 📂 Beispiele für faster-whisper

Hier findest du praktische Beispielskripte für den Einstieg und eigene Experimente.

## Übersicht

- **transcribe_file.py**: Transkribiert eine einzelne Audiodatei mit automatischer Device-Erkennung.
- **batch_transcribe.py**: Stapelverarbeitung beliebig vieler Audiodateien.
- **cli_demo.py**: Kommandozeilen-Interface mit Modell- und Device-Auswahl.
- **print_devices.py**: Zeigt alle verfügbaren Devices für CTranslate2/Whisper an.
  - Beispiel: `.venv/bin/python examples/print_devices.py`
- **print_model_info.py**: Zeigt Infos zu einem Whisper-Modell (Name, Sprachen, Pfad).
  - Beispiel: `.venv/bin/python examples/print_model_info.py --model tiny`

## Nutzung

1. Virtuelle Umgebung aktivieren und Abhängigkeiten installieren:
   ```bash
   uv venv .venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

2. Beispiel ausführen, z.B.:
   ```bash
   .venv/bin/python examples/transcribe_file.py audio.wav
   .venv/bin/python examples/batch_transcribe.py audio1.wav audio2.wav
   .venv/bin/python examples/cli_demo.py --model tiny --device mps --file audio.wav
   ```

## Hinweise
- Die Beispiele funktionieren mit allen unterstützten Devices (CPU, MPS, CUDA).
- Für optimale Geschwindigkeit auf Apple Silicon siehe [../CTranslate2/BUILD_MPS.md](../CTranslate2/BUILD_MPS.md).
- Weitere Hilfe: [../FAQ.md](../FAQ.md)

---

Viel Spaß beim Ausprobieren!
