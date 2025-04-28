# ❓ FAQ – Häufige Fragen

## Wie installiere ich die Abhängigkeiten?
- Nutze immer `uv`:
  ```bash
  uv venv .venv
  source .venv/bin/activate
  uv pip install -r requirements.txt
  ```

## Wie kann ich prüfen, ob MPS/CUDA unterstützt wird?
- Nach der Installation:
  ```python
  import ctranslate2; print(ctranslate2.list_supported_devices())
  ```
- Erwartung: `['cpu', 'mps']` (Apple Silicon) oder `['cpu', 'cuda']` (NVIDIA)

## Warum schlägt der Import von ctranslate2 fehl?
- Prüfe, ob du das richtige venv aktiviert hast.
- Installiere das Wheel ggf. neu:
  ```bash
  uv pip install /pfad/zu/ctranslate2-<version>-cp*-macosx_*.whl --force-reinstall
  ```

## Wie führe ich die Tests aus?
- Device-Detection-Tests:
  ```bash
  .venv/bin/python -m pytest tests/test_auto_device.py
  ```
- Alle anderen Tests:
  ```bash
  .venv/bin/python -m pytest tests/test_tokenizer.py tests/test_transcribe.py
  ```

## Wie kann ich die Performance optimieren?
- Nutze das beste Device (`cuda`, `mps`, `cpu`).
- Stelle sicher, dass CTranslate2 mit MPS/CUDA gebaut wurde.
- Für große Audiodateien: Mehr Threads (`cpu_threads`), mehrere Worker (`num_workers`).

## Wie aktiviere ich Debug-Logging?
- Setze die Umgebungsvariable `FASTER_WHISPER_DEBUG=1`, um detaillierte Debug-Ausgaben zu erhalten (Device-Auswahl, Modellpfad, Fallbacks etc.):
  ```bash
  FASTER_WHISPER_DEBUG=1 .venv/bin/python examples/transcribe_file.py audio.wav
  ```
- Das hilft bei der Fehlersuche und Hardware-Diagnose.

## Wo finde ich Hilfe bei Problemen?
- Lies README, FAQ und [CTranslate2/BUILD_MPS.md](./CTranslate2/BUILD_MPS.md).
- Erstelle ein Issue mit möglichst vielen Infos (siehe Bug-Template).

---

**Noch Fragen? Einfach ein Issue erstellen oder ins README schauen!**
