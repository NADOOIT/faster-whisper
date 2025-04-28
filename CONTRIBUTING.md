# 🤝 Contributing Guide

Danke, dass du zu **faster-whisper** beitragen möchtest!

## 1. Projekt einrichten

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## 2. Tests ausführen

```bash
.venv/bin/python -m pytest tests/test_auto_device.py
.venv/bin/python -m pytest tests/test_tokenizer.py tests/test_transcribe.py
```

## 3. Code-Style
- Halte dich an [PEP8](https://peps.python.org/pep-0008/).
- Schreibe **aussagekräftige Docstrings** für alle Funktionen/Klassen.
- Nutze sprechende Variablennamen und Kommentare.

## 4. Pull Requests
- Erstelle vor größeren Änderungen ein Issue oder diskutiere deine Idee.
- Schreibe klare Commit-Nachrichten.
- Teste vor dem PR alle relevanten Tests.
- Nutze die [Issue-Templates](.github/ISSUE_TEMPLATE/) und das [FAQ](FAQ.md) für schnelle Hilfe.

## 5. Fragen & Hilfe
- Lies README, FAQ und BUILD_MPS.md.
- Nutze Issues für Bugs oder Feature-Wünsche.

---

**Wir freuen uns auf deinen Beitrag!**
