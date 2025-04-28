# How to run tests reliably in this project

## Device Detection Tests (Mocked torch)
These tests patch Python internals and must be run in a clean interpreter process. Run them **separately** from the other tests:

```bash
uv pip install pytest
pytest tests/test_auto_device.py
```

## All Other Tests (Model, Tokenizer, Transcribe)
Run these in a separate process, after the device tests:

```bash
pytest tests/test_tokenizer.py tests/test_transcribe.py
```

## Why?
Mocking `sys.modules["torch"]` in device detection tests can break the PyTorch import for subsequent tests in the same process. Deshalb: **Immer getrennt ausführen!**

## Optional: Parallel Testing
Mit `pytest-xdist` kannst du alle Tests parallel und isoliert laufen lassen:

```bash
uv pip install pytest-xdist
pytest -n auto
```

---

**Für CI oder whisperx genügt es, die Model/Tokenizer-Tests laufen zu lassen. Die Device-Detection-Tests sind nur für die interne Logik relevant.**
