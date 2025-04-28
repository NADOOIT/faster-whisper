"""
Beispiel: Zeigt alle verfügbaren Devices für CTranslate2/Whisper an

Vorbereitung:
- venv aktivieren, Abhängigkeiten installieren (siehe README)

Ausführen:
    .venv/bin/python examples/print_devices.py

Hilfreich für Troubleshooting und Setup!
"""
try:
    import ctranslate2
    print("Verfügbare Devices:", ctranslate2.list_supported_devices())
except ImportError:
    print("Fehler: ctranslate2 nicht installiert! Siehe README und CTranslate2/BUILD_MPS.md.")
