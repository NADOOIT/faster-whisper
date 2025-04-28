"""
Testet die Beispielskripte im examples/-Ordner auf Importierbarkeit und Fehlerfreiheit (Smoke-Test).

Hinweis: Diese Tests prüfen nicht die Audioausgabe, sondern ob die Skripte ohne Syntaxfehler und offensichtliche Fehler laufen.
"""
import importlib.util
import os
import sys
import pytest

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")

@pytest.mark.parametrize("script", [
    "transcribe_file.py",
    "batch_transcribe.py",
    "cli_demo.py",
])
def test_example_scripts_import(script):
    path = os.path.join(EXAMPLES_DIR, script)
    spec = importlib.util.spec_from_file_location("example", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except SystemExit:
        # Skripte beenden sich ggf. mit sys.exit bei fehlenden Args
        pass
    except Exception as e:
        pytest.fail(f"Script {script} failed with error: {e}")
