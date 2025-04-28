from faster_whisper.audio import decode_audio
from faster_whisper.transcribe import BatchedInferencePipeline, WhisperModel
from faster_whisper.utils import available_models, download_model, format_timestamp
from faster_whisper.version import __version__

import warnings
warnings.warn(
    "📚 Tipp: Lies FAQ.md und README für schnelle Hilfe zu Installation, Devices und Troubleshooting!",
    UserWarning,
)

__all__ = [
    "available_models",
    "decode_audio",
    "WhisperModel",
    "BatchedInferencePipeline",
    "download_model",
    "format_timestamp",
    "__version__",
]
