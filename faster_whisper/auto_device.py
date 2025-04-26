import platform
import os
import sys

def detect_best_device_and_type(prefer_gpu=True):
    """
    Automatically detect the best device ('cuda', 'cpu', 'mps') and compute_type ('float16', 'int8', ...)
    for optimal performance, supporting CUDA (NVIDIA), MPS (Apple Silicon), and fallback to CPU.
    Returns: (device, compute_type)
    """
    # CUDA (NVIDIA GPU)
    try:
        import torch
        if prefer_gpu and torch.cuda.is_available():
            return 'cuda', 'float16'
    except ImportError:
        pass

    # Apple Silicon (M1/M2/M3) - MPS
    if sys.platform == 'darwin':
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps', 'float16'
        except ImportError:
            pass
        # Fallback: Use int8 on CPU for Apple Silicon if MPS not available
        if platform.machine() in ('arm64', 'aarch64'):
            return 'cpu', 'int8'

    # CPU fallback (x86, ARM, etc.)
    return 'cpu', 'int8'

# Optional: CLI helper
if __name__ == "__main__":
    device, compute_type = detect_best_device_and_type()
    print(f"Best device: {device}, compute_type: {compute_type}")
