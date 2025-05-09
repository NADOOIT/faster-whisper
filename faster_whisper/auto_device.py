import platform
import os
import sys

def detect_best_device_and_type(prefer_gpu=True):
    """
    Erkennt automatisch das beste verfügbare Device (CUDA, MPS, CPU) und passenden compute_type.

    - CUDA: NVIDIA-GPU (float16)
    - MPS: Apple Silicon GPU (float16)
    - CPU: Fallback (float32)
    Gibt ein Tupel (device, compute_type) zurück, z.B. ("mps", "float16").
    """
    """
    Automatically detect the best device ('cuda', 'cpu', 'mps') and compute_type ('float16', 'float32', ...)
    for optimal performance, supporting CUDA (NVIDIA), MPS (Apple Silicon), and fallback to CPU.
    Returns: (device, compute_type)
    """
    import platform
    import sys
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
                return 'mps', 'float16'  # float16 ist für MPS unterstützt
        except ImportError:
            pass
        # Fallback: CPU auf Apple Silicon
        if platform.machine() in ('arm64', 'aarch64'):
            return 'cpu', 'float32'  # float32 ist für ARM-CPU sicher

    # CPU fallback (x86, ARM, etc.)
    return 'cpu', 'float32'  # float32 ist universell unterstützt


# Optional: CLI helper
if __name__ == "__main__":
    device, compute_type = detect_best_device_and_type()
    print(f"Best device: {device}, compute_type: {compute_type}")
