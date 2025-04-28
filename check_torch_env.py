import torch
import platform
print("PyTorch-Version:", torch.__version__)
print("Python-Architektur:", platform.machine())
print("MPS verfügbar:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
print("CUDA verfügbar:", torch.cuda.is_available() if hasattr(torch, "cuda") else False)
