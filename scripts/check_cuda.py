"""Actual local-container CUDA arithmetic check; no host configuration changes."""

import json

import torch

if not torch.cuda.is_available():
    raise RuntimeError(f"CUDA unavailable in installed Torch {torch.__version__}")
x = torch.arange(1024, dtype=torch.float64, device="cuda")
actual = (x * x).sum().item()
expected = sum(i * i for i in range(1024))
if actual != expected:
    raise ValueError(f"CUDA arithmetic mismatch: {actual} versus {expected}")
print(
    json.dumps(
        {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "float64_sum_of_squares": actual,
            "passed": True,
        }
    )
)
