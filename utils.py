import os
import sys
import torch.nn as nn

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# -------------------------
# Utilities for DDP wrappers
# -------------------------

def unwrap_model(model: nn.Module) -> nn.Module:
    """
    If model is wrapped in DistributedDataParallel or DataParallel, return the underlying module,
    otherwise return model itself. This is useful because named_modules() on wrappers includes the wrapper.
    """
    # DDP wrappers in torch usually expose `.module`
    return getattr(model, "module", model)
