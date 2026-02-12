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

def print_param_counts(model: nn.Module) -> None:
    """
    Print total parameter count and number of trainable parameters.
    Useful to verify replacement and trainability settings after freezing/unfreezing.
    """
    model = unwrap_model(model)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: trainable {trainable:,} / total {total:,} ({100.0 * trainable / total:.2f}% trainable)")
    
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

import torch
import gc

def clean_up_memory():
    if 'dataset' in globals(): del dataset
    if 'saved_data' in globals(): del saved_data
    if 'dataloader' in globals(): del dataloader
    
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("Memory clean up finished.")
