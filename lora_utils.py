import torch
import torch.nn as nn
import loralib as lora
from typing import Dict, List, Tuple, Optional, Type
from utils import unwrap_model


# --------------------------------------------
# Safe replacement: replace only nn.Linear nodes
# --------------------------------------------

def replace_linear_with_lora(
    module: nn.Module,
    r: int = 4,
    lora_alpha: float = 16.0,
    target_types: Tuple[Type[nn.Module], ...] = (nn.Linear,),
) -> None:
    """
    Recursively replace nn.Linear modules inside `module` with loralib.Linear while copying
    original base weights and biases.

    Parameters
    ----------
    module : nn.Module
        The module to traverse and modify in-place.
    r : int
        LoRA rank (low-rank dimension).
    lora_alpha : float
        LoRA scaling alpha (scaling applied as alpha / r).
    target_types : tuple of torch.nn.Module types
        Types to treat as "linear-like" and replace. Default only replaces nn.Linear.
        If you have custom linear layers, include their classes in this tuple.
    """
    for name, child in list(module.named_children()):
        # Recursively process children first
        replace_linear_with_lora(child, r=r, lora_alpha=lora_alpha, target_types=target_types)

        # If child is one of the target types (e.g. nn.Linear), replace it
        if isinstance(child, target_types):
            # construct a loralib.Linear with same shape and bias presence
            new_linear = lora.Linear(
                in_features=child.in_features,
                out_features=child.out_features,
                r=r,
                lora_alpha=lora_alpha,
                bias=(child.bias is not None),
            )

            # copy base weight and bias into the new module's base parameters
            with torch.no_grad():
                # many lora implementations use `.weight` and `.bias` for base params
                new_linear.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new_linear.bias.data.copy_(child.bias.data)

            # replace the module in-place
            setattr(module, name, new_linear)

def build_lora_model_from_pretrained(base_model, r=4, lora_alpha=16):
    """
    assume base_model is a pretrained model containing nn.Linear
    1) clone or replace module as lora.Linear (copy base weights)
    2) return replaced model
    """
    replace_linear_with_lora(base_model, r=r, lora_alpha=lora_alpha)
    return base_model


# ------------------------------------------
# Checking utilities: list modules & params
# ------------------------------------------
def check_lora_replacement(model: nn.Module, verbose: bool = True) -> Dict[str, object]:
    """
    Inspect the model to report:
      - how many loralib.Linear modules exist
      - how many nn.Linear modules remain (i.e. not replaced)
      - list modules with 2D weight matrices (helpful to find custom linear-like layers)
      - optional: parameter names that contain 'lora'
    Returns a dict summary.
    """
    model = unwrap_model(model)
    lora_modules: List[Tuple[str, nn.Module]] = []
    linear_modules: List[Tuple[str, nn.Module]] = []
    weight_like: List[Tuple[str, nn.Module]] = []

    for name, module in model.named_modules():
        if isinstance(module, lora.Linear):
            lora_modules.append((name, module))
        elif isinstance(module, nn.Linear):
            linear_modules.append((name, module))
        else:
            # detect modules that expose a 2D weight matrix but are not nn.Linear (e.g., embeddings or custom linears)
            w = getattr(module, "weight", None)
            if isinstance(w, torch.Tensor) and getattr(w, "dim", lambda: None)() == 2:
                # exclude plain Embedding from the catch-all list (we report embeddings separately when needed)
                if not isinstance(module, nn.Embedding):
                    weight_like.append((name, module))

    lora_param_names = [n for n, _ in model.named_parameters() if "lora" in n.lower()]

    summary = {
        "total_lora_linear": len(lora_modules),
        "total_nn_linear_remaining": len(linear_modules),
        "lora_modules": lora_modules,
        "nn_linear_modules": linear_modules,
        "weight_like_modules": weight_like,
        "lora_param_names": lora_param_names,
    }

    if verbose:
        print("===== LoRA Replacement Check =====")
        print(f"loralib.Linear modules found : {summary['total_lora_linear']}")
        print(f"nn.Linear modules remaining  : {summary['total_nn_linear_remaining']}")
        if linear_modules:
            print("Examples of remaining nn.Linear modules (name, type):")
            for name, mod in linear_modules[:30]:
                print("  -", name, type(mod))
        if lora_modules:
            print("Examples of loralib.Linear modules (name):")
            for name, _ in lora_modules[:30]:
                print("  -", name)
        if weight_like:
            print("\nModules with a 2-D weight matrix (not nn.Linear):")
            for name, mod in weight_like[:30]:
                print("  -", name, type(mod))
        print(f"\nTotal parameters with 'lora' in name: {len(summary['lora_param_names'])}")
        if summary["lora_param_names"]:
            for n in summary["lora_param_names"][:50]:
                print("  -", n)
        print("==================================")

    return summary


def assert_all_replaced(model: nn.Module) -> None:
    """
    Assert that no nn.Linear modules remain in the model. Raises AssertionError if any are found.
    """
    summary = check_lora_replacement(model, verbose=False)
    remaining = summary["total_nn_linear_remaining"]
    if remaining > 0:
        first = summary["nn_linear_modules"][0][0]
        raise AssertionError(f"Found {remaining} nn.Linear modules that were NOT replaced. Example: {first}")
    print("✅ All nn.Linear modules have been replaced (or none existed).")


def print_param_counts(model: nn.Module) -> None:
    """
    Print total parameter count and number of trainable parameters.
    Useful to verify replacement and trainability settings after freezing/unfreezing.
    """
    model = unwrap_model(model)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: trainable {trainable:,} / total {total:,} ({100.0 * trainable / total:.2f}% trainable)")


def set_trainable_base_and_lora(model: nn.Module, train_base: bool, train_lora: bool):
    """
    set requires_grad for each module in model
    for loralib.Linear module: non-lora parameter (like weight, bias) was regarded as base;
                               'lora' subparameters was regarded as lora。
    for other modules: all parameters are regarded as base。
    """
    for module in model.modules():
        # get only parameters that belong exactly to this module (no recursion)
        for pname, p in module.named_parameters(recurse=False):
            is_lora_param = 'lora' in pname.lower()  # e.g. 'lora_A', 'lora_B', 'lora_alpha'
            if isinstance(module, lora.Linear) or isinstance(module, lora.Embedding):
                # within lora.Linear, toggle according to flag
                p.requires_grad = train_lora if is_lora_param else train_base
            else:
                # non-lora module: labeled as base
                p.requires_grad = train_base
    