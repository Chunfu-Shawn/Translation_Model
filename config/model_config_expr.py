from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

__author__ = "Chunfu Xiao"
__version__="1.1.0"
__email__ = "chunfushawn@gmail.com"


# -------------------------
# Configuration dataclass
# -------------------------

@dataclass
class ModelConfig:
    """
    Simple dataclass capturing model hyperparameters.
    Fields are intentionally aligned with TranslationBaseModel.__init__ arguments.
    """
    d_seq: int
    d_count: int
    d_model: int
    d_expr: int
    d_cell_env: int = 32
    expr_dict_path: Optional[str] = None
    n_heads: int = 8
    number_of_layers: int = 6
    d_ff: int = 2048
    adaptive_dim: int = 32
    gamma_scale: float = 0.5
    p_drop: float = 0.1

    # optional metadata
    model_name: Optional[str] = None
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)