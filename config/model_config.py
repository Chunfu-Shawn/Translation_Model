from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

__author__ = "Chunfu Xiao"
__version__="1.0.0"
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
    pmt_len: int
    all_cell_types: List[str]
    n_heads: int = 8
    number_of_layers: int = 6
    d_ff: int = 2048
    p_drop: float = 0.1
    # optional metadata
    model_name: Optional[str] = None
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)