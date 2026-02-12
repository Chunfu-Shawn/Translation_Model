import torch
import torch.nn as nn
from typing import Optional

__author__ = "Chunfu Xiao"
__version__="1.0.0"
__email__ = "chunfushawn@gmail.com"

class MaskedSeqPredictorHead(nn.Module):
    def __init__(self, pmt_len, d_model, d_seq, d_pred_h, p_drop=0.1):
        super().__init__()
        self.pmt_len = pmt_len
        # class output shape: (bs, seq_len, d_seq)
        self.classify_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_pred_h),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_pred_h, d_model),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_model, d_seq)
        )
        
        # initialize parameters (Xavier for weight tensors by default)
        self.init_params()

    def init_params(self, default_initialization=False):
        """
        Initialize parameters. By default uses Xavier uniform for parameters that have dim > 1.
        If `default_initialization` is True, skip custom initialization (use PyTorch defaults).
        """
        if default_initialization:
            return
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_reps):
        # src_reps: [bs, pmt_len + seq_len, d_model], src_mask: [bs, pmt_len + seq_len]
        src_reps = src_reps[:, self.pmt_len:, :] # -> [bs, seq_len, d_model]

        return self.classify_mlp(src_reps) # [bs, seq_len, d_seq]

class PsitePredictorHead(nn.Module):
    """
    PsitePredictorHead

    - If some dims are not provided, you can create this head from a `parent_model` using
      `PsitePredictorHead.create_from_model(parent_model, **overrides)`.
    - Default hidden size d_pred_h is set to max(64, d_model // 2) if not provided.
    - Applies LayerNorm -> Linear -> GELU -> Dropout blocks and ends with ReLU to produce non-negative counts.
    """

    def __init__(
        self,
        pmt_len: int,
        d_model: int,
        d_count: int,
        d_pred_h: Optional[int] = None,
        p_drop: float = 0.1,
        init_xavier: bool = True,
    ):
        """
        Parameters
        ----------
        pmt_len : int
            Prompt length (number of prompt tokens inserted at start of sequence).
        d_model : int
            Model embedding dimension.
        d_count : int
            Output count dimension per token.
        d_pred_h : Optional[int]
            Hidden dimension for MLP. If None choose a reasonable default.
        p_drop : float
            Dropout probability.
        init_xavier : bool
            Whether to run Xavier initialization on linear weights.
        """
        super().__init__()
        self.name = "PsiteDensityHead"
        self.pmt_len = int(pmt_len)
        self.d_model = int(d_model)
        self.d_count = int(d_count)
        self.d_pred_h = int(d_pred_h) if d_pred_h is not None else max(64, self.d_model // 2)
        self.p_drop = float(p_drop)

        # Build MLP: LayerNorm -> Linear(d_model -> d_pred_h) -> GELU -> Dropout -> Linear(d_pred_h -> d_model) -> GELU -> Dropout -> Linear(d_model -> d_count) -> ReLU
        self.net = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_pred_h),
            nn.GELU(),
            nn.Dropout(self.p_drop),
            nn.Linear(self.d_pred_h, self.d_count),
            nn.ReLU()
        )

        if init_xavier:
            self._init_parameters()

    def _init_parameters(self):
        """Initialize linear weights with Xavier and biases to zero; LayerNorm weight=1 bias=0."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, src_reps: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        src_reps : torch.Tensor
            Encoder representations with shape (bs, pmt_len + seq_len, d_model).
            This method will slice away the prompt prefix and predict counts for the remaining tokens.

        Returns
        -------
        torch.Tensor
            Predicted counts with shape (bs, seq_len, d_count).
        """
        # slice away prompt tokens at the beginning
        # if src_reps is (bs, pmt_len + seq_len, d_model)
        x = src_reps[:, self.pmt_len:, :]  # -> (bs, seq_len, d_model)
        out = self.net(x)         # -> (bs, seq_len, d_count)

        if pad_mask is not None:
            # pad_mask might be bool; convert to float and move to same device/dtype
            m = pad_mask[:, self.pmt_len:].to(x.device).unsqueeze(-1)  # (bs,L,1)
            m = m.to(dtype=x.dtype)
            # zero-out padded outputs so they don't leak into next layer
            out = out * m
            
        return out

    # ---------------------------
    # Factory / helper functions
    # ---------------------------
    @classmethod
    def create_from_model(
        cls,
        parent_model: object,
        pmt_len: Optional[int] = None,
        d_model: Optional[int] = None,
        d_count: Optional[int] = None,
        d_pred_h: Optional[int] = None,
        p_drop: float = 0.1,
        init_xavier: bool = True,
    ) -> "PsitePredictorHead":
        """
        Create a PsitePredictorHead by inferring missing dimensions from `parent_model`.

        parent_model is expected to expose attributes:
          - pmt_len (prompt length) or "pmt_len"
          - d_model (model hidden size)
          - d_count (count feature dim) OR you can pass explicit d_count

        Parameters: the same as __init__, but any None will be inferred where possible.
        """
        # infer pmt_len
        if pmt_len is None:
            if hasattr(parent_model, "pmt_len"):
                pmt_len = int(getattr(parent_model, "pmt_len"))
            else:
                raise ValueError("pmt_len not provided and parent_model has no attribute 'pmt_len'")

        # infer d_model
        if d_model is None:
            if hasattr(parent_model, "d_model"):
                d_model = int(getattr(parent_model, "d_model"))
            else:
                raise ValueError("d_model not provided and parent_model has no attribute 'd_model'")

        # infer d_count
        if d_count is None:
            # try common attribute names used in your codebase
            if hasattr(parent_model, "d_count"):
                d_count = int(getattr(parent_model, "d_count"))
            elif hasattr(parent_model, "src_emb") and hasattr(parent_model.src_emb, "d_count"):
                d_count = int(getattr(parent_model.src_emb, "d_count"))
            else:
                raise ValueError("d_count not provided and cannot be inferred from parent_model")

        # now construct
        return cls(pmt_len=pmt_len, d_model=d_model, d_count=d_count, d_pred_h=d_pred_h, p_drop=p_drop, init_xavier=init_xavier)


class PsiteDensityHead(nn.Module):
    """
    PsiteDensityHead

    - If some dims are not provided, you can create this head from a `parent_model` using
      `PsiteDensityHead.create_from_model(parent_model, **overrides)`.
    - Default hidden size d_pred_h is set to max(64, d_model // 2) if not provided.
    - Applies LayerNorm -> Linear -> GELU -> Dropout blocks and ends with ReLU to produce non-negative counts.
    """

    def __init__(
        self,
        d_model: int,
        d_count: int,
        d_pred_h: Optional[int] = None,
        p_drop: float = 0.1,
        init_xavier: bool = True,
    ):
        """
        Parameters
        ----------
        d_model : int
            Model embedding dimension.
        d_count : int
            Output count dimension per token.
        d_pred_h : Optional[int]
            Hidden dimension for MLP. If None choose a reasonable default.
        p_drop : float
            Dropout probability.
        init_xavier : bool
            Whether to run Xavier initialization on linear weights.
        """
        super().__init__()
        self.name = "PsiteDensityHead"
        self.d_model = int(d_model)
        self.d_count = int(d_count)
        self.d_pred_h = int(d_pred_h) if d_pred_h is not None else max(64, self.d_model // 2)
        self.p_drop = float(p_drop)

        # Build MLP: LayerNorm -> Linear(d_model -> d_pred_h) -> GELU -> Dropout -> Linear(d_pred_h -> d_model) -> GELU -> Dropout -> Linear(d_model -> d_count) -> ReLU
        self.net = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_pred_h),
            nn.GELU(),
            nn.Dropout(self.p_drop),
            nn.Linear(self.d_pred_h, self.d_count),
            nn.ReLU()
        )

        if init_xavier:
            self._init_parameters()

    def _init_parameters(self):
        """Initialize linear weights with Xavier and biases to zero; LayerNorm weight=1 bias=0."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, src_reps: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        src_reps : torch.Tensor
            Encoder representations with shape (bs, pmt_len + seq_len, d_model).
            This method will slice away the prompt prefix and predict counts for the remaining tokens.

        Returns
        -------
        torch.Tensor
            Predicted counts with shape (bs, seq_len, d_count).
        """
        # slice away prompt tokens at the beginning
        out = self.net(src_reps)  # -> (bs, seq_len, d_count)

        if pad_mask is not None:
            # pad_mask might be bool; convert to float and move to same device/dtype
            m = pad_mask.to(src_reps.device).unsqueeze(-1)  # (bs,L,1)
            m = m.to(dtype=src_reps.dtype)
            # zero-out padded outputs so they don't leak into next layer
            out = out * m
            
        return out

    # ---------------------------
    # Factory / helper functions
    # ---------------------------
    @classmethod
    def create_from_model(
        cls,
        parent_model: object,
        d_model: Optional[int] = None,
        d_count: Optional[int] = None,
        d_pred_h: Optional[int] = None,
        p_drop: float = 0.1,
        init_xavier: bool = True,
    ) -> "PsiteDensityHead":
        """
        Create a PsiteDensityHead by inferring missing dimensions from `parent_model`.

        parent_model is expected to expose attributes:
          - d_model (model hidden size)
          - d_count (count feature dim) OR you can pass explicit d_count

        Parameters: the same as __init__, but any None will be inferred where possible.
        """

        # infer d_model
        if d_model is None:
            if hasattr(parent_model, "d_model"):
                d_model = int(getattr(parent_model, "d_model"))
            else:
                raise ValueError("d_model not provided and parent_model has no attribute 'd_model'")

        # infer d_count
        if d_count is None:
            # try common attribute names used in your codebase
            if hasattr(parent_model, "d_count"):
                d_count = int(getattr(parent_model, "d_count"))
            elif hasattr(parent_model, "src_emb") and hasattr(parent_model.src_emb, "d_count"):
                d_count = int(getattr(parent_model.src_emb, "d_count"))
            else:
                raise ValueError("d_count not provided and cannot be inferred from parent_model")

        # now construct
        return cls(d_model=d_model, d_count=d_count, d_pred_h=d_pred_h, p_drop=p_drop, init_xavier=init_xavier)
    

class CellClassificationHead(nn.Module):
    """
    Reduce prompt tokens by scalar-weighted pooling and classify into num_cells categories.
    - Uses (num_cells) logits (no softmax).
    - This head is compact: uses mean pooling -> linear -> optional hidden -> output.
    """
    def __init__(self, 
                 pmt_len: int, 
                 d_model: int, 
                 num_cells: int, 
                 d_pred_h: int = None,
                 p_drop: float = 0.1,
                 init_xavier: bool = True,):
        super().__init__()
        self.name = "MaskedCellTypeHead"
        self.pmt_len = int(pmt_len)
        self.d_model = int(d_model)
        self.num_cells = int(num_cells)
        self.d_pred_h = int(d_pred_h) if d_pred_h is not None else max(64, self.d_model // 2)
        self.score = nn.Linear(d_model, 1)
        self.p_drop = float(p_drop)

        self.net = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_pred_h),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(self.d_pred_h, self.num_cells)  # logits
        )

        if init_xavier:
            self._init_parameters()
    
    def _init_parameters(self):
        """Initialize linear weights with Xavier and biases to zero; LayerNorm weight=1 bias=0."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, src_reps: torch.Tensor, src_mask: torch.Tensor = None, return_weights = False):
        """
        src_reps: (bs, pmt_len + seq_len, d_model)
        src_mask: optional bool mask (bs, pmt_len + seq_len) -> used to weigh prompt tokens if provided.
        returns logits: (bs, num_cells) with scalar-weight pooling
        """
        prompt_reps = src_reps[:, :self.pmt_len, :]    # (bs, pmt_len, d_model)
        scores = self.score(prompt_reps).squeeze(-1)          # (bs, pmt_len)

        if src_mask is not None:
            # mask for prompt positions (True=valid)
            prompt_mask = src_mask[:, :self.pmt_len]
            scores = scores.masked_fill(~prompt_mask, float('-inf'))

        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)   # (bs, pmt_len, 1)
        pooled = (weights * prompt_reps).sum(dim=1)             # (bs, d)
        logits = self.net(pooled)
        if return_weights:
            return logits, weights.squeeze(-1)                # weights (bs, pmt_len)
        return logits
    
    # ---------------------------
    # Factory / helper functions
    # ---------------------------
    @classmethod
    def create_from_model(
        cls,
        parent_model: object,
        pmt_len: Optional[int] = None,
        d_model: Optional[int] = None,
        num_cells: Optional[int] = None,
        d_pred_h: Optional[int] = None,
        p_drop: float = 0.1,
        init_xavier: bool = True,
    ) -> "CellClassificationHead":
        """
        Create a CellClassificationHead by inferring missing dimensions from `parent_model`.

        parent_model is expected to expose attributes:
          - pmt_len (prompt length) or "pmt_len"
          - d_model (model hidden size)
          - num_cells (number of cell types) OR you can pass explicit num_cells

        Parameters: the same as __init__, but any None will be inferred where possible.
        """
        # infer pmt_len
        if pmt_len is None:
            if hasattr(parent_model, "pmt_len"):
                pmt_len = int(getattr(parent_model, "pmt_len"))
            else:
                raise ValueError("pmt_len not provided and parent_model has no attribute 'pmt_len'")

        # infer d_model
        if d_model is None:
            if hasattr(parent_model, "d_model"):
                d_model = int(getattr(parent_model, "d_model"))
            else:
                raise ValueError("d_model not provided and parent_model has no attribute 'd_model'")

        # infer num_cells
        if num_cells is None:
            # try common attribute names used in your codebase
            if hasattr(parent_model, "num_cells"):
                num_cells = int(getattr(parent_model, "num_cells"))
            else:
                raise ValueError("num_cells not provided and cannot be inferred from parent_model")

        # now construct
        return cls(pmt_len=pmt_len, d_model=d_model, num_cells=num_cells, d_pred_h=d_pred_h, p_drop=p_drop, init_xavier=init_xavier)