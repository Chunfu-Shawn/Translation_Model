import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

__author__ = "Chunfu Xiao"
__version__="1.0.0"
__email__ = "chunfushawn@gmail.com"

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
            nn.ReLU() # nn.Softplus(beta=10)
        )

        if init_xavier:
            self._init_parameters()

    def _init_parameters(self):
        """Initialize linear weights with Xavier and biases to zero; LayerNorm weight=1 bias=0."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    if m.out_features == self.d_count: # 最后一层
                        nn.init.constant_(m.bias, 0.1) # 初始化为微小正数防死
                    else:
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
            Encoder representations with shape (bs, seq_len, d_model).
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


class DecoupledCountHead(nn.Module):
    """
    DecoupledCountHead (Dual-Stream Architecture)

    - Decouples the prediction into two streams:
        1. Shape Stream: Predicts relative ribosome distribution (profile) across the sequence.
        2. Scale Stream: Predicts global log-abundance (TE / Total Counts).
    - Fuses them internally to produce the final absolute counts.
    - If some dims are not provided, you can create this head from a `parent_model` using
      `DecoupledCountHead.create_from_model(parent_model, **overrides)`.
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
            Output count dimension per token (e.g., 1 for density, 10 for read lengths).
        d_pred_h : Optional[int]
            Hidden dimension for MLP. If None choose a reasonable default.
        p_drop : float
            Dropout probability.
        init_xavier : bool
            Whether to run Xavier initialization on linear weights.
        """
        super().__init__()
        self.name = "DecoupledCountHead"
        self.d_model = int(d_model)
        self.d_count = int(d_count)
        self.d_pred_h = int(d_pred_h) if d_pred_h is not None else max(64, self.d_model // 2)
        self.p_drop = float(p_drop)
        
        # ---------------------------------------------------
        # Stream 1: Shape (局部相对密度)
        # 负责预测 L 长度的相对概率分布
        # ---------------------------------------------------
        self.shape_net = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_pred_h),
            nn.GELU(),
            nn.Dropout(self.p_drop),
            nn.Linear(self.d_pred_h, self.d_count),
            nn.ReLU()
        )
        
        # ---------------------------------------------------
        # Stream 2: Scale (全局翻译规模/TE)
        # 负责预测 1 个 Log 空间的总体丰度
        # ---------------------------------------------------
        self.attention_pool = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.Tanh(),
            nn.Linear(self.d_model // 2, 1) # 输出一个注意力得分
        )

        self.scale_net = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_pred_h),
            nn.GELU(),
            nn.Dropout(self.p_drop),
            nn.Linear(self.d_pred_h, 1) # 输出 1 个 Log-TE
        )

        if init_xavier:
            self._init_parameters()

    def _init_parameters(self):
        """Initialize linear weights with Xavier and biases to zero; LayerNorm weight=1 bias=0."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    if m.out_features == self.d_count: # 最后一层
                        nn.init.constant_(m.bias, 0.1) # 初始化为微小正数防死
                    else:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, src_reps: torch.Tensor, src_mask: torch.Tensor = None, **kwargs):
        """
        Forward pass.

        Parameters
        ----------
        src_reps : torch.Tensor
            Encoder representations with shape (bs, seq_len, d_model).
        src_mask : Optional[torch.Tensor]
            Boolean pad mask with shape (bs, seq_len), True for valid tokens.
            (Note: In your previous code this was named pad_mask).

        Returns
        -------
        dict
            Contains:
            - "psite_shape": shape (bs, seq_len, d_count)
            - "te_scale": shape (bs, d_count)
            - "profile": final fused counts, shape (bs, seq_len, d_count)
        """

        # ==========================================
        # 1. 预测 Shape (局部相对密度)
        # ==========================================
        shape_preds = self.shape_net(src_reps)
        
        if src_mask is not None:
            # 直接把 Padding 区域清零即可
            mask_float = src_mask.to(src_reps.device).unsqueeze(-1).to(dtype=src_reps.dtype)
            shape_preds = shape_preds * mask_float

        # ==========================================
        # 2. 预测 Scale (全局 Log-TE 或 Log-Total-Counts)
        # ==========================================
        # 1. 计算每个 token 的原始注意力得分
        attn_logits = self.attention_pool(src_reps) # (B, L, 1)

        # 2. Mask 掉 Padding 区域
        if src_mask is not None:
            mask_bool = src_mask.to(src_reps.device).unsqueeze(-1)
            attn_logits = attn_logits.masked_fill(~mask_bool, -1e9)

        # 3. 转化为权重 (和为 1)
        attn_weights = torch.softmax(attn_logits, dim=1) # (B, L, 1)

        # 4. 加权求和 (代替原来的 mean pooling)，关键序列（如 Kozak 周围）的特征会被放大，不重要的 UTR 会被忽略
        global_rep = (src_reps * attn_weights).sum(dim=1) # (B, d_model)

        # 5. 预测 Log-TE
        scale_log = self.scale_net(global_rep) # (B, 1)

        # ==========================================
        # 3. 内部融合 (Fusion)
        # ==========================================
        # 将预测的 Log 规模通过 exp 还原为物理倍数 (仅用于融合，算 Loss 不用这个)
        scale_linear = torch.exp(scale_log) # (B,)
        
        # 绝对翻译量 = 总规模 * 相对概率分布
        fused_counts = scale_linear.unsqueeze(1) * shape_preds # (B, 1) * (B, L) -> (B, L)

        # 返回一个字典，里面包含解耦的所有部分，供 Trainer 分别算 Loss
        return {
            "psite_shape": shape_preds,
            "te_scale": scale_log,
            "profile": fused_counts
        }
    
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
    ) -> "DecoupledCountHead":
        """
        Create a DecoupledCountHead by inferring missing dimensions from `parent_model`.
        """

        # infer d_model
        if d_model is None:
            if hasattr(parent_model, "d_model"):
                d_model = int(getattr(parent_model, "d_model"))
            else:
                raise ValueError("d_model not provided and parent_model has no attribute 'd_model'")

        # infer d_count
        if d_count is None:
            if hasattr(parent_model, "d_count"):
                d_count = int(getattr(parent_model, "d_count"))
            elif hasattr(parent_model, "src_emb") and hasattr(parent_model.src_emb, "d_count"):
                d_count = int(getattr(parent_model.src_emb, "d_count"))
            else:
                raise ValueError("d_count not provided and cannot be inferred from parent_model")

        # now construct
        return cls(d_model=d_model, d_count=d_count, d_pred_h=d_pred_h, p_drop=p_drop, init_xavier=init_xavier)


class TranslationProfileHead(nn.Module):
    """
    TranslationProfileHead

    - If some dims are not provided, you can create this head from a `parent_model` using
      `TranslationProfileHead.create_from_model(parent_model, **overrides)`.
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
        self.name = "TranslationProfileHead"
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
            nn.Softplus(beta=10)
        )

        if init_xavier:
            self._init_parameters()

    def _init_parameters(self):
        """Initialize linear weights with Xavier and biases to zero; LayerNorm weight=1 bias=0."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    if m.out_features == self.d_count: # 最后一层
                        nn.init.constant_(m.bias, 0.1) # 初始化为微小正数防死
                    else:
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
            Encoder representations with shape (bs, seq_len, d_model).
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