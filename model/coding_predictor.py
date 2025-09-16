import torch
import torch.nn as nn
from typing import Dict, Optional, List

__author__ = "Chunfu Xiao"
__license__ = ""
__version__="1.0.0"
__email__ = "chunfushawn@gmail.com"

class CodingNetHead(nn.Module):
    """
    Shared trunk + three binary heads: start / stop / in_orf.
    Input:
      - src_reps: (bs, L, d_model)
    Output:
      - dict of logits (not probs): keys 'start', 'stop', 'in_orf' -> tensors (bs, L)
    """
    def __init__(
            self, 
            pmt_len: int, 
            d_model: int, 
            d_pred_h: int = 256,
            p_drop: float = 0.1, 
            init_xavier: bool = True,
            use_ln: bool = True
        ):
        super().__init__()
        self.use_ln = use_ln
        self.pmt_len = int(pmt_len)
        self.ln = nn.LayerNorm(d_model) if use_ln else nn.Identity()
        self.trunk = nn.Sequential(
            nn.Linear(d_model, d_pred_h),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_pred_h, d_pred_h),
            nn.GELU(),
            nn.Dropout(p_drop),
        )

        # three heads -> scalar logit per position
        self.start_head = nn.Linear(d_pred_h, 1)
        self.stop_head  = nn.Linear(d_pred_h, 1)
        self.inorf_head = nn.Linear(d_pred_h, 1)

        # init
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

    def forward(self, src_reps: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        src_reps: (bs, L, d_model)
        returns logits dict with each (bs, L)
        """
        x = self.ln(src_reps[:, self.pmt_len:, :])  # (bs, L, d_model)
        x = self.trunk(x)      # (bs, L, d_pred_h)
        start_logits = self.start_head(x).squeeze(-1)  # (bs, L)
        stop_logits  = self.stop_head(x).squeeze(-1)
        inorf_logits = self.inorf_head(x).squeeze(-1)
        return torch.stack((start_logits, stop_logits, inorf_logits), dim=2)  # (bs, L, 3)
        # {'start': start_logits, 'stop': stop_logits, 'in_orf': inorf_logits}
    
    @classmethod
    def create_from_model(
        cls,
        parent_model: object,
        pmt_len: Optional[int] = None,
        d_model: Optional[int] = None,
        d_pred_h: Optional[int] = None,
        p_drop: float = 0.1,
        init_xavier: bool = True,
    ) -> "CodingNetHead":
        """
        Create a CodingNetHead by inferring missing dimensions from `parent_model`.

        parent_model is expected to expose attributes:
          - d_model (model hidden size)

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

        # now construct
        return cls(pmt_len=pmt_len, d_model=d_model, d_pred_h=d_pred_h, p_drop=p_drop, init_xavier=init_xavier)


# Depthwise separable conv 1D
class DepthwiseSeparableConv1D(nn.Module):
    """
    One block: depthwise dilated conv -> pointwise conv -> activation -> dropout
    Input/Output shapes for conv1d: (bs, channels, L)
    depthwise: groups=in_ch, keeps channels
    pointwise: kernel=1, maps channels -> channels
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, p_drop: float = 0.1):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation
        # depthwise
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size, stride=1, padding=pad, dilation=dilation, groups=in_ch, bias=False)
        # pointwise
        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1, bias=True)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        # x: (bs, channels, L)
        return self.dropout(
            self.act(
                self.pw(
                    self.dw(x)
                    )))

# -------------------------
# Depthwise separable convolution (lighter)
# -------------------------
class DSConvCodingHead(nn.Module):
    def __init__(
            self,
            pmt_len: int,
            d_model: int, 
            d_pred_h: int = 256,
            kernel_sizes: List[int] = [3,3],
            p_drop: float = 0.1,
            init_xavier: bool = True,
        ):
        super().__init__()
        self.name = "SepConv"
        self.pmt_len = int(pmt_len)
        self.ln = nn.LayerNorm(d_model)
        layers = []
        in_ch = d_model
        for k in kernel_sizes:
            layers.append(DepthwiseSeparableConv1D(in_ch, d_pred_h, kernel_size=k, p_drop=p_drop))
            in_ch = d_pred_h
        self.conv_stack = nn.Sequential(*layers)
        self.start_head = nn.Conv1d(d_pred_h, 1, kernel_size=1)
        self.stop_head  = nn.Conv1d(d_pred_h, 1, kernel_size=1)
        self.inorf_head = nn.Conv1d(d_pred_h, 1, kernel_size=1)

        if init_xavier:
            self._init_parameters()

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None: nn.init.ones_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, src_reps: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        src_reps: (bs, pmt_len + L, d_model)
        returns logits dict with tensors (bs, L, 3)
        """
        # LayerNorm over last dim, then permute to (bs, channels, L)
        x = self.ln(src_reps[:, self.pmt_len:, :]).permute(0,2,1)  # (bs, d_model, L)
        h = self.conv_stack(x)  # -> (bs, d_pred_h, L)

        start_logits = self.start_head(h).squeeze(1) # (bs, 1, L) -> (bs, L)
        stop_logits = self.stop_head(h).squeeze(1)
        inorf_logits = self.inorf_head(h).squeeze(1)

        return torch.stack((start_logits, stop_logits, inorf_logits), dim=2)  # (bs, L, 3)
    
    @classmethod
    def create_from_model(
        cls,
        parent_model: object,
        pmt_len: Optional[int] = None,
        d_model: Optional[int] = None,
        d_pred_h: Optional[int] = None,
        p_drop: float = 0.1,
        init_xavier: bool = True,
    ) -> "DSConvCodingHead":
        """
        Create a DSConvCodingHead by inferring missing dimensions from `parent_model`.

        parent_model is expected to expose attributes:
          - d_model (model hidden size)

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

        # now construct
        return cls(pmt_len=pmt_len, d_model=d_model, d_pred_h=d_pred_h, p_drop=p_drop, init_xavier=init_xavier)


class DepthwiseSeparableDilatedBlock(nn.Module):
    """
    One block: depthwise dilated conv -> pointwise conv -> activation -> dropout
    Input/Output shapes for conv1d: (bs, channels, L)
    depthwise: groups=in_ch, keeps channels
    pointwise: kernel=1, maps channels -> channels
    """
    def __init__(self, in_ch, out_ch: int, kernel_size: int = 3, dilation: int = 1, p_drop: float = 0.1):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation
        # depthwise
        self.dw = nn.Conv1d(
            in_ch,
            in_ch,
            kernel_size=kernel_size, 
            stride=1,
            padding=pad,
            dilation=dilation,
            groups=out_ch,
            bias=False
        )
        # pointwise
        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1, bias=True)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (bs, channels, L)
        out = self.dw(x)
        out = self.pw(out)
        out = self.act(out)
        out = self.dropout(out)
        return out

# -------------------------
# Depthwise separable dilated convolutions
# -------------------------
    
class DSDilatedConvCodingHead(nn.Module):
    """
    TCN-style head using depthwise separable dilated convs.
    - Project d_model -> d_pred_h via 1x1 conv
    - Apply num_layers of (dw dilated conv -> pw) with residuals
    - Output three 1x1 conv heads for start/stop/in_orf logits (bs, L)
    """
    def __init__(self,
                 pmt_len: int,
                 d_model: int,
                 d_pred_h: int = 128,
                 trunk_layers: int = 2,
                 task_layers: int = 2,
                 kernel_size: int = 3,
                 p_drop: float = 0.1,
                 init_xavier = True):
        super().__init__()
        self.name = "SepDiatedConv"
        self.pmt_len = pmt_len
        self.ln = nn.LayerNorm(d_model)

        # Build shared trunk (dilated conv stack)
        trunk_layers_list = []
        in_ch = d_model
        for i in range(trunk_layers):
            dil = 2 ** i
            trunk_layers_list.append(DepthwiseSeparableConv1D(in_ch, d_pred_h, kernel_size=kernel_size, dilation=dil, p_drop=p_drop))
            in_ch = d_pred_h
        self.trunk = nn.Sequential(*trunk_layers_list)

        # task conv for smaller dimension
        # local_stack: used to produce representations suitable for start/stop (local detection)
        local_layers = []
        for j in range(task_layers):
            # smaller dilation for boundary detection (more local)
            dil = 1 if j == 0 else 2 ** j
            local_layers.append(DepthwiseSeparableConv1D(d_pred_h, d_pred_h, kernel_size=kernel_size, dilation=dil, p_drop=p_drop))
        self.local_stack = nn.Sequential(*local_layers)

        # region stack: for in_orf (needs smoother/larger receptive field)
        # region_layers = []
        # for j in range(task_layers):
        #     dil = 2 ** (j + 1)  # start from larger dilation
        #     region_layers.append(DepthwiseSeparableConv1D(d_pred_h, d_pred_h, kernel_size=kernel_size, dilation=dil, p_drop=p_drop))
        # self.region_stack = nn.Sequential(*region_layers)

        # final pointwise heads
        self.start_stop_head = nn.Conv1d(d_pred_h, 3, kernel_size=1)
        # self.inorf_head = nn.Conv1d(d_pred_h, 1, kernel_size=1)

        if init_xavier:
            self._init_parameters()

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None: nn.init.ones_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, src_reps: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        src_reps: (bs, pmt_len + L, d_model)
        returns logits dict with tensors (bs, L, 3)
        """
        # LayerNorm over last dim, then permute to (bs, channels, L)
        x = self.ln(src_reps[:, self.pmt_len:, :]).permute(0,2,1)  # (bs, d_model, L)
        shared = self.trunk(x)  # -> (bs, d_pred_h, L)
        local = self.local_stack(shared)  # -> (bs, d_pred_h, L)
        # region = self.region_stack(shared)  # -> (bs, d_pred_h, L)

        start_stop_logits = self.start_stop_head(local) # (bs, d_pred_h, L) -> (bs, 3, L)
        # inorf_logits = self.inorf_head(region)

        return start_stop_logits # torch.cat((start_stop_logits, inorf_logits), dim=1).permute(0,2,1)  # (bs, L, 3)
    
    @classmethod
    def create_from_model(
        cls,
        parent_model: object,
        pmt_len: Optional[int] = None,
        d_model: Optional[int] = None,
        d_pred_h: Optional[int] = None,
        trunk_layers: Optional[int] = None,
        task_layers: Optional[int] = None,
        kernel_size: int = 3,
        p_drop: float = 0.1,
        init_xavier: bool = True,
    ) -> "DSDilatedConvCodingHead":
        """
        Create a DSDilatedConvCodingHead by inferring missing dimensions from `parent_model`.

        parent_model is expected to expose attributes:
          - d_model (model hidden size)

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

        # now construct
        return cls(pmt_len=pmt_len, d_model=d_model, d_pred_h=d_pred_h, trunk_layers=trunk_layers, task_layers=task_layers,
                   kernel_size=kernel_size,  p_drop=p_drop, init_xavier=init_xavier)

