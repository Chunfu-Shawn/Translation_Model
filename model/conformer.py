import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from model.flash_multi_headed_attention import FlashMultiHeadedAttention

__author__ = "Chunfu Xiao"
__license__ = ""
__version__="1.0.0"
__maintainer__ = "Chunfu Xiao"
__email__ = "chunfushawn@gmail.com"


class ConformerConvModule(nn.Module):
    """
    Convolution module in Conformer block:
    - Pointwise conv -> GLU
    - Depthwise conv -> BatchNorm -> Swish
    - Pointwise conv
    - Dropout
    """
    def __init__(self, d_model, kernel_size=31, p_drop=0.1):
        super().__init__()
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1) # along seq_len for all d_model, 2*d_model kernels
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                                   groups=d_model, padding=kernel_size//2) # along seq_len for each d_model, d_model kernels
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.swish = nn.SiLU()
        self.pointwise2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        # x: (bs, seq_len, d_model)
        x = x.transpose(1, 2)           # -> (bs, d_model, seq_len)
        x = self.pointwise1(x)          # -> (bs, 2*d_model, seq_len)
        x = self.glu(x)                 # -> (bs, d_model, seq_len)
        x = self.depthwise(x)           # -> (bs, d_model, seq_len)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise2(x)          # -> (bs, d_model, seq_len)
        x = self.dropout(x)
        return x.transpose(1, 2)        # -> (bs, seq_len, d_model)

class ConformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, heads, conv_kernel_size=31,
                 p_drop=0.1, causal=False):
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_ff, d_model),
            nn.Dropout(p_drop)
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.multi_headed_attention = FlashMultiHeadedAttention(d_model, heads, p_drop, causal)
        self.conv_module = ConformerConvModule(d_model, kernel_size=conv_kernel_size, p_drop=p_drop)
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_ff, d_model),
            nn.Dropout(p_drop)
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, pad_mask=None):
        # Feedforward 1 (with half-step residual)
        # if pad_mask:
        x = x * pad_mask.unsqueeze(-1) # mask padding
        x = x + 0.5 * self.ffn1(x)

        # Multi-head self-attention
        attn_input = self.attn_norm(x)
        x = x + self.multi_headed_attention(attn_input, pad_mask)

        # Convolution module
        # if pad_mask:
        x = x * pad_mask.unsqueeze(-1) # mask padding
        x = x + self.conv_module(x)

        # Feedforward 2
        x = x + 0.5 * self.ffn2(x)

        # Final LayerNorm
        return self.final_norm(x)


class ConformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, d_ff, heads, conv_kernel_size=31,
                 p_drop=0.1, causal=False):
        super().__init__()
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, d_ff, heads, conv_kernel_size, p_drop, causal)
            for _ in range(num_layers)
        ])

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

class CodingConformerHead(nn.Module):
    def __init__(self, d_model, d_ff, heads, num_layers,
                 conv_kernel_size=31, p_drop=0.1):
        super().__init__()
        self.input_emb = nn.Linear(d_model, d_model)
        self.encoder = ConformerEncoder(num_layers, d_model, d_ff, heads,
                                         conv_kernel_size, p_drop)
        # conv for each frame, ignoring last 1-2 nt
        self.conv_f0 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=3, padding=0)
        self.conv_f1 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=3, padding=0)
        self.conv_f2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=3, padding=0)
        # codon downsampling for last dimension
        self.classify = nn.Linear(d_model, 1)

    def forward(self, src_data, mask):
        # src_data: (bs, seq_len, d_seq+d_count)
        x = self.input_emb(src_data)
        x = self.encoder(x, mask)

        # downsample: -> (bs, d_coding, seq_len//3), drop out last 1-2 nt
        x = x.transpose(1,2)
        ys = [
            self.conv_f0(x[:, :, :]),
            self.conv_f1(x[:, :, 1:]),
            self.conv_f2(x[:, :, 2:])
        ]
        # padding for seq_len//3
        max_n = max(y.size(2) for y in ys)
        ys_padded = [
            F.pad(y, (0, max_n - y.size(2), 0, 0))
            for y in ys
        ] # add 0 in right

        # -> (bs, seq_len/3, d_coding)
        Y = [y.transpose(1,2) for y in ys_padded]

        # get 3 * (bs, n, 1) and cantenata to (bs, n, 3)
        logits = torch.cat([self.classify(y) for y in Y], dim=-1)
        return logits