import math
from typing import Optional
import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input
from model.position_embedding import LlamaRotaryEmbeddingExt


class FlashMultiHeadedAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, p_drop: float = 0.1, causal: bool = False):
        """
        Flash attention wrapper that supports:
        - training / parallel path: flash_attn_varlen_qkvpacked_func (fast for full-seq)
        - eval / incremental path: flash_attn_with_kvcache (KV cache incremental decode)
        Use forward(..., compute="flash") to force varlen; compute="kvcache" to force kvcache;
        compute="auto" picks varlen when module is in training mode, otherwise kvcache if caches provided.

        parameters:
        d_model: total dimension of model
        heads: attention heads num
        p_drop: attention dropout
        causal: if causal attention
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.RoPE = LlamaRotaryEmbeddingExt(self.head_dim) # for head dim
        self.causal = causal
        self.p_drop = p_drop

        # projection layers (can be wrapped by LoRA externally if desired)
        self.toqueries = nn.Linear(d_model, d_model)
        self.tokeys = nn.Linear(d_model, d_model)
        self.tovalues = nn.Linear(d_model, d_model)
        self.unifyheads = nn.Linear(d_model, d_model) # transform to d_model after catenating multi-head

        # default softmax scale value（buffer, stored along with module.to(device))
        default_scale = 1.0 / math.sqrt(self.head_dim)
        self.register_buffer('softmax_scale', torch.tensor(default_scale, dtype=torch.float32))
        

    def forward(self, 
                queries: torch.Tensor,  # (bs, seq_len, d_model) — here seq_len same as kv
                kv: torch.Tensor,       # (bs, seq_len, d_model)
                attention_mask: Optional[torch.Tensor] = None,  # (bs, seq_len) 1=valid  
                softmax_scale: Optional[float] = None) -> torch.Tensor:
        """
        Self- or Cross-attention computed by flash varlen kernel
        x: shape (bs, seq_len, d_model) after positional embedding
        attention_mask: a bool (1/0) tensor of shape (bs, seq_len)
        Returns: (bs, seq_len, d_model)  -- outputs for each query position.
        Precondition: queries and kv have same seq_len (or you've padded them to same length)
        """
        bs, seq_len = queries.shape[0], queries.shape[1]
        # 1) project to head space: (bs, seq_len, n_heads, head_dim) -> transpose -> (bs, n_heads, seq_len, head_dim)
        query = self.toqueries(queries).view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        key   = self.tokeys(kv).view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        value = self.tovalues(kv).view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # 2) apply RoPE to q/k if you use rotary; ensure positions align (absolute positions)
        #    If your RoPE expects explicit positions, pass them; otherwise RoPE() may use implicit 0..L-1.
        query = self.RoPE(query)
        key   = self.RoPE(key)

        # 3) pack q,k,v into the layout expected by varlen kernel (matches your training forward)
        #    In your earlier code you used: qkv = torch.stack([query, key, value], dim=2).transpose(1, 3)
        qkv = torch.stack([query, key, value], dim=2).transpose(1, 3)  # -> (bs, seq_len, 3, n_heads, head_dim) or kernel-specific layout

        # 4) unpad (remove padding positions for speed). kernel expects int mask
        if attention_mask is None:
            attention_mask = torch.ones((bs, seq_len), dtype=torch.int, device=queries.device)
        #   qkv_unpad: (sum_bs(seq_lens), 3, n_heads, head_dim)
        #   attention_mask: a bool (1/0) tensor of shape (bs, seq_len), 1 means valid and 0 means not valid.
        #   indices: for recover pad
        qkv_unpad, indices, cu_seqlens, max_seqlen, _ = unpad_input(qkv, attention_mask.to(torch.int))

        # 5) call flash-attn varlen kernel
        # NOTE: keep dtype choices consistent with what flash-attn expects on your HW (e.g. bfloat16)
        #       1) qkv_unpad -> bfloat16
        #       2) kernel return bfloat16/float16 -> cast to float32
        #    out_unpad: (sum_bs(seq_lens), heads, head_dim)
        attention_qkv_unpad = flash_attn_varlen_qkvpacked_func(
            qkv_unpad.to(torch.bfloat16),
            cu_seqlens,
            max_seqlen,
            dropout_p = self.p_drop,
            causal = self.causal,
            softmax_scale = (float(softmax_scale) if softmax_scale is not None else None)
        ).to(torch.float32)

        # 6) recover padded layout: (bs, seq_len, n_heads, head_dim)
        attention_qkv = pad_input(attention_qkv_unpad, indices, bs, seq_len)

        # 7) reshape to (bs, seq_len, d_model) and project back
        attention_qkv = attention_qkv.reshape(bs, seq_len, self.n_heads * self.head_dim)
        representations_batch = self.unifyheads(attention_qkv)

        return representations_batch
    