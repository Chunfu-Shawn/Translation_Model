import math
import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input
from model.rotary_position_embedding import LlamaRotaryEmbeddingExt


class FlashMultiHeadedAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, p_drop: float = 0.1, causal: bool = False):
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
        assert d_model % heads == 0, "d_model must be divisible by heads"
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
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
        

    def forward(self, x, attention_mask, softmax_scale=None):
        """
        x: shape (bs, seq_len, d_model) after positional embedding
        attention_mask: a bool (1/0) tensor of shape (bs, seq_len)
        returns: out (bs, seq_len, d_model)
        """
        bs, seq_len = x.shape[0:2]

        # project to q/k/v using LoRA-wrapped linears (these act like nn.Linear)
        # x shape: (bs, seq_len, n_heads*head_dim)
        # head_dim * n_heads = d_model
        # .view(bs, seq_len, n_heads, head_dim) split d_model to [n_heads, head_dim]
        # .transpose(1,2) -> (bs, n_heads, seq_len, head_dim) for calculating parallelly in muti-heads
        # be careful for attention_mask shape
        query = self.toqueries(x).view(bs, seq_len, self.heads, self.head_dim).transpose(1, 2)
        key = self.tokeys(x).view(bs, seq_len, self.heads, self.head_dim).transpose(1, 2)
        value = self.tovalues(x).view(bs, seq_len, self.heads, self.head_dim).transpose(1, 2)

        # RoPE: expect shape (bs, n_heads, seq_len, head_dim)
        query = self.RoPE(query)
        key = self.RoPE(key)

        # 1. pack q,k,v into the expected qkv layout for the varlen kernel
        #    qkv: (bs, n_heads, 3, seq_len, head_dim) -> transpose to (bs, seq_len, 3, n_heads, head_dim)
        qkv = torch.stack([query, key, value], dim=2).transpose(1, 3)

        # 2. remove pad
        #    qkv_unpad: (sum_bs(seq_lens), 3, n_heads, head_dim)
        #    attention_mask: a bool (1/0) tensor of shape (bs, seq_len), 1 means valid and 0 means not valid.
        #    indices: for recover pad
        qkv_unpad, indices, cu_seqlens, max_seqlen, seqused = unpad_input(qkv, attention_mask.to(torch.int))
        
        # decide softmax_scale to pass to the kernel
        if softmax_scale is None:
            softmax_scale_val = float(self.softmax_scale.item())
        else:
            softmax_scale_val = float(softmax_scale)

        # 3. call flash-attn varlen kernel
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
            softmax_scale = softmax_scale_val
        ).to(torch.float32)

        # 4. recover padded layout: (bs, seq_len, heads, head_dim)
        attention_qkv = pad_input(attention_qkv_unpad, indices, bs, seq_len)

        # 5. reshape output of multi-heads to (bs, seq_len, n_heads, head_dim) -> (bs, seq_len, n_heads*head_dim)
        attention_qkv = attention_qkv.reshape(bs, seq_len, self.heads * self.head_dim)

        # 6. final linear to project back to d_model
        representations_batch = self.unifyheads(attention_qkv)

        return representations_batch
    