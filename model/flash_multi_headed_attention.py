import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input
from model.rotary_position_embedding import LlamaRotaryEmbeddingExt

class FlashMultiHeadedAttention(nn.Module):
    def __init__(self, d_model, heads, p_drop=0.1, causal=False):
        """
        d_model: 输入/输出的总维度 (bs, seq_len, model dimension)
        heads: number of heads
        p_drop: attention dropout prob
        causal: whether caual mask
        """
        super().__init__()
        assert d_model % heads == 0
        self.d_model  = d_model
        self.heads  = heads
        self.head_dim   = d_model // heads
        self.RoPE = LlamaRotaryEmbeddingExt(self.head_dim) # for head dim
        self.causal = causal
        self.p_drop = p_drop

        # Compute the queries, keys and values for all heads
        self.toqueries = nn.Linear(d_model, d_model)
        self.tokeys = nn.Linear(d_model, d_model)
        self.tovalues = nn.Linear(d_model, d_model)
        self.unifyheads = nn.Linear(d_model, d_model) # transform to d_model after catenating multi-head

    def forward(self, x, attention_mask):
        """
        x: shape (bs, seq_len, d_model) after positional embedding
        attention_mask: a bool (1/0) tensor of shape (bs, seq_len)
        returns: out (bs, seq_len, d_model)
        """
        bs, seq_len = x.shape[0:2]
        # x shape: (bs, seq_len, n_heads*head_dim)
        # head_dim * n_heads = d_model
        # .view(bs, seq_len, n_heads, head_dim) split d_model to [n_heads, head_dim]
        # .transpose(1,2) -> (bs, n_heads, seq_len, head_dim) for calculating parallelly in muti-heads
        # be careful for attention_mask shape
        query = self.toqueries(x).view(bs, seq_len, self.heads, self.head_dim).transpose(1, 2)
        key = self.tokeys(x).view(bs, seq_len, self.heads, self.head_dim).transpose(1, 2)
        value = self.tovalues(x).view(bs, seq_len, self.heads, self.head_dim).transpose(1, 2)

        if "RoPE":
            query = self.RoPE(query)
            key = self.RoPE(key)
        
        # 1. pack query, key and value
        #    qkv: (bs, n_heads, 3, seq_len, head_dim) -> (bs, seq_len, 3, n_heads, head_dim)
        qkv = torch.stack([query, key, value], dim=2).transpose(1, 3)

        # 2. generate cu_seqlens
        # seq_lens = attention_mask.sum(dim=1).to(torch.int32)  # [bs]
        # cu_seqlens = torch.cat([
        #     torch.zeros(1, device=x.device, dtype=torch.int32),
        #     seq_lens.cumsum(dim=0)], dim=0)  # [bs+1]
        # max_seqlen = int(seq_lens.max())

        # 3. remove pad
        #    qkv_unpad: (sum_bs(seq_lens), 3, n_heads, head_dim)
        #    attention_mask: a bool (1/0) tensor of shape (bs, seq_len), 1 means valid mask and 0 means not valid.
        #    indices: for recover pad
        qkv_unpad, indices, cu_seqlens, max_seqlen, seqused = unpad_input(qkv, attention_mask.to(torch.int))
        
        # 4. use FlashAttention-2 kernel (varlen)
        #    out_unpad: (sum_bs(seq_lens), heads, head_dim)
        attention_qkv_unpad = flash_attn_varlen_qkvpacked_func(
            qkv_unpad.to(torch.bfloat16),
            cu_seqlens,
            max_seqlen,
            dropout_p=self.p_drop,
            causal=self.causal
        ).to(torch.float32)

        # 5. recover to (bs, seq_len, heads, head_dim)
        attention_qkv = pad_input(attention_qkv_unpad, indices, bs, seq_len)

        # 6. catenate output of multi-heads: (bs, seq_len, n_heads, head_dim) -> (bs, seq_len, n_heads*head_dim)
        attention_qkv = attention_qkv.reshape(bs, seq_len, self.heads * self.head_dim)
        representations_batch = self.unifyheads(attention_qkv)

        return representations_batch
