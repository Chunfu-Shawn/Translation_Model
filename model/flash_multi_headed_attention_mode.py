import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Literal, Any
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func, flash_attn_with_kvcache
from flash_attn.bert_padding import unpad_input, pad_input
from model.rotary_position_embedding import LlamaRotaryEmbeddingExt


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
        n_heads: attention heads num
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

    # ----------------------------
    # small helpers
    # ----------------------------
    def _project_qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project input x -> (q, k, v) with shapes (bs, n_heads, seq_len, head_dim).
        """
        bs, seq_len = x.shape[0], x.shape[1]
        q = self.toqueries(x).view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.tokeys(x).view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.tovalues(x).view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        return q, k, v
    
    @staticmethod
    def _to_bf16(t: torch.Tensor) -> torch.Tensor:
        """Cast to bfloat16 in a single place (so easy to switch to .half() if needed)."""
        return t.to(torch.bfloat16)
    
    @staticmethod
    def _ensure_contiguous(t: torch.Tensor) -> torch.Tensor:
        return t.contiguous() if not t.is_contiguous() else t
    
    @staticmethod
    def _unpack_flash_kvcache_return(ret: Any) -> torch.Tensor:
        """
        flash_attn_with_kvcache may return different shapes/types across versions.
        Normalize to return `out` tensor (unpadded or per-batch depending on impl).
        If ret is tuple, assume first element is output.
        """
        if isinstance(ret, tuple) or isinstance(ret, list):
            return ret[0]
        return ret

    # ----------------------------
    # KV cache utilities for decoding
    # ----------------------------
    def init_kv_cache(self, max_seq_len: int, batch_size: int, device: Optional[torch.device] = None, dtype=torch.bfloat16):
        """
        Allocate KV cache buffers and initial cache_seqlens for a decode session.
        Returns: k_cache, v_cache, cache_seqlens
        Layout used here: (batch_size, max_seq_len, n_heads, head_dim)
        """
        device = device or next(self.parameters()).device
        k_cache = torch.zeros((batch_size, max_seq_len, self.n_heads, self.head_dim), dtype=dtype, device=device)
        v_cache = torch.zeros((batch_size, max_seq_len, self.n_heads, self.head_dim), dtype=dtype, device=device)
        cache_seqlens = torch.zeros((batch_size,), dtype=torch.int32, device=device)  # how many tokens written so far per batch item
        return k_cache, v_cache, cache_seqlens


    ## ----- self-attention for inference (autoregression)
    def decode_step_with_kvcache(self,
                                 x_new: torch.Tensor,
                                 k_cache: torch.Tensor,
                                 v_cache: torch.Tensor,
                                 cache_seqlens: torch.Tensor,
                                 softmax_scale: Optional[float] = None):
        """
        Incremental decode for one new step.
        x_new: (bs, 1, d_model)
        k_cache, v_cache: (bs, seq_len, n_heads, head_dim)
        cache_seqlens: (bs,) current lengths (so new token positions = cache_seqlens)
        Returns out_new (bs,1,d_model), and updated caches/seq_lens.
        """
        bs = x_new.shape[0]

        q = self.toqueries(x_new).view(bs, 1, self.n_heads, self.head_dim).transpose(1,2)  # (bs, n_heads, 1, head_dim)
        k = self.tokeys(x_new).view(bs, 1, self.n_heads, self.head_dim).transpose(1,2)
        v = self.tovalues(x_new).view(bs, 1, self.n_heads, self.head_dim).transpose(1,2)

        # apply RoPE using per-sample absolute positions (RoPE supports positions)
        q = self.RoPE(q, positions=cache_seqlens)
        k = self.RoPE(k, positions=cache_seqlens)

        ret = flash_attn_with_kvcache(
            self._to_bf16(q).transpose(1,2),
            k_cache, v_cache,
            k=self._to_bf16(k).transpose(1,2), v=self._to_bf16(v).transpose(1,2),
            rotary_cos=None, rotary_sin=None,
            cache_seqlens=cache_seqlens,
            softmax_scale=(float(softmax_scale) if softmax_scale is not None else None),
            causal=self.causal
        ).to(torch.float32)  # cast back to float32 for stability
        out = self._unpack_flash_kvcache_return(ret)

        # reshape to (bs,1,d_model)
        try:
            out_tensor = out.reshape(bs, 1, self.n_heads * self.head_dim)
        except Exception:
            out_tensor = out.reshape(bs, self.n_heads, 1, self.head_dim).transpose(1, 2).reshape(bs, 1, -1)
        out_final = self.unifyheads(out_tensor)

        # increment per-batch seqlen by 1 (if some batch items are inactive in beam search, update with mask externally)
        cache_seqlens = cache_seqlens + 1
        return out_final, k_cache, v_cache, cache_seqlens
    
    ## ----- cross attention for inference (autoregression)
    def build_kv_cache_from_kv(self,
                               src_representations_batch: torch.Tensor,
                               max_seq_len: Optional[int] = None,
                               dtype=torch.bfloat16) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert encoder K/V (bs, n_heads, src_len, head_dim) into k_cache/v_cache layout expected by kvcache:
            k_cache/v_cache: (max_seq_len, bs, n_heads, head_dim)
        Returns: (k_cache, v_cache, cache_seqlens) where cache_seqlens is (bs,) int32 with value src_len per batch.
        If max_seq_len is None, set max_seq_len = src_len.
        Note: this does NOT apply RoPE; we expect k_enc already have RoPE applied if needed.
        """
        # project to head space: (bs, seq_len, n_heads, head_dim) -> transpose -> (bs, n_heads, seq_len, head_dim)
        _, k_enc, v_enc = self._project_qkv(src_representations_batch)

        # apply RoPE to k_enc (encoder positions assumed 0..src_len-1)
        k_enc = self.RoPE(k_enc)

        # k_enc: (bs, n_heads, src_len, head_dim)
        assert k_enc.dim() == 4 and v_enc.dim() == 4, "k_enc/v_enc expected shape (bs, n_heads, src_len, head_dim)"
        bs, n_heads, src_len, head_dim = k_enc.shape
        if max_seq_len is None:
            max_seq_len = src_len
        if max_seq_len < src_len:
            raise ValueError("max_seq_len must be >= src_len")

        device = k_enc.device
        # allocate cache buffers
        k_cache = torch.zeros((bs, max_seq_len, n_heads, head_dim), dtype=dtype, device=device)
        v_cache = torch.zeros((bs, max_seq_len, n_heads, head_dim), dtype=dtype, device=device)
        # fill first src_len slots with k_enc/v_enc (need to transpose to match layout)
        # k_enc currently (bs, n_heads, src_len, head_dim) -> take [:, :, pos, :] and place into k_cache[pos, bs, n_heads, head_dim]
        # efficient fill:
        # transpose to (src_len, bs, n_heads, head_dim)
        k_trans = k_enc.permute(0, 2, 1, 3).contiguous()  # (bs, src_len, n_heads, head_dim)
        v_trans = v_enc.permute(0, 2, 1, 3).contiguous()
        k_cache[:, :src_len, :, :] = self._to_bf16(k_trans)
        v_cache[:, :src_len, :, :] = self._to_bf16(v_trans)
        cache_seqlens = torch.full((bs,), src_len, dtype=torch.int32, device=device)
        return k_cache, v_cache, cache_seqlens

    ## ----- cross attention for inference (autoregression)
    def cross_attend_using_kvcache(self,
                                   q: torch.Tensor,
                                   k_cache: torch.Tensor,
                                   v_cache: torch.Tensor,
                                   cache_seqlens: torch.Tensor,
                                   softmax_scale: Optional[float] = None,
                                   causal: bool = False) -> torch.Tensor:
        """
        Use flash_attn_with_kvcache to compute attention for queries q against prefilled k_cache/v_cache.
        - q: (bs, q_len, n_heads, head_dim)
        - k_cache/v_cache: (bs, max_seq_len, n_heads, head_dim)
        - cache_seqlens: (bs,) int32
        Returns: out tensor shaped (bs, q_len, d_model)
        Notes:
          - We expect q to already have RoPE applied consistently with how k_cache was produced.
          - We do NOT pass k/v (no appending) so k_cache is treated as static keys/values.
        """
        # ensure q is (bs, q_len, n_heads, head_dim)
        # call kernel: signature (q, k_cache, v_cache, k=None, v=None, ..., cache_seqlens=...)
        ret = flash_attn_with_kvcache(
            self._to_bf16(q),
            k_cache, v_cache,
            k=None, v=None,
            rotary_cos=None, rotary_sin=None,
            cache_seqlens=cache_seqlens,
            softmax_scale=(float(softmax_scale) if softmax_scale is not None else None),
            causal=causal
        ).to(torch.float32)  # cast back to float32 for stability
        out = self._unpack_flash_kvcache_return(ret)
        # try reshape: commonly out is (bs, q_len, n_heads, head_dim) or (bs, n_heads, q_len, head_dim)
        bs = q.shape[0]
        q_len = q.shape[1]
        try:
            out_tensor = out.reshape(bs, q_len, self.n_heads * self.head_dim)
        except Exception:
            # try other common layout
            out_tensor = out.reshape(bs, self.n_heads, q_len, self.head_dim).transpose(1, 2).reshape(bs, q_len, -1)
        out_final = self.unifyheads(out_tensor)
        return out_final
    
    ## ----- for training or eval (parallely)
    ## could used for both self-attention and cross-attention due to the same length of q, k, v
    def forward(self,
                queries: torch.Tensor,  # (bs, seq_len, d_model) — here seq_len same as kv
                kv: torch.Tensor,       # (bs, seq_len, d_model)
                pad_mask: Optional[torch.Tensor] = None,  # (bs, seq_len) 1=valid
                softmax_scale: Optional[float] = None) -> torch.Tensor:
        """
        Self- or Cross-attention computed by flash varlen kernel.
        Returns: (bs, seq_len, d_model)  -- outputs for each query position.
        Precondition: queries and kv have same seq_len (or you've padded them to same length).
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
        if pad_mask is None:
            pad_mask = torch.ones((bs, seq_len), dtype=torch.int, device=queries.device)

        qkv_unpad, indices, cu_seqlens, max_seqlen, _ = unpad_input(qkv, pad_mask.to(torch.int))

        # 5) call flash-attn varlen kernel (dtype -> bfloat16 for best perf on recent hardware)
        out_unpad = flash_attn_varlen_qkvpacked_func(
            self._to_bf16(qkv_unpad),
            cu_seqlens,
            max_seqlen,
            dropout_p = self.p_drop,
            causal = self.causal,
            softmax_scale = (float(softmax_scale) if softmax_scale is not None else None),
        ).to(torch.float32)  # cast back to float32 for stability

        # 6) recover padded layout: (bs, seq_len, n_heads, head_dim)
        out_padded = pad_input(out_unpad, indices, bs, seq_len)

        # 7) reshape to (bs, seq_len, d_model) and project back
        out_concat = out_padded.reshape(bs, seq_len, self.n_heads * self.head_dim)
        out_final = self.unifyheads(out_concat)  # (bs, seq_len, d_model)
        return out_final