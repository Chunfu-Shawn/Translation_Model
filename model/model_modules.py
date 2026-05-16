import copy
import warnings
import math
import torch
import torch.nn as nn
from model.position_embedding import LlamaRotaryEmbeddingExt

__author__ = "Chunfu Xiao"
__license__ = ""
__version__="1.1.0"
__maintainer__ = "Chunfu Xiao"
__email__ = "chunfushawn@gmail.com"

# =========================================================================
# 1. Dynamic Import and Environment Detection
# =========================================================================
try:
    from model.flash_multi_headed_attention import FlashMultiHeadedAttention
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    warnings.warn(
        "FlashMultiHeadedAttention could not be imported. "
        "The model will automatically fall back to standard MultiHeadedAttention."
    )

"""
ABBR.
bs: batch size,
seq_len: max src/trg token-sequence length,
head_dim: key/value size; head dimensionality
n_heads/h: number of heads
d_model: model dimension
pe: positional encoding
d_ff:  inner-layer dimensionality
p_drop: probability of dropout
ffn:  position-wise feed-forward networks
MHA: multi-head attention
"""

# Part 1: ================== Modules ==================

def replicate_module(module, copies):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(copies)]) 

class LinearEmbedding(nn.Module):
    """
    Project sequence and RPF density safely with non-linearity and normalization.
    """
    def __init__(self, d_seq, d_count, output_model, p_drop=0.1):
        super().__init__()
        self.seq_emb_layer = nn.Linear(d_seq, output_model)
        self.count_emb_layer = nn.Linear(d_count, output_model)
        
        self.seq_ln = nn.LayerNorm(output_model)
        self.count_ln = nn.LayerNorm(output_model)
        
        self.unify_emb_layer = nn.Sequential(
            nn.Linear(output_model * 2, output_model),
            nn.GELU(), 
            nn.Dropout(p_drop)
        )

    def forward(self, seq_tokens, count_tokens):
        seq_embeddings = self.seq_ln(self.seq_emb_layer(seq_tokens))
        count_embeddings = self.count_ln(self.count_emb_layer(count_tokens))
        
        concat_emb = torch.cat([seq_embeddings, count_embeddings], dim=-1)
        return self.unify_emb_layer(concat_emb)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) 
        self.linear2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()

    def forward(self, reps_batch):
        return self.linear2(self.gelu(self.linear1(reps_batch))) 


class AddAdaZeroLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization with Gating (adaLN-Zero) with Information Bottleneck.
    """
    def __init__(self, d_model, p_drop, adaptive_dim=16, gamma_scale=1):
        super().__init__()
        self.d_model = d_model
        self.gamma_scale = gamma_scale
        self.dropout = nn.Dropout(p=p_drop)
        
        self.LN = nn.LayerNorm(d_model, elementwise_affine=False)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(adaptive_dim, d_model * 3)
        )
        
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, reps_batch, sublayer_module, compact_style):
        style = self.adaLN_modulation(compact_style)        
        gamma, beta, alpha = style.chunk(3, dim=-1)         
        
        gamma = torch.tanh(gamma) * self.gamma_scale
        
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        alpha = alpha.unsqueeze(1)
        
        normed = (1 + gamma) * self.LN(reps_batch) + beta
        output = self.dropout(sublayer_module(normed))
        
        return reps_batch + (alpha * output)
    
# =========================================================================
# 2. Standard Attention Implementation (Aligned with Flash API)
# =========================================================================
class Attention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.toqueries = nn.Linear(input_dim, output_dim)
        self.tokeys = nn.Linear(input_dim, output_dim)
        self.tovalues = nn.Linear(input_dim, output_dim)
        self.head_dim = output_dim

    def self_attention(self, query, key, value, attention_mask):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim) 

        if attention_mask is not None:
             scores.masked_fill_(attention_mask == False, float("-inf"))

        attention_weights = nn.Softmax(dim=-1)(scores) 
        attention_qkv = torch.matmul(attention_weights, value)  
        return attention_qkv

    # Removed the legacy forward from the base Attention class 
    # as we now implement the exact unified signature in the subclass.


class MultiHeadedAttention(Attention):
    def __init__(self, d_model, heads, p_drop=0.1): 
        super().__init__(d_model, d_model) 
        assert d_model % heads == 0
        self.head_dim = d_model // heads  
        self.heads = heads
        self.RoPE = LlamaRotaryEmbeddingExt(self.head_dim) 
        self.unifyheads = nn.Linear(d_model, d_model) 
        self.dropout = nn.Dropout(p_drop) 

    def forward(self, queries: torch.Tensor, kv: torch.Tensor, attention_mask=None):
        """
        Aligned with FlashMultiHeadedAttention forward signature.
        queries: (bs, seq_len_q, d_model)
        kv: (bs, seq_len_kv, d_model)
        attention_mask: (bs, seq_len_kv)
        """
        bs = queries.shape[0]
        
        # Project queries from the 'queries' tensor
        query = self.toqueries(queries).view(bs, -1, self.heads, self.head_dim).transpose(1, 2)
        
        # Project keys and values from the 'kv' tensor
        key = self.tokeys(kv).view(bs, -1, self.heads, self.head_dim).transpose(1, 2)
        value = self.tovalues(kv).view(bs, -1, self.heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        query = self.RoPE(query)
        key = self.RoPE(key)

        if attention_mask is not None:
            # Expand mask for broadcast: (bs, 1, 1, seq_len_kv)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        attention_qkv = self.self_attention(query, key, value, attention_mask).transpose(1, 2)

        attention_qkv = attention_qkv.reshape(bs, -1, self.heads * self.head_dim)
        reps_batch = self.unifyheads(attention_qkv)
        
        return self.dropout(reps_batch)


# =========================================================================
# 3. Smart Encoder Layer (Automatic Dispatch)
# =========================================================================
class AdaZeroEncoderLayer(nn.Module):
    """
    Encoder Layer using Adaptive Layer Normalization.
    Automatically handles Flash Attention vs Standard Attention fallback.
    """
    def __init__(self, d_model, d_ff, heads, p_drop, adaptive_dim, gamma_scale):
        super().__init__()
        
        self.sublayers = replicate_module(
            AddAdaZeroLayerNorm(d_model, p_drop, adaptive_dim, gamma_scale), 2
        )
        
        self.d_model = d_model
        
        # -------------------------------------------------------------
        # Robust dispatch logic: prioritize Flash Attention, 
        # otherwise gracefully fall back to standard MHA
        # -------------------------------------------------------------
        self.use_flash_attention = False
        
        if HAS_FLASH_ATTN:
            try:
                self.multi_headed_attention = FlashMultiHeadedAttention(d_model, heads, p_drop)
                self.use_flash_attention = True
            except Exception as e:
                warnings.warn(f"FlashMultiHeadedAttention initialization failed: {e}. Falling back.")
        
        if not self.use_flash_attention:
            self.multi_headed_attention = MultiHeadedAttention(d_model, heads, p_drop)
            
        self.ffn = PositionwiseFeedForward(d_model, d_ff)

    def forward(self, src_reps, src_mask, compact_style):
        """
        Passes the continuous `compact_style` to the AdaLN sublayers.
        Handles API routing transparently.
        """
        # Since both Attention modules now share the exact same signature, 
        # we can call them identically. For self-attention, queries and kv are both src_reps.
        def encoder_self_attention(srb):
            return self.multi_headed_attention(queries=srb, kv=srb, attention_mask=src_mask)

        # 1. Self-Attention Block with AdaLN
        src_reps = self.sublayers[0](src_reps, encoder_self_attention, compact_style)
        
        # 2. FFN Block with AdaLN
        src_reps = self.sublayers[1](src_reps, self.ffn, compact_style)

        return src_reps
    

# Part 2: =================== Encoder ======================

class AdaEncoder(nn.Module):
    """
    Stack of AdaEncoderLayers.
    """
    def __init__(self, encoder_layer, number_of_layers):
        super().__init__()
        self.encoder_layers = replicate_module(encoder_layer, number_of_layers)
        self.LN = nn.LayerNorm(encoder_layer.d_model)

    def forward(self, src_embs, src_mask, compact_style):
        """
        Propagates the `compact_style` context through all encoder layers.
        """
        src_reps = src_embs
        
        for encoder_layer in self.encoder_layers:
            src_reps = encoder_layer(src_reps, src_mask, compact_style)
            
        return self.LN(src_reps)