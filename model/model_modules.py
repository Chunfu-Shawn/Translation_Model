import copy
import torch
import torch.nn as nn
from typing import Optional
from model.rotary_position_embedding import LlamaRotaryEmbeddingExt
from model.flash_multi_headed_attention import FlashMultiHeadedAttention

__author__ = "Chunfu Xiao"
__license__ = ""
__version__="1.0.0"
__maintainer__ = "Chunfu Xiao"
__email__ = "chunfushawn@gmail.com"

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

# Part1: ================== modules ==================

def replicate_module(module, copies):
    # deepcopy for independent parameters in different layers
    return nn.ModuleList([copy.deepcopy(module) for _ in range(copies)]) # Module list

class HeadAdapter(nn.Module):
    """
    Wrap an existing head-like module and present canonical signature:
      forward(src_reps, src_mask=None, trg_inputs=None, **kwargs)
    `call_style` controls how to call wrapped module:
      - "src_only": call module(src_reps, src_mask, **kwargs)
      - "decoder_like": call module(src_reps, trg_inputs, src_mask, **kwargs)
      - "custom": use user-supplied callable adapter_fn(src_reps, src_mask, trg_inputs, **kwargs)
    """
    def __init__(self, module: nn.Module, requires_trg_inputs: bool = False, name: Optional[str] = None, adapter_fn=None):
        super().__init__()
        self.module = module
        self.requires_trg_inputs = bool(requires_trg_inputs)
        self.adapter_fn = adapter_fn
        self.name = name or getattr(module, "name", "BaseHead")

    def forward(self, src_reps, src_mask=None, trg_inputs=None, **kwargs):
        if self.adapter_fn is not None:
            return self.adapter_fn(src_reps, src_mask, trg_inputs, **kwargs)

        if self.requires_trg_inputs:
            # assume wrapped module expects (src_reps, src_mask, trg_inputs, ...)
            return self.module(src_reps, src_mask, trg_inputs, **kwargs)
        else:
            # assume wrapped module expects (src_reps, src_mask, ...)
            return self.module(src_reps, src_mask, **kwargs)
        
class Attention(nn.Module):
    def __init__(self, input_dim, output_dim):
        # for single-head, input_dim is equal to output_dim
        super().__init__()
        # Compute the queries, keys and values for all heads
        self.toqueries = nn.Linear(input_dim, output_dim)
        self.tokeys = nn.Linear(input_dim, output_dim)
        self.tovalues = nn.Linear(input_dim, output_dim)
        self.head_dim = output_dim

    # Scaled dot-product attention:
    def self_attention(self, query, key, value, attention_mask):
        # query/key/value:  (bs, seq_len, head_dim)/(bs, n_heads, seq_len, head_dim)
        # attention_mask shape = (bs, 1, seq_len)/(bs, 1, 1, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim)) 
        # if shape of query and key is (bs, seq_len, head_dim), shape of scores = (bs, seq_len, seq_len)
        # if shape of query and key is (bs, n_heads, seq_len, head_dim), shape of scores = (bs, n_heads, seq_len, seq_len)

        if attention_mask is not None:
            # attention_mask: boolen tensor, shape is (bs, 1, seq_len) or (bs, 1, 1, seq_len) for broadcast
                scores.masked_fill_(attention_mask == torch.tensor(False), float("-inf"))

        # Softmax dim=-1 stands for apply the softmax along the last dimension
        attention_weights = nn.Softmax(dim=-1)(scores)  # (bs, n_heads, seq_len, seq_len)/(bs, seq_len, seq_len)
        attention_qkv = torch.matmul(attention_weights, value)   # (bs, seq_len, head_dim)/(bs, n_heads, seq_len, head_dim)
        return attention_qkv

    def forward(self, x, attention_mask):
        # qkv shape: (bs, seq_len, d_model)
        query = self.toqueries(x)
        key = self.tokeys(x)
        value = self.tovalues(x)
        attention_qkv = self.self_attention(query, key, value, attention_mask)  # shape:  (bs, seq_len, d_model)
        return attention_qkv


class MultiHeadedAttention(Attention):
    def __init__(self, d_model, heads):
        super().__init__(d_model, d_model) # set toqueries, tokeys, tovalues in Class Attention
        assert d_model % heads == 0
        self.head_dim = d_model // heads  # head dimension
        self.heads = heads
        self.RoPE = LlamaRotaryEmbeddingExt(self.head_dim) # for head dim
        self.unifyheads = nn.Linear(d_model, d_model) # transform to d_model after catenating multi-head
        self.sqrt_head_dim = torch.sqrt(torch.tensor(self.head_dim)) # precomputate for attention

    def forward(self, x, attention_mask):
        batch_size = x.shape[0]
        # x shape: (bs, seq_len, n_heads*head_dim)
        # head_dim * n_heads = d_model
        # .view(bs, seq_len, n_heads, head_dim) split d_model to [n_heads, head_dim]
        # .transpose(1,2) -> (bs, n_heads, seq_len, head_dim) for calculating parallelly in muti-heads
        # be careful for attention_mask shape
        query = self.toqueries(x).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        key = self.tokeys(x).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        value = self.tovalues(x).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)

        if "RoPE":
            query = self.RoPE(query)
            key = self.RoPE(key)

        # mask: shape = (bs, 1, seq_len)/(bs, 1, 1, seq_len)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # output shape:  (bs, n_heads, seq_len, head_dim) -> (bs, seq_len, n_heads, head_dim)
        attention_qkv = self.self_attention(query, key, value, attention_mask).transpose(1, 2)

        # catenate output of multi-heads: (bs, seq_len, n_heads, head_dim) -> (bs, seq_len, n_heads*head_dim)
        attention_qkv = attention_qkv.reshape(batch_size, -1, self.heads * self.head_dim)
        reps_batch = self.unifyheads(attention_qkv)
        return reps_batch


class AddPromptEmbedding(nn.Module):
    """
    Prompt = prompt_base (pmt_len x d_model, shared) + cell_embed (d_model, per-cell).
    - Saves parameters vs storing pmt_len * d_model per cell.
    - Supports a special mask index = num_cells (i.e. embedding table size = num_cells + 1).
    """
    def __init__(self, pmt_len: int, num_cells: int, d_model: int, prompt_scale: float = 1.0):
        super().__init__()
        self.pmt_len = pmt_len
        self.d_model = d_model
        self.num_cells = num_cells
        self.mask_idx = num_cells  # reserved index for [MASK]
        # shared prompt base (learned, shape pmt_len x d_model)
        self.prompt_base = nn.Parameter(torch.randn(pmt_len, d_model) * 0.02)
        # per-cell vector (num_cells + 1 for mask slot)
        self.cell_embed = nn.Embedding(num_cells + 1, d_model) # cell_idx == num_cells is [MASK]
        # optional scale (small scalar) to control magnitude
        self.prompt_scale = prompt_scale

        # init
        nn.init.normal_(self.cell_embed.weight, mean=0.0, std=0.02)

    def forward(self, src_embs: torch.Tensor, src_mask: torch.Tensor, cell_idx: torch.Tensor):
        """
        src_embs: (bs, seq_len, d_model)
        src_mask: (bs, seq_len)  boolean (True=valid)
        cell_idx: (bs,) long with values in [0, num_cells] where num_cells = mask idx.
        returns:
            x: (bs, pmt_len + seq_len, d_model)
            new_mask: (bs, pmt_len + seq_len) bool
        """
        bs = src_embs.size(0)
        device = src_embs.device

        # cell embedding: (bs, d_model)
        cell_vec = self.cell_embed(cell_idx.to(device))    # (bs, d_model)

        # prompt_base: (pmt_len, d_model) -> expand to (bs, pmt_len, d_model)
        prompt_base = self.prompt_base.unsqueeze(0).expand(bs, -1, -1)  # (bs, pmt_len, d_model)
        # add cell vector to each prompt token (broadcast)
        prompt_tokens = prompt_base + cell_vec.unsqueeze(1) * self.prompt_scale  # (bs, pmt_len, d_model)

        # prepend prompts and update mask
        x = torch.cat([prompt_tokens, src_embs], dim=1)

        # prompt_mask True (valid tokens)
        prompt_mask = torch.ones((bs, self.pmt_len), dtype=src_mask.dtype, device=device)
        new_mask = torch.cat([prompt_mask, src_mask.to(device)], dim=1)

        return x, new_mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, p_drop=None, max_seq_length=20000, pos_nor=False):
        super().__init__()
        self.dropout = nn.Dropout(p=p_drop) if p_drop is not None else None
        position_id = torch.arange(0, max_seq_length).unsqueeze(1)  # (max_seq_length, 1)
        frequencies = torch.pow(10000., -torch.arange(0, d_model, 2, dtype=torch.float) / d_model)

        pe = torch.zeros(max_seq_length, d_model)
        # sine on even positions
        pe[:, 0::2] = torch.sin(position_id * frequencies)
        # cosine on odd positions (different dimensions)
        pe[:, 1::2] = torch.cos(position_id * frequencies) if d_model % 2 == 0 else torch.cos(position_id * frequencies)[:,:-1]
        self.register_buffer('pe', pe)

    def forward(self, embeddings_batch):
        # embedding_batch  shape: (bs, seq_len, d_model)
        # pe shape: (max_seq_length, d_model)
        # pe shape broad_casted -> (bs, seq_len, d_model)
        assert embeddings_batch.ndim == 3 and embeddings_batch.shape[-1] == self.pe.shape[-1]
        positional_encodings = embeddings_batch + self.pe[:embeddings_batch.shape[1]]
        if self.dropout is not None:
            positional_encodings = self.dropout(positional_encodings)
        return positional_encodings


class LinearEmbedding(nn.Module):
    '''
    project sequence and RPF density seperately
    '''
    def __init__(self, d_seq, d_count, output_model):
        super().__init__()
        self.seq_emb_layer = nn.Linear(d_seq, output_model)
        self.count_emb_layer = nn.Linear(d_count, output_model)
        self.unify_emb_layer = nn.Linear(output_model*2, output_model)

    def forward(self, seq_tokens, count_tokens):
        # input (bs, seq_len, input_model) to output (bs, seq_len, output_model)
        seq_embeddings = self.seq_emb_layer(seq_tokens)
        count_embeddings = self.count_emb_layer(count_tokens)
        embeddings = self.unify_emb_layer(torch.cat([seq_embeddings, count_embeddings], axis=-1))

        return embeddings


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # often d_ff > d_model
        self.linear2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()

    def forward(self, reps_batch):
        return self.linear2(self.gelu(self.linear1(reps_batch))) # (bs, seq_len, d_model)


class AddNormLayer(nn.Module):
    # LayerNorm -> sublayer (MHA or FFN) -> dropout -> residual connection
    def __init__(self, d_model, p_drop):
        super().__init__()
        self.LN = nn.LayerNorm(d_model) # normalized to mean 0 and variance 1 for (seq_len, model dimension)
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, reps_batch, sublayer_module):
        # any modules could be packaged by uniform interface (sublayer_module)
        # residual connections and normalization layer
        return reps_batch + self.dropout(
            sublayer_module(self.LN(reps_batch))
            ) # (bs, seq_len, d_model)


class AddAdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization with Gating (adaLN-Zero).
    Replaces AddNormLayer.
    Structure: Residual +  Dropout(Sublayer(gamma * LN(x) + beta))
    """
    def __init__(self, d_model, p_drop, num_classes):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=p_drop)
        
        # Standard LayerNorm without learnable parameters (elementwise_affine=False)
        self.LN = nn.LayerNorm(d_model, elementwise_affine=False)
        
        # Embedding to generate gamma (scale), beta (shift) 
        # Input: cell_id -> Output: 2 * d_model (split into gamma, beta)
        self.embed = nn.Embedding(num_classes, d_model * 2) 
        
        # Initialize embedding weights
        # Zero-centered initialization for stability (adaLN-Zero)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

    def forward(self, reps_batch, sublayer_module, cell_id):
        """
        Args:
            reps_batch: (bs, seq_len, d_model)
            sublayer_module: lambda function or module (e.g., MHA or FFN)
            cell_id: (bs,) Tensor containing cell type indices
        """
        # 1. Generate adaptive parameters
        style = self.embed(cell_id) # [Batch, 3*Dim]
        # Split into 3 parts
        gamma, beta = style.chunk(2, dim=-1) # Each is [Batch, Dim]
        
        # Expand dimensions for broadcasting: (Batch, 1, Dim)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        
        # 2. Apply Adaptive Norm (Pre-Norm)
        # norm = gamma * LN(x) + beta
        normed = gamma * self.LN(reps_batch) + beta
        
        # 3. Apply Sublayer -> Dropout -> Gating -> Residual
        output = self.dropout(sublayer_module(normed))
        
        return reps_batch + output


class AddAdaZeroLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization with Gating (adaLN-Zero) with Information Bottleneck.
    Structure: Residual + alpha * Dropout(Sublayer((1 + gamma) * LN(x) + beta))
    """
    def __init__(self, d_model, p_drop, num_classes, adaptive_dim=32):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=p_drop)
        
        # Standard LayerNorm without learnable parameters
        self.LN = nn.LayerNorm(d_model, elementwise_affine=False)
        
        # --- 修改点 1：引入 Information Bottleneck ---
        # 第一步：将 Cell ID 映射到低维紧凑空间
        self.cell_embed = nn.Embedding(num_classes, adaptive_dim)
        
        # 第二步：将低维特征投影回需要的调制维度 (d_model * 3)
        # 很多现代架构 (如 DiT) 会在这里加一个非线性激活函数，增强表达能力同时保持信息压缩
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(adaptive_dim, d_model * 3)
        )
        
        # --- 修改点 2：更健壮的 Zero-Initialization ---
        # 我们不再单独清零某一部分，而是直接将最后一步线性层的权重和偏置全部初始化为 0。
        # 这样输出的 gamma, beta, alpha 初始状态全部为 0。
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, reps_batch, sublayer_module, cell_id):
        # 1. Generate adaptive parameters
        compact_style = self.cell_embed(cell_id)            
        style = self.adaLN_modulation(compact_style)        
        
        gamma, beta, alpha = style.chunk(3, dim=-1)         
        
        # ==========================================
        # 限制 gamma 的幅度，防止特征坍塌
        # torch.tanh 将输出限制在 (-1, 1)，乘以 0.7 后限制在 (-0.5, 0.5)
        # 这样 1 + gamma 永远在 (0.5, 1.5) 之间，输入信号绝对不会被抹杀
        # ==========================================
        gamma = torch.tanh(gamma) * 0.5
        
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        alpha = alpha.unsqueeze(1)
        
        # 2. Apply Adaptive Norm (Pre-Norm)
        normed = (1 + gamma) * self.LN(reps_batch) + beta
        
        # 3. Apply Sublayer -> Dropout -> Gating -> Residual
        output = self.dropout(sublayer_module(normed))
        
        # 初始化时 alpha 为 0，所以初始状态是纯净的残差连接
        return reps_batch + (alpha * output)
    
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, heads, p_drop):
        super().__init__()
        # twice "Add & Norm" for one Encoder Layer
        self.sublayers = replicate_module(AddNormLayer(d_model, p_drop), 2)
        self.multi_headed_attention = FlashMultiHeadedAttention(d_model, heads, p_drop) # if flash_attn else MultiHeadedAttention(d_model, heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)

        self.d_model = d_model

    def forward(self, src_reps, src_mask):
        # Define anonymous (lambda) function which only takes src_reps (srb) as input,
        # this way we have a uniform interface for the sublayer logic.
        encoder_self_attention = lambda srb: self.multi_headed_attention(srb, srb, attention_mask=src_mask)

        # Self-attention MHA sublayer followed by point-wise feed forward net sublayer
        ## pre-normalization before MHA and feedforward net sublayer
        src_reps = self.sublayers[0](src_reps, encoder_self_attention)
        src_reps = self.sublayers[1](src_reps, self.ffn)

        return src_reps

class InterleavedEncoderLayer(nn.Module):
    """
    A single layer of the deeply interleaved encoder.
    Implements the logic from Blueprint B:
    1. P-site stream: Self-Attention -> Cross-Attention (querying RNA stream) -> FFN
    2. RNA stream: Self-Attention -> FFN
    """
    def __init__(self, d_model, d_ff, heads, p_drop):
        super().__init__()
        self.d_model = d_model

        # Attention modules
        self.rna_self_attn = FlashMultiHeadedAttention(d_model, heads, p_drop)
        self.psite_self_attn = FlashMultiHeadedAttention(d_model, heads, p_drop)
        self.cross_attn = FlashMultiHeadedAttention(d_model, heads, p_drop)

        # Feed-forward networks
        self.rna_ffn = PositionwiseFeedForward(d_model, d_ff)
        self.psite_ffn = PositionwiseFeedForward(d_model, d_ff)

        # Sublayer connections (Add & Norm)
        # 2 for RNA stream (self-attn, ffn)
        self.sublayers_rna = replicate_module(AddNormLayer(d_model, p_drop), 2)
        # 3 for P-site stream (self-attn, cross-attn, ffn)
        self.sublayers_psite = replicate_module(AddNormLayer(d_model, p_drop), 3)

    def forward(self, rna_reps, psite_reps, rna_mask, psite_mask):
        """
        Processes both RNA and P-site streams for one layer.
        
        Args:
            rna_reps (torch.Tensor): Representations for the RNA modality.
            psite_reps (torch.Tensor): Representations for the P-site modality.
            rna_mask (torch.Tensor): Attention mask for the RNA modality.
            psite_mask (torch.Tensor): Attention mask for the P-site modality.

        Returns:
            Tuple: Updated representations for RNA and P-site.
        """
        # --- 1. RNA Stream Update ---
        # RNA stream updates its own context via self-attention.
        rna_reps_updated = self.sublayers_rna[0](rna_reps, lambda x: self.rna_self_attn(x, x, attention_mask=rna_mask))
        rna_out = self.sublayers_rna[1](rna_reps_updated, self.rna_ffn)

        # --- 2. P-site Stream Update ---
        # a) P-site stream first updates its own context via self-attention.
        psite_reps_self_attended = self.sublayers_psite[0](psite_reps, lambda x: self.psite_self_attn(x, x, attention_mask=psite_mask))

        # b) Then, it queries the RNA stream using cross-attention.
        # RNA is the query, P-site is the key and value.
        cross_attention_func = lambda q: self.cross_attn(rna_reps_updated, q, attention_mask=rna_mask)
        psite_reps_fused = self.sublayers_psite[1](psite_reps_self_attended, cross_attention_func)

        # c) Finally, it passes through its feed-forward network.
        psite_out = self.sublayers_psite[2](psite_reps_fused, self.psite_ffn)

        return rna_out, psite_out

class AdaEncoderLayer(nn.Module):
    """
    Encoder Layer using Adaptive Layer Normalization.
    """
    def __init__(self, d_model, d_ff, heads, p_drop, num_classes):
        super().__init__()
        # Use ModuleList for correct parameter registration
        # Layer 0: For Self-Attention wrapper
        # Layer 1: For FFN wrapper
        self.sublayers = replicate_module(AddAdaLayerNorm(d_model, p_drop, num_classes), 2)
        self.multi_headed_attention = FlashMultiHeadedAttention(d_model, heads, p_drop)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.d_model = d_model

    def forward(self, src_reps, src_mask, cell_id):
        """
        Added cell_id argument.
        """
        # Define anonymous function for self-attention
        encoder_self_attention = lambda srb: self.multi_headed_attention(srb, srb, attention_mask=src_mask)

        # 1. Self-Attention Block with AdaLN
        # Note: We pass cell_id to sublayer
        src_reps = self.sublayers[0](src_reps, encoder_self_attention, cell_id)
        
        # 2. FFN Block with AdaLN
        src_reps = self.sublayers[1](src_reps, self.ffn, cell_id)

        return src_reps


class AdaZeroEncoderLayer(nn.Module):
    """
    Encoder Layer using Adaptive Layer Normalization.
    """
    def __init__(self, d_model, d_ff, heads, p_drop, num_classes, adaptive_dim):
        super().__init__()
        # Use ModuleList for correct parameter registration
        # Layer 0: For Self-Attention wrapper
        # Layer 1: For FFN wrapper
        self.sublayers = replicate_module(AddAdaZeroLayerNorm(d_model, p_drop, num_classes, adaptive_dim), 2)
        self.multi_headed_attention = FlashMultiHeadedAttention(d_model, heads, p_drop)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.d_model = d_model

    def forward(self, src_reps, src_mask, cell_id):
        """
        Added cell_id argument.
        """
        # Define anonymous function for self-attention
        encoder_self_attention = lambda srb: self.multi_headed_attention(srb, srb, attention_mask=src_mask)

        # 1. Self-Attention Block with AdaLN
        # Note: We pass cell_id to sublayer
        src_reps = self.sublayers[0](src_reps, encoder_self_attention, cell_id)
        
        # 2. FFN Block with AdaLN
        src_reps = self.sublayers[1](src_reps, self.ffn, cell_id)

        return src_reps
    

# Part2: =================== Encoder ======================

class Encoder(nn.Module):
    def __init__(self, encoder_layer, number_of_layers):
        super().__init__()
        # multiple Encoder Layers
        self.encoder_layers = replicate_module(encoder_layer, number_of_layers)
        self.LN = nn.LayerNorm(encoder_layer.d_model)

    def forward(self, src_embs, src_mask):
        src_reps = src_embs
        
        for encoder_layer in self.encoder_layers:
            src_reps = encoder_layer(src_reps, src_mask)
        return self.LN(src_reps)

class InterleavedEncoder(nn.Module):
    """
    A stack of InterleavedEncoderLayers to form the complete encoder.
    """
    def __init__(self, encoder_layer, number_of_layers):
        super().__init__()
        self.encoder_layers = replicate_module(encoder_layer, number_of_layers)
        self.LN = nn.LayerNorm(encoder_layer.d_model)

    def forward(self, rna_reps, psite_reps, rna_mask, psite_mask):
        """
        Passes both RNA and P-site streams through all layers.
        """
        for layer in self.encoder_layers:
            rna_reps, psite_reps = layer(rna_reps, psite_reps, rna_mask, psite_mask)
        
        # The final output is the refined P-site representation, which has gathered
        # information from the RNA stream throughout the layers.
        return self.LN(psite_reps)


class AdaEncoder(nn.Module):
    """
    Stack of AdaEncoderLayers.
    """
    def __init__(self, encoder_layer, number_of_layers):
        super().__init__()
        self.encoder_layers = replicate_module(encoder_layer, number_of_layers)
        # Final normalization no needs to be adaptive to maintain cell context
        self.LN = nn.LayerNorm(encoder_layer.d_model)

    def forward(self, src_embs, src_mask, cell_id):
        """
        Added cell_id argument.
        """
        src_reps = src_embs
        
        for encoder_layer in self.encoder_layers:
            src_reps = encoder_layer(src_reps, src_mask, cell_id)
        return self.LN(src_reps)