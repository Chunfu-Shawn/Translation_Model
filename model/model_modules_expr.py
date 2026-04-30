import copy
import torch
import torch.nn as nn
from model.flash_multi_headed_attention import FlashMultiHeadedAttention

__author__ = "Chunfu Xiao"
__license__ = ""
__version__="1.1.0"
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


class LinearEmbedding(nn.Module):
    '''
    Project sequence and RPF density safely with non-linearity and normalization.
    '''
    def __init__(self, d_seq, d_count, output_model, p_drop=0.1):
        super().__init__()
        self.seq_emb_layer = nn.Linear(d_seq, output_model)
        self.count_emb_layer = nn.Linear(d_count, output_model)
        
        # 引入 LayerNorm 防止密集信号淹没稀疏信号
        self.seq_ln = nn.LayerNorm(output_model)
        self.count_ln = nn.LayerNorm(output_model)
        
        self.unify_emb_layer = nn.Sequential(
            nn.Linear(output_model * 2, output_model),
            nn.GELU(), # 打破线性塌陷
            nn.Dropout(p_drop)
        )

    def forward(self, seq_tokens, count_tokens):
        seq_embeddings = self.seq_ln(self.seq_emb_layer(seq_tokens))
        count_embeddings = self.count_ln(self.count_emb_layer(count_tokens))
        
        # 使用 dim=-1 是 PyTorch 更标准的写法 (代替 axis=-1)
        concat_emb = torch.cat([seq_embeddings, count_embeddings], dim=-1)
        return self.unify_emb_layer(concat_emb)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # often d_ff > d_model
        self.linear2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()

    def forward(self, reps_batch):
        return self.linear2(self.gelu(self.linear1(reps_batch))) # (bs, seq_len, d_model)


class AddAdaZeroLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization with Gating (adaLN-Zero) with Information Bottleneck.
    Structure: Residual + alpha * Dropout(Sublayer((1 + gamma) * LN(x) + beta))
    
    Update: 
      - Removed discrete embedding. Now strictly accepts a continuous `compact_style` tensor
        from the global model projector.
    """
    def __init__(self, d_model, p_drop, adaptive_dim=16, gamma_scale=0.2):
        super().__init__()
        self.d_model = d_model
        self.gamma_scale = gamma_scale
        self.dropout = nn.Dropout(p=p_drop)
        
        # Standard LayerNorm without learnable parameters
        self.LN = nn.LayerNorm(d_model, elementwise_affine=False)
        
        # Project the low-dimensional compact style to the required modulation dimension (d_model * 3)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(adaptive_dim, d_model * 3)
        )
        
        # Robust Zero-Initialization:
        # Initialize the final linear layer's weights and biases to 0.
        # This ensures gamma, beta, and alpha start completely neutral.
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, reps_batch, sublayer_module, compact_style):
        """
        Args:
            reps_batch: (bs, seq_len, d_model)
            sublayer_module: Callable (MHA or FFN)
            compact_style: (bs, adaptive_dim) Continuous environment representation
        """
        # 1. Generate adaptive parameters         
        style = self.adaLN_modulation(compact_style)        
        
        gamma, beta, alpha = style.chunk(3, dim=-1)         
        
        # ==========================================
        # Hard Regularization: Constrain gamma amplitude to prevent feature collapse.
        # torch.tanh restricts output to (-1, 1), multiplied by gamma_scale (e.g., 0.2)
        # ensures (1 + gamma) stays safely within (0.8, 1.2).
        # ==========================================
        gamma = torch.tanh(gamma) * self.gamma_scale
        
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        alpha = alpha.unsqueeze(1)
        
        # 2. Apply Adaptive Norm (Pre-Norm)
        normed = (1 + gamma) * self.LN(reps_batch) + beta
        
        # 3. Apply Sublayer -> Dropout -> Gating -> Residual
        output = self.dropout(sublayer_module(normed))
        
        # During initialization, alpha is 0, keeping the residual stream clean
        return reps_batch + (alpha * output)
    

class AdaZeroEncoderLayer(nn.Module):
    """
    Encoder Layer using Adaptive Layer Normalization.
    """
    def __init__(self, d_model, d_ff, heads, p_drop, adaptive_dim, gamma_scale):
        super().__init__()
        # Use ModuleList for correct parameter registration
        # Layer 0: For Self-Attention wrapper
        # Layer 1: For FFN wrapper
        self.sublayers = replicate_module(
            AddAdaZeroLayerNorm(d_model, p_drop, adaptive_dim, gamma_scale), 2
        )
        self.multi_headed_attention = FlashMultiHeadedAttention(d_model, heads, p_drop)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.d_model = d_model

    def forward(self, src_reps, src_mask, compact_style):
        """
        Passes the continuous `compact_style` to the AdaLN sublayers.
        """
        # Define anonymous function for self-attention
        encoder_self_attention = lambda srb: self.multi_headed_attention(srb, srb, attention_mask=src_mask)

        # 1. Self-Attention Block with AdaLN
        src_reps = self.sublayers[0](src_reps, encoder_self_attention, compact_style)
        
        # 2. FFN Block with AdaLN
        src_reps = self.sublayers[1](src_reps, self.ffn, compact_style)

        return src_reps
    

# Part2: =================== Encoder ======================

class AdaEncoder(nn.Module):
    """
    Stack of AdaEncoderLayers.
    """
    def __init__(self, encoder_layer, number_of_layers):
        super().__init__()
        self.encoder_layers = replicate_module(encoder_layer, number_of_layers)
        # Final normalization no needs to be adaptive to maintain cell context
        self.LN = nn.LayerNorm(encoder_layer.d_model)

    def forward(self, src_embs, src_mask, compact_style):
        """
        Propagates the `compact_style` context through all encoder layers.
        """
        src_reps = src_embs
        
        for encoder_layer in self.encoder_layers:
            src_reps = encoder_layer(src_reps, src_mask, compact_style)
            
        return self.LN(src_reps)