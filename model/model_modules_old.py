import copy
import torch
import torch.nn as nn
from model.model_modules import EncoderLayer
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

class AddPromptEmbedding(nn.Module):
    # embeds tissue information in prompt
    def __init__(self, pmt_len, num_tissues, d_model):
        super().__init__()
        self.pmt_len = pmt_len # prompt len = pmt_len
        self.d_model = d_model
        self.prompt_emb = nn.Embedding(num_tissues, self.pmt_len * self.d_model)

    def forward(self, src_embs, src_mask, tissue_idx):
        bs = src_embs.size(0)
        p_embs = self.prompt_emb(tissue_idx)
        p_embs = p_embs.view(bs, self.pmt_len, self.d_model)
        x = torch.cat([p_embs, src_embs], dim=1)
        src_mask = torch.cat([torch.zeros(bs, self.pmt_len, device="cuda"), src_mask], dim=1)
        return x, src_mask
    
    
class MaskedEncoderPredictorHead(nn.Module):
    def __init__(self, d_model, d_seq, d_count, d_ff, heads, num_layers, d_pred_h, p_drop=0.1):
        super().__init__()
        # separate transformer blocks
        self.regress_layers = nn.ModuleList([
            EncoderLayer(d_model, d_ff, heads, p_drop)
            for _ in range(num_layers)
        ])
        self.classify_layers = nn.ModuleList([
            EncoderLayer(d_model, d_ff, heads, p_drop)
            for _ in range(num_layers)
        ])

        # count shape: (bs, seq_len, d_count)
        self.regress_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_pred_h),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_pred_h, d_model),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_model, d_count),
            nn.ReLU(),
        )
        # class output shape: (bs, seq_len, d_seq)
        self.classify_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_pred_h),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_pred_h, d_model),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_model, d_seq),
            nn.LogSoftmax(dim=-1)
        )

class MaskedSeqPredictorHead(nn.Module):
    def __init__(self, d_model, d_seq, d_pred_h, p_drop=0.1):
        super().__init__()

        # class output shape: (bs, seq_len, d_seq)
        self.classify_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_pred_h),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_pred_h, d_model),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_model, d_seq),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, src_reps):
        # src_reps: [bs, pmt_len + seq_len, d_model], src_mask: [bs, pmt_len + seq_len]
        src_reps = src_reps[:, 3:, :] # -> [bs, seq_len, d_model]

        return self.classify_mlp(src_reps) # [bs, seq_len, d_seq]
    
      
class TissueClassificationHead(nn.Module):
    def __init__(self, pmt_len, d_model, num_tissues, d_pred_h=None, p_drop=0.1):
        super().__init__()
        self.pmt_len = pmt_len # prompt_len
        self.d_model = d_model
        self.num_tissues = num_tissues
        if d_pred_h is None:
            d_pred_h = d_model
        # get flatten prompt tokens to predict tissue type 
        self.classify_mlp = nn.Sequential(
            nn.Linear(self.pmt_len * self.d_model, d_pred_h), # all prompt tokens
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_pred_h, num_tissues)
        )

    def forward(self, encoder_outputs):
        # encoder_outputs: [bs, prompt_len + seq_len, d_model]
        pmt_reps = encoder_outputs[:, :self.pmt_len, :]
        flat_pmt_repr = pmt_reps.view(-1, self.pmt_len * self.d_model) # flat the prompt to one-dim vector
        logits = self.classify_mlp(flat_pmt_repr)

        return logits
    