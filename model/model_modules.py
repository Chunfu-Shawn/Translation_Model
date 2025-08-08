import copy
import torch
import torch.nn as nn
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
    def __init__(self, input_model, output_model):
        super().__init__()
        self.embeddings_layer = nn.Linear(input_model, output_model)
        self.sqrt_output_model = torch.sqrt(torch.tensor(output_model))

    def forward(self, tokens):
        # input (bs, seq_len, input_model) to output (bs, seq_len, output_model)
        embeddings = self.embeddings_layer(tokens)
        # Paper P-5, Chapter 3.4 "Embeddings and Softmax": multiply the embedding weights by the square root of d_model
        # embeddings = embeddings * self.sqrt_output_model
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
        encoder_self_attention = lambda srb: self.multi_headed_attention(srb, attention_mask=src_mask)

        # Self-attention MHA sublayer followed by point-wise feed forward net sublayer
        ## pre-normalization before MHA and feedforward net sublayer
        src_reps = self.sublayers[0](src_reps, encoder_self_attention)
        src_reps = self.sublayers[1](src_reps, self.ffn)

        return src_reps

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
        return self.LN(src_reps) # Using LN. not mentioned explicitly in the paper.

# Part3: =================== Output Heads ======================
    
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

class MaskedCountPredictorHead(nn.Module):
    def __init__(self, d_model, d_count, d_pred_h, p_drop=0.1):
        super().__init__()

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

    def forward(self, src_reps):
        # src_reps: [bs, pmt_len + seq_len, d_model], src_mask: [bs, pmt_len + seq_len]
        src_reps = src_reps[:, 3:, :] # -> [bs, seq_len, d_model]

        return self.regress_mlp(src_reps)  # [bs, seq_len, d_count]

class TissueClassificationHead(nn.Module):
    def __init__(self, pmt_len, d_model, num_tissues, d_pred_h, p_drop=0.1):
        super().__init__()
        self.pmt_len = pmt_len # prompt_len
        self.d_model = d_model
        # get flatten prompt tokens to predict tissue type 
        self.classify_mlp = nn.Sequential(
            nn.Linear(self.pmt_len * self.d_model, d_pred_h), # all prompt tokens
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_pred_h, num_tissues)
        )

    def forward(self, encoder_outputs):
        # encoder_outputs: [bs, prompt_len + seq_len, d_model]
        pmt_repr = encoder_outputs[:, :self.pmt_len, :]
        flat_pmt_repr = pmt_repr.view(-1, self.pmt_len * self.d_model) # flat the prompt to one-dim vector
        logits = self.classify_mlp(flat_pmt_repr)

        return logits
