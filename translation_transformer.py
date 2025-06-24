import copy
import torch
import torch.nn as nn
from RotaryEmbedding import LlamaRotaryEmbeddingExt
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input

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
dff:  inner-layer dimensionality
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
        representations_batch = self.unifyheads(attention_qkv)
        return representations_batch


class FlashMultiHeadedAttention(nn.Module):
    def __init__(self, d_model, heads, dropout=0.0, causal=False):
        """
        d_model: 输入/输出的总维度 (bs, seq_len, model dimension)
        heads: number of heads
        dropout: attention dropout
        causal: whether caual mask
        """
        super().__init__()
        assert d_model % heads == 0
        self.d_model  = d_model
        self.heads  = heads
        self.head_dim   = d_model // heads
        self.RoPE = LlamaRotaryEmbeddingExt(self.head_dim) # for head dim
        self.causal = causal
        self.dropout = dropout

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
        bs = x.shape[0]
        # x shape: (bs, seq_len, n_heads*head_dim)
        # head_dim * n_heads = d_model
        # .view(bs, seq_len, n_heads, head_dim) split d_model to [n_heads, head_dim]
        # .transpose(1,2) -> (bs, n_heads, seq_len, head_dim) for calculating parallelly in muti-heads
        # be careful for attention_mask shape
        query = self.toqueries(x).view(bs, -1, self.heads, self.head_dim).transpose(1, 2)
        key = self.tokeys(x).view(bs, -1, self.heads, self.head_dim).transpose(1, 2)
        value = self.tovalues(x).view(bs, -1, self.heads, self.head_dim).transpose(1, 2)

        if "RoPE":
            query = self.RoPE(query)
            key = self.RoPE(key)
        
        # 1. pack query, key and value
        #    qkv: (bs, n_heads, 3, seq_len, head_dim) -> (bs, seq_len, 3, n_heads, head_dim)
        qkv = torch.stack([query, key, value], dim=2).transpose(1, 3)

        # 2. generate cu_seqlens
        seq_lens = attention_mask.sum(dim=1).to(torch.int32)  # [bs]
        cu_seqlens = torch.cat([
            torch.zeros(1, device=x.device, dtype=torch.int32),
            seq_lens.cumsum(dim=0)], dim=0)  # [bs+1]
        max_seqlen = int(seq_lens.max())

        # 3. remove pad
        #    qkv_unpad: (sum_bs(seq_lens), 3, n_heads, head_dim)
        #    attention_mask: a bool (1/0) tensor of shape (bs, seq_len)
        #    indices: for recover pad
        qkv_unpad, indices, cu_seqlens, max_seqlen, seqused = unpad_input(qkv, attention_mask.to(torch.int))
        
        # 4. use FlashAttention-2 kernel (varlen)
        #    out_unpad: (sum_bs(seq_lens), heads, head_dim)
        attention_qkv_unpad = flash_attn_varlen_qkvpacked_func(
            qkv_unpad.to(torch.bfloat16),
            cu_seqlens,
            max_seqlen,
            dropout_p=self.dropout,
            causal=self.causal
        ).to(torch.float32)

        # 5. recover to (bs, seq_len, heads, head_dim)
        attention_qkv = pad_input(attention_qkv_unpad, indices, bs, max_seqlen)

        # 6. catenate output of multi-heads: (bs, seq_len, n_heads, head_dim) -> (bs, seq_len, n_heads*head_dim)
        attention_qkv = attention_qkv.reshape(bs, -1, self.heads * self.head_dim)
        representations_batch = self.unifyheads(attention_qkv)

        return representations_batch

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
    def __init__(self, d_model, dff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dff) # often dff > d_model
        self.linear2 = nn.Linear(dff, d_model)
        self.gelu = nn.GELU()

    def forward(self, representations_batch):
        return self.linear2(self.gelu(self.linear1(representations_batch))) # (bs, seq_len, d_model)


class AddNormLayer(nn.Module):
    # LayerNorm -> sublayer (MHA or FFN) -> dropout -> residual connection
    def __init__(self, d_model, p_prob):
        super().__init__()
        self.LN = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=p_prob)

    def forward(self, representations_batch, sublayer_module):
        # any modules could be packaged by uniform interface (sublayer_module)
        # residual connections and normalization layer
        return representations_batch + self.dropout(
            sublayer_module(self.LN(representations_batch))
            ) # (bs, seq_len, d_model)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, dff, heads, p_prob, flash_attn=False):
        super().__init__()
        # twice "Add & Norm" for one Encoder Layer
        ## 
        self.sublayers = replicate_module(AddNormLayer(d_model, p_prob), 2)
        self.multi_headed_attention = FlashMultiHeadedAttention(d_model, heads) if flash_attn else MultiHeadedAttention(d_model, heads)
        self.ffn = PositionwiseFeedForward(d_model, dff)

        self.d_model = d_model

    def forward(self, src_representations_batch, src_mask):
        # Define anonymous (lambda) function which only takes src_representations_batch (srb) as input,
        # this way we have a uniform interface for the sublayer logic.
        encoder_self_attention = lambda srb: self.multi_headed_attention(srb, attention_mask=src_mask)

        # Self-attention MHA sublayer followed by point-wise feed forward net sublayer
        ## pre-normalization before MHA and feedforward net sublayer
        src_representations_batch = self.sublayers[0](src_representations_batch, encoder_self_attention)
        src_representations_batch = self.sublayers[1](src_representations_batch, self.ffn)

        return src_representations_batch


class MaskedSeqPredictor(nn.Module):
    def __init__(self, d_model, d_output, dff=1024):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_output)
        self.softplus = nn.Softplus()
        # self.linear2 = nn.Linear(dff, d_output)

    def forward(self, src_representations_batch):
        # src_representations_batch shape: (bs, seq_len, d_model)
        # output shape: (bs, seq_len, d_output)
        return self.softplus(self.linear1(src_representations_batch))

class CodingPredictor(nn.Module):
    def __init__(self, d_model, d_output):
        super().__init__()
        self.linear = nn.Linear(d_model, d_output)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src_representations_batch):
        # src_representations_batch shape: (bs, seq_len, d_model)
        # output shape: (bs, seq_len, d_output)
        return self.log_softmax(self.linear(src_representations_batch))

# Part2: =================== Encoder ======================


class Encoder(nn.Module):
    def __init__(self, encoder_layer, number_of_layers):
        super().__init__()
        # multiple Encoder Layers
        self.encoder_layers = replicate_module(encoder_layer, number_of_layers)
        self.LN = nn.LayerNorm(encoder_layer.d_model)

    def forward(self, src_embeddings_batch, src_mask):
        src_representations_batch = src_embeddings_batch
        
        for encoder_layer in self.encoder_layers:
            src_representations_batch = encoder_layer(src_representations_batch, src_mask)
        return self.LN(src_representations_batch) # Using LN. not mentioned explicitly in the paper.


# Part3: ================== transformer ==================

class EncoderOnlyTransformer(nn.Module):
    def __init__(self, d_input, d_model, dff, heads, number_of_layers, max_seq_len, PE="Sinusoidal", flash_attn=False, p_prob=0.1):
        super().__init__()

        # Embeds source data into high-dimensional potent embedding vectors
        self.src_embedding = LinearEmbedding(d_input, d_model)
        self.PE = PE
        # Adds positional information to source/target token's embedding vector
        self.src_pos_embedding = PositionalEncoding(d_model, p_prob, max_seq_len)
        encoder_layer = EncoderLayer(d_model, dff, heads, p_prob, flash_attn)
        self.encoder = Encoder(encoder_layer, number_of_layers)
        self.predictor = MaskedSeqPredictor(d_model, d_input)

        self.init_params()

    def init_params(self, default_initialization=False):
        if not default_initialization:
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, src_data_batch, src_mask):
        src_embeddings_batch = self.src_embedding(src_data_batch)  # (bs, seq_len, d_input) -> (bs, seq_len, d_model)
        if self.PE == "Sinusoidal":
            src_embeddings_batch = self.src_pos_embedding(src_embeddings_batch)
        src_representations_batch = self.encoder(src_embeddings_batch, src_mask)
        pred_representations_batch = self.predictor(src_representations_batch)

        return pred_representations_batch
