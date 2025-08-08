import copy
import torch
import torch.nn as nn
from model.rotary_position_embedding import LlamaRotaryEmbeddingExt
from model.flash_multi_headed_attention import FlashMultiHeadedAttention
from model.conformer_coding_predictor import CodingPredictorHead

__author__ = "Chunfu Xiao"
__license__ = ""
__version__="1.0.0"
__maintainer__ = "Chunfu Xiao"
__email__ = "chunfushawn@gmail.com"


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, p_drop=None, max_seq_length=20000):
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