import torch
import torch.nn as nn
from model.conformer_coding_predictor import CodingPredictorHead
from model.sinusoidal_position_embedding import SinusoidalPositionalEncoding
from model.model_modules import LinearEmbedding, EncoderLayer, Encoder, \
    MaskedSeqPredictorHead, MaskedCountPredictorHead, AddPromptEmbedding, TissueClassificationHead

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

class TranslationModel(nn.Module):
    def __init__(self, d_seq, d_count, d_model, pmt_len, num_tissues, d_ff, heads, number_of_layers, max_seq_len, PE="Sinusoidal", p_drop=0.1):
        super().__init__()
        self.d_input = d_seq + d_count
        self.pmt_len = pmt_len
        self.num_tissues = num_tissues
        # embeds source data into high-dimensional potent embedding vectors
        self.src_emb = LinearEmbedding(self.d_input, d_model)
        self.add_prompt_emb = AddPromptEmbedding(self.pmt_len, self.num_tissues, d_model)

        # Adds positional information to source/target token's embedding vector
        self.PE = PE
        self.src_pos_embedding = SinusoidalPositionalEncoding(d_model, p_drop, max_seq_len)

        # main self-attention architecture
        encoder_layer = EncoderLayer(d_model, d_ff, heads, p_drop)
        self.encoder = Encoder(encoder_layer, number_of_layers)

        # predictor heads
        self.seq_head = MaskedSeqPredictorHead(
            pmt_len = self.pmt_len,
            d_model = d_model,
            d_seq = d_seq,
            d_pred_h = d_ff,
            p_drop = p_drop
        )
        self.count_head = MaskedCountPredictorHead(
            pmt_len = self.pmt_len,
            d_model = d_model,
            d_count = d_count,
            d_pred_h = d_ff,
            p_drop = p_drop
        )
        self.tissue_head = TissueClassificationHead(
            pmt_len = self.pmt_len,
            d_model = d_model,
            num_tissues = self.num_tissues,
            d_pred_h = d_ff
        )
        self.coding_head = CodingPredictorHead(
            d_model = d_model, 
            d_ff = d_ff, 
            heads = 4, 
            num_layers = 2, 
            conv_kernel_size = 31, 
            p_drop = p_drop
            )

        self.init_params()

    def init_params(self, default_initialization=False):
        if not default_initialization:
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, seq_batch, count_batch, tissue_idx, src_mask, head_names=["seq", "count"]):
        src_data_batch = torch.cat((seq_batch, count_batch), axis=-1)
        src_embs = self.src_emb(src_data_batch)  # (bs, seq_len, d_input) -> (bs, seq_len, d_model)
        src_embs, src_mask = self.add_prompt_emb(src_embs, src_mask, tissue_idx) # -> (bs, prompt_len + seq_len, d_model)

        if self.PE == "Sinusoidal":
            src_embs = self.src_pos_embedding(src_embs)
        src_reps = self.encoder(src_embs, src_mask)

        if not head_names:
            print(" #### Must assign a head ! ####")
            return False

        if "coding" in head_names:
            # predict coding ORF
            return self.coding_head(src_reps, src_mask)
        else:
            outputs = {}
            if "seq" in head_names:
                outputs["seq"] = self.seq_head(src_reps)
            if "count" in head_names:
                outputs["count"] = self.count_head(src_reps)
            if "tissue" in head_names:
                outputs["tissue"] = self.tissue_head(src_reps)
            return outputs