import copy
import torch
import torch.nn as nn
from model.flash_multi_headed_attention import FlashMultiHeadedAttention

__author__ = "Chunfu Xiao"
__license__ = ""
__version__="1.0.0"
__maintainer__ = "Chunfu Xiao"
__email__ = "chunfushawn@gmail.com"


def replicate_module(module, copies):
    # deepcopy for independent parameters in different layers
    return nn.ModuleList([copy.deepcopy(module) for _ in range(copies)]) # Module list

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
        self.linear1 = nn.Linear(d_model, d_ff) # often d_ff > d_model
        self.linear2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()

    def forward(self, reps_batch):
        return self.linear2(self.gelu(self.linear1(reps_batch))) # (bs, seq_len, d_model)
    

class MLPEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, p_drop):
        super().__init__()
        # twice "Add & Norm" for one Encoder Layer
        self.sublayers = replicate_module(AddNormLayer(d_model, p_drop), 2)
        self.ffn_1 = nn.Linear(d_model, d_model) # replace self-attention with MLP
        self.ffn_2 = PositionwiseFeedForward(d_model, d_ff)

        self.d_model = d_model

    def forward(self, src_reps, src_mask):
        # Define anonymous (lambda) function which only takes src_reps (srb) as input,

        ## pre-normalization
        src_reps = self.sublayers[0](src_reps, self.ffn_1)
        src_reps = self.sublayers[1](src_reps, self.ffn_2)

        return src_reps
    

class LocalConv(nn.Module):
    def __init__(self, d_model, kernel_size=7, p_drop=0.1):
        """
        Args:
            d_model: 输入特征维度
            kernel_size: 卷积核大小，决定了"局部"的范围。
                         建议设为 3, 5, 7 或 9 来强调局部性。
            p_drop: Dropout 概率
        """
        super().__init__()
        # 计算 padding 以保持序列长度不变 (Seq Len)
        # padding = (kernel_size - 1) / 2
        assert kernel_size % 2 == 1, "Kernel size must be odd to maintain sequence length easily."
        padding = (kernel_size - 1) // 2
        
        # 1D 卷积: 捕捉局部依赖
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=padding
        )
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, d_model)
        
        # Conv1d 需要输入形状为 (batch_size, d_model, seq_len)
        # 所以我们需要交换维度 1 和 2
        x = x.transpose(1, 2)
        
        x = self.conv1d(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # 还原形状回 (batch_size, seq_len, d_model)
        x = x.transpose(1, 2)
        
        # 注意：卷积是局部操作，通常不需要像 Attention 那样使用 mask 来屏蔽未来信息或 padding。
        # 且在 Encoder 中（非自回归），通常不使用 causal mask。
        # 如果需要严格屏蔽 padding token 的影响，可以在这里应用 src_mask，
        # 但对于 ablation 实验，标准卷积已经足够展示"无长程相关性"的特点。
        
        return x

class ConvEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, kernel_size, p_drop):
        """
        Args:
            kernel_size: 控制局部感受野的大小，这是 ablation 的关键参数。
        """
        super().__init__()
        # 保持与原 EncoderLayer 一致的 Add & Norm 结构
        # 假设 replicate_module 和 AddNormLayer 在你的上下文中已定义
        self.sublayers = replicate_module(AddNormLayer(d_model, p_drop), 2)
        
        # [关键修改] 使用局部 CNN 替换 Multi-Head Attention
        # 移除了 heads 参数，新增了 kernel_size 参数
        self.cnn_module = LocalConv(d_model, kernel_size=kernel_size, p_drop=p_drop)
        
        # FFN 保持不变，用于特征变换
        self.ffn = PositionwiseFeedForward(d_model, d_ff)

        self.d_model = d_model

    def forward(self, src_reps, src_mask):
        # 定义 CNN 操作的 lambda 函数，保持接口一致性
        # 虽然 CNN 不强依赖 src_mask，但保留接口以兼容 AddNormLayer 的调用方式
        encoder_cnn_op = lambda srb: self.cnn_module(srb, mask=src_mask)

        # Sublayer 1: CNN (Token Mixing) -> 替换了 Self-Attention
        src_reps = self.sublayers[0](src_reps, encoder_cnn_op)
        
        # Sublayer 2: FFN (Channel Mixing) -> 保持不变
        src_reps = self.sublayers[1](src_reps, self.ffn)

        return src_reps

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