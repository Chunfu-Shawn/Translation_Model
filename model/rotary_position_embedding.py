import torch
import torch.nn as nn

class LlamaRotaryEmbeddingExt(nn.Module):
    def __init__(self, dim, max_position_embeddings=16384, base=10000, alpha=8, device=None):
        super().__init__()
        assert dim % 2 == 0
        alpha = alpha
        base = base * alpha ** (dim / (dim-2))
        # two embedding dimension for one group for rotation theta_i
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        # token index t
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        # freqs shape = [seq_len, dim//2]
        freqs = torch.einsum("i,j->ij", t, self.inv_freq) # torch.outer, calculate m*theta 
        # cache shape: [1, 1, seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
    
    def rotate_half(self, x):
        # 将最后一维均分成两半，然后做复数乘i的旋转
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    def cos_sin_cached(self, seq_len=None):
        if seq_len is None:
            return ()
        else:
            return(
                self.cos_cached[:, :, :seq_len, :],
                self.sin_cached[:, :, :seq_len, :]
            )

    def forward(self, x, seq_len=None):
        '''
        x: Tensor of shape [batch size, n_heads, seq_len, head_dim]
        '''
        if seq_len is None:
            seq_len = x.shape[2]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        
        cos = self.cos_cached[:, :, :seq_len, :].to(x.dtype) # [1, 1, seq_len, dim]
        sin = self.sin_cached[:, :, :seq_len, :].to(x.dtype)

        # apply NTW-aware scaled RoPE
        # x * cos + rotate_half(x)*sin
        x_rot = x * cos + self.rotate_half(x) * sin
        return x_rot