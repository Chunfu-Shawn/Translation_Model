import torch
import torch.nn as nn

class LlamaRotaryEmbeddingExt(nn.Module):
    def __init__(self, dim, max_position_embeddings=30000, base=10000, alpha=8, device=None):
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

    def forward_bk(self, x, seq_len=None):
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

    def get_cos_sin_by_pos(self, positions):
        """
        positions: either
          - int scalar -> returns cos,sin for slice [pos : pos+seq_len] (caller must ensure seq_len if needed),
          - tensor of shape (bs,) -> returns cos: (bs, dim) and sin: (bs, dim)
          - None -> behavior as before (slice 0..seq_len-1 should be handled by calling cos_sin_cached)
        """
        if positions is None:
            raise ValueError("positions must be provided for get_cos_sin_by_pos")
        if isinstance(positions, int):
            pos = positions
            # Return slice starting at pos of length decided by caller. We will let caller slice further.
            return self.cos_cached[:, :, pos:pos+1, :], self.sin_cached[:, :, pos:pos+1, :]
        positions = torch.as_tensor(positions, device=self.cos_cached.device)
        if positions.dim() == 0:
            pos = int(positions.item())
            return self.cos_cached[:, :, pos:pos+1, :], self.sin_cached[:, :, pos:pos+1, :]
        # positions is (bs,)
        # gather per-batch cos/sin for each position
        # cos_cached shape: (1,1,max_seq_len,dim) -> squeeze leading dims to (max_seq_len, dim)
        cos_ = self.cos_cached[0,0]  # (max_seq_len, dim)
        sin_ = self.sin_cached[0,0]
        # positions -> (bs, ) index into first dim
        cos_b = cos_[positions.long(), :]  # (bs, dim)
        sin_b = sin_[positions.long(), :]  # (bs, dim)
        # reshape to (bs, 1, 1, dim) to align with x shape (bs, heads, seq_len, dim)
        cos_b = cos_b.unsqueeze(1).unsqueeze(1)  # (bs,1,1,dim)
        sin_b = sin_b.unsqueeze(1).unsqueeze(1)
        return cos_b, sin_b

    def forward(self, x, positions=None):
        """
        Apply RoPE to x using absolute positions.

        x: (bs, n_heads, seq_len, head_dim)
           note: seq_len may be 1 for single-step decode
        positions:
           - None: fallback to original behavior using 0..seq_len-1 (as before).
           - int: treat as starting absolute position => use positions start..start+seq_len-1 for all batch entries
           - tensor (bs,): per-batch absolute positions for the first token in x (useful for seq_len==1)
           - tensor (bs, seq_len): explicit positions for each element (advanced)
        """
        bs, num_head, seq_len, _ = x.shape
        if positions is None:
            # old behavior: use cos/sin for 0..seq_len-1
            cos, sin = self.cos_sin_cached(seq_len=seq_len)
            # cos/sin shapes: (1,1,seq_len,dim) -> broadcast to x dtype
            cos = cos.to(x.dtype).to(x.device)
            sin = sin.to(x.dtype).to(x.device)
            return x * cos + self.rotate_half(x) * sin

        # positions is provided
        if isinstance(positions, int):
            start = positions
            # slice contiguous block start..start+seq_len-1
            cos = self.cos_cached[:, :, start:start+seq_len, :].to(x.dtype).to(x.device)  # (1,1,seq_len,dim)
            sin = self.sin_cached[:, :, start:start+seq_len, :].to(x.dtype).to(x.device)
            return x * cos + self.rotate_half(x) * sin

        pos_t = torch.as_tensor(positions, device=self.cos_cached.device)
        if pos_t.dim() == 1 and seq_len == 1:
            # per-batch positions for single token: pos_t shape (bs,)
            # get cos/sin per batch: (bs, dim) then reshape to (bs,1,1,dim)
            cos_b, sin_b = self.get_cos_sin_by_pos(pos_t)
            # cos_b shape (bs,1,1,dim) -> need (bs, num_head, 1, dim)
            cos_b = cos_b.expand(-1, num_head, -1, -1).to(x.dtype).to(x.device)
            sin_b = sin_b.expand(-1, num_head, -1, -1).to(x.dtype).to(x.device)
            return x * cos_b + self.rotate_half(x) * sin_b

        if pos_t.dim() == 1 and seq_len > 1:
            # per-batch start positions but multi-token contiguous block: produce cos/sin for each sample
            # we need cos for positions start_i .. start_i+seq_len-1 for each batch element i.
            # Simplest (but slightly less efficient): build cos for each batch by indexing range individually and stack.
            cos_list = []
            sin_list = []
            for i in range(bs):
                s = int(pos_t[i].item())
                cos_i = self.cos_cached[:, :, s:s+seq_len, :].squeeze(0).squeeze(0)  # (seq_len,dim)
                sin_i = self.sin_cached[:, :, s:s+seq_len, :].squeeze(0).squeeze(0)
                cos_list.append(cos_i)
                sin_list.append(sin_i)
            # stack -> (bs, seq_len, dim), then reshape to (bs,1,seq_len,dim) and expand heads
            cos_stack = torch.stack(cos_list, dim=0).unsqueeze(1).to(x.dtype).to(x.device)  # (bs,1,seq_len,dim)
            sin_stack = torch.stack(sin_list, dim=0).unsqueeze(1).to(x.dtype).to(x.device)
            cos_stack = cos_stack.expand(-1, num_head, -1, -1)  # (bs, num_head, seq_len, dim)
            sin_stack = sin_stack.expand(-1, num_head, -1, -1)
            return x * cos_stack + self.rotate_half(x) * sin_stack

        # fallback: unsupported combinations
        raise NotImplementedError("apply_rotary: positions shape not supported")
