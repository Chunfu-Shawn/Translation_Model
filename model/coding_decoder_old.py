import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, List, Tuple, Any
from model.model_modules import replicate_module, AddNormLayer, PositionwiseFeedForward
from model.flash_multi_headed_attention_mode import FlashMultiHeadedAttention

__author__ = "Chunfu Xiao"
__license__ = ""
__version__="1.0.0"
__email__ = "chunfushawn@gmail.com"

ComputeMode = Literal["parallel", "regression"]

# -------------------------
# DecoderLayer: parallel forward + incremental step (self-attn kvcache + cross-attn)
# -------------------------
class DecoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, p_drop):
        super().__init__()
        self.sublayers = replicate_module(AddNormLayer(d_model, p_drop), 3)
        # self-attn is causal (for autoreg), src-attn (cross) is not
        self.trg_multi_headed_attention = FlashMultiHeadedAttention(d_model, n_heads, p_drop, causal=True)
        self.src_multi_headed_attention = FlashMultiHeadedAttention(d_model, n_heads, p_drop, causal=False)
        self.ffn = PositionwiseFeedForward(d_model)
        self.d_model = d_model

    # ---------- parallel training / eval forward ----------
    def forward(self,
                trg_representations_batch: torch.Tensor,
                src_representations_batch: torch.Tensor,
                trg_mask: Optional[torch.Tensor],
                src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Parallel full-sequence forward used in training (uses flash varlen path internally).
        trg_representations_batch: (bs, tgt_len, d_model) already embedded
        src_representations_batch: (bs, src_len, d_model)
        trg_mask / src_mask: (bs, len) float/bool masks where 1 indicates valid tokens
        """
        # sublayer 0: self-attention (parallel)
        decoder_trg_self_attention = lambda trb: self.trg_multi_headed_attention(queries=trb, kv=trb, pad_mask=trg_mask)
        trg = self.sublayers[0](trg_representations_batch, decoder_trg_self_attention)
        # sublayer 1: cross-attention (parallel) - use cross_attend_bmm helper from Flash wrapper
        decoder_src_attention = lambda trb: self.src_multi_headed_attention(queries=trb, kv=src_representations_batch, pad_mask=src_mask)
        trg = self.sublayers[1](trg, decoder_src_attention)
        # sublayer 2: FFN
        trg = self.sublayers[2](trg, self.ffn)
        return trg
    
    # ---------- incremental single-step (decode) ----------
    def incremental_step(self,
                         x_t: torch.Tensor,
                         src_kv_per_layer: Optional[Any],
                         self_k_cache: torch.Tensor,
                         self_v_cache: torch.Tensor,
                         self_cache_seqlens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process one decoding step for this layer.
        x_t: (bs,1,d_model)
        src_kv_per_layer: one of:
            - None : no cross-attention
            - (k_cache_enc, v_cache_enc, cache_seqlens_enc) with shapes (max_seq_len, bs, n_heads, head_dim), (max_seq_len,...), (bs,) :
                -> use flash kvcache read-only cross-attn
        Returns:
            out_t: (bs,1,d_model), updated self_k_cache, self_v_cache, self_cache_seqlens
        """
        bs = x_t.shape[0]

        # 1) self-attention incremental: append new k/v into the layer's self cache and get output for new token
        out_self, self_k_cache, self_v_cache, self_cache_seqlens = self.trg_multi_headed_attention.decode_step_with_kvcache(
            x_t, self_k_cache, self_v_cache, self_cache_seqlens
        )  # out_self: (bs,1,d_model)

        # 2) cross-attention: prefer using encoder-cache via kvcache if provided; else fallback to bmm with precomputed k_enc/v_enc
        if src_kv_per_layer is None:
            out_cross = torch.zeros_like(out_self)
        else:
            # detect kvcache-shaped input: tuple length == 3 and first dim != bs (layout (max_seq_len, bs, n_heads, head_dim))
            if isinstance(src_kv_per_layer, tuple) and len(src_kv_per_layer) == 3 and src_kv_per_layer[0].dim() == 4 and src_kv_per_layer[0].shape[0] == bs:
                k_cache_enc, v_cache_enc, cache_seqlens_enc = src_kv_per_layer
                # project q for cross-attn: shape (bs, n_heads, 1, head_dim)
                q = self.src_multi_headed_attention.toqueries(out_self).view(bs, 1, 
                                                                             self.src_multi_headed_attention.n_heads, 
                                                                             self.src_multi_headed_attention.head_dim).transpose(1,2)
                # apply RoPE for q using absolute position = new token position (self_cache_seqlens - 1)
                pos = (self_cache_seqlens - 1).to(dtype=torch.int64)
                q = self.src_multi_headed_attention.RoPE(q, positions=pos).transpose(1,2)
                # use flash kernel to read-only attend to prefilled encoder cache
                out_cross = self.src_multi_headed_attention.cross_attend_using_kvcache(q, k_cache_enc, v_cache_enc, cache_seqlens_enc, causal=False)
            else:
                raise Exception("incremental_step: src_kv_per_layer has unsupported shape")

        # 3) feed-forward sublayer & return (we mimic AddNorm ordering used in training)
        out_ffn = self.ffn(out_cross)  # (bs,1,d_model)
        return out_ffn, self_k_cache, self_v_cache, self_cache_seqlens


# -------------------------
# Decoder: manage layers, caches, incremental loop
# -------------------------
class Decoder(nn.Module):
    def __init__(self, decoder_layer: DecoderLayer, number_of_layers: int):
        super().__init__()
        self.decoder_layers: nn.ModuleList = replicate_module(decoder_layer, number_of_layers)
        self.LN = nn.LayerNorm(decoder_layer.d_model)
        self.num_layers = number_of_layers
        self.d_model = decoder_layer.d_model

    # parallel full-sequence forward (training)
    def forward(self,
                trg_embeddings_batch: torch.Tensor,
                src_representations_batch: torch.Tensor,
                trg_mask: Optional[torch.Tensor],
                src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = trg_embeddings_batch
        for layer in self.decoder_layers:
            x = layer(x, src_representations_batch, trg_mask, src_mask)
        return self.LN(x)
    
    # create per-layer self-attn kv caches (for incremental decoding)
    def prepare_kv_caches(self, max_decode_len: int, batch_size: int, device: Optional[torch.device] = None):
        device = device or next(self.parameters()).device
        k_caches, v_caches, cache_seqlens = [], [], []
        for layer in self.decoder_layers:
            k_cache, v_cache, cache_seqlen = layer.trg_multi_headed_attention.init_kv_cache(max_decode_len, batch_size, device=device)
            k_caches.append(k_cache)
            v_caches.append(v_cache)
            cache_seqlens.append(cache_seqlen)
        return k_caches, v_caches, cache_seqlens
    
    # alternative: produce encoder caches in kvcache layout for use with flash kvcache cross-attend
    def prepare_encoder_kv_as_caches(self, 
                                     src_representations_batch: torch.Tensor, 
                                     src_mask: Optional[torch.Tensor] = None, 
                                     max_seq_len: Optional[int] = None):
        """
        Return list of (k_cache, v_cache, cache_seqlens) per layer.
        src_mask: (bs, src_len) with 1=valid. If provided, compute cache_seqlens per sample from this mask.
        """
        caches = []
        for layer in self.decoder_layers:
            _, k_enc, v_enc = layer.src_multi_headed_attention._project_qkv(src_representations_batch)
            # apply RoPE to k_enc (encoder positions assumed 0..src_len-1)
            k_enc = layer.src_multi_headed_attention.RoPE(k_enc)
            k_cache, v_cache, cache_seqlens = layer.src_multi_headed_attention.build_kv_cache_from_kv(k_enc, v_enc, max_seq_len=max_seq_len)
            # override cache_seqlens with per-sample lengths if mask provided
            if src_mask is not None:
                # compute lengths (int32) per sample from mask (assume mask 1=valid)
                lengths = src_mask.sum(dim=1).to(dtype=torch.int32, device=cache_seqlens.device)
                # clamp so not exceed max_seq_len
                lengths = torch.clamp(lengths, min=0, max=k_cache.size(0))
                cache_seqlens = lengths
            caches.append((k_cache, v_cache, cache_seqlens))
        return caches
    
    # incremental greedy decode loop (accepts encoder caches or k_enc/v_enc tuples)
    def incremental_decode(self,
                           start_embeddings: torch.Tensor,
                           generator_fn,
                           src_representations_batch: torch.Tensor,
                           src_mask: torch.Tensor,
                           max_steps: int,
                           k_caches: List[torch.Tensor],
                           v_caches: List[torch.Tensor],
                           cache_seqlens: List[torch.Tensor],
                           encoder_kv_caches: Optional[List[Any]] = None,
                           greedy: bool = True) -> torch.Tensor:
        """
        start_embeddings: (bs,1,d_model)
        generator_fn: callable(batch_decoder_output (bs,1,d_model)) -> logits (bs, vocab)
        encoder_kv_caches: optional list per-layer: either (k_enc,v_enc) or (k_cache_enc,v_cache_enc,cache_seqlens_enc)
        """
        outputs = []
        cur_input = start_embeddings  # (bs,1,d_model)

        # if no encoder_kv_caches provided, precompute simple k_enc/v_enc for bmm cross-attn
        if encoder_kv_caches is None:
            encoder_kv_caches = self.prepare_encoder_kv_as_caches(
                src_representations_batch, src_mask= src_mask,
                max_seq_len=src_representations_batch.size[1])

        for _ in range(max_steps):
            x = cur_input
            for li, layer in enumerate(self.decoder_layers):
                k_cache = k_caches[li]
                v_cache = v_caches[li]
                cache_seqlen = cache_seqlens[li]
                src_kv = encoder_kv_caches[li]  # may be (k_enc,v_enc) or (k_cache_enc,v_cache_enc,cache_seqlens_enc)
                x, k_cache, v_cache, cache_seqlen = layer.incremental_step(x, src_kv, k_cache, v_cache, cache_seqlen)
                k_caches[li] = k_cache
                v_caches[li] = v_cache
                cache_seqlens[li] = cache_seqlen
            # x is (bs,1,d_model)
            logits = generator_fn(x)  # (bs, vocab)
            if greedy:
                # generate token class
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                # generate logits
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            outputs.append(next)
            cur_input = self._tokenid_to_embedding(next_id)
        seq = torch.cat(outputs, dim=1)  # (bs, max_steps)
        return seq
    
    # default hook to convert token ids to embedding; user can override or monkeypatch
    def _tokenid_to_embedding(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Default: token_ids (bs,1) -> one-hot (bs,1,3) -> linear embedding via parent's trg_embedding.
        If you have different embedding mechanism, override this method on the Decoder instance.
        """
        raise NotImplementedError("Decoder._tokenid_to_embedding must be monkeypatched by the higher-level decoder (CodingDecoder)")


# -------------------------
# Generator and CodingDecoder (training forward + generate wrapper)
# -------------------------
class Generator(nn.Module):
    def __init__(self, d_model, d_output):
        super().__init__()
        self.linear = nn.Linear(d_model, d_output)

    def forward(self, trg_representations_batch):
        # trg_representations_batch shape: (bs, seq_len, d_model)
        # returns logits (bs, seq_len, d_output)
        return self.linear(trg_representations_batch)


class CodingDecoder(nn.Module):
    def __init__(self, pmt_len: int, 
                 d_output: int, d_model: int, 
                 n_heads: int, number_of_layers: int, p_drop: float, init_xavier: bool = True):
        super().__init__()
        self.name = "CodingDecoder"
        self.pmt_len = pmt_len
        self.d_output = d_output
        base_layer = DecoderLayer(d_model, n_heads, p_drop)
        self.decoder = Decoder(base_layer, number_of_layers)
        self.d_model = d_model
        # embed 3-class one-hot into d_model
        self.trg_embedding = nn.Linear(self.d_output, d_model)
        self.generator = Generator(d_model, self.d_output)
        if init_xavier:
            for n, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    # training forward (parallel)
    def forward(self,
                trg_inputs: torch.Tensor,
                src_representations_batch: torch.Tensor,
                src_mask: Optional[torch.Tensor],
                mode: ComputeMode = "parallel",
                greedy: bool = True) -> torch.Tensor:
        """
        trg_inputs: (bs, tgt_len, 3) one-hot or continuous features for 3 classes
        returns logits: (bs, tgt_len, 3)
        """
        srb = src_representations_batch[:, self.pmt_len:, :]
        bs, max_steps, _ = srb.shape
        if mode == "parallel":
            trg_emb = self.trg_embedding(trg_inputs)  # (bs, tgt_len, d_model)
            if src_mask is None:
                trg_mask = None
            else:
                trg_mask = src_mask # src_mask[:, self,pmt_len:]
            dec_out = self.decoder(trg_emb, srb, trg_mask, src_mask)
            logits = self.generator(dec_out)
            if greedy:
                # generate token class
                seq = logits.argmax(dim=-1, keepdim=True)
            else:
                # generate logits
                seq = F.softmax(logits, dim=-1)
            return seq
        elif mode == "regression":
            # set <start>
            start_token_onehot = torch.ones((bs, 1, self.d_output), dtype=torch.float32)
            seq = self.generate(srb, 
                          src_mask, 
                          start_token_onehot,
                          max_steps,
                          greedy)
            return seq
        else:
            raise Exception("== Mode you provided can't be recognized ==")

    # generation: greedy incremental decode using encoder-as-kv-cache
    def generate(self,
                 src_representations_batch: torch.Tensor,
                 src_mask: torch.Tensor,
                 start_token_onehot: torch.Tensor,
                 max_steps: int,
                 greedy: bool = True) -> torch.Tensor:
        bs = src_representations_batch.size(0)
        device = src_representations_batch.device

        # initial decoder input embedding
        start_emb = self.trg_embedding(start_token_onehot.to(device))  # (bs,1,d_model)

        # prepare per-layer self-attn caches
        k_caches, v_caches, cache_seqlens = self.decoder.prepare_kv_caches(max_steps, bs, device=device)

        # prepare encoder caches in kvcache layout (so cross-attend can use flash kvcache)
        encoder_kv_caches = self.decoder.prepare_encoder_kv_as_caches(src_representations_batch, max_seq_len=src_representations_batch.size(1))

        # provide generator_fn mapping layer output -> logits (bs, vocab)
        def generator_fn(x_out: torch.Tensor) -> torch.Tensor:
            return self.generator(x_out)[:, 0, :]

        # monkeypatch decoder tokenid->embedding hook to use trg_embedding one-hot mapping
        def id2emb(ids: torch.Tensor) -> torch.Tensor:
            bs_local = ids.size(0)
            one_hot = torch.zeros((bs_local, 1, 3), device=ids.device, dtype=torch.float32)
            one_hot.scatter_(2, ids.unsqueeze(-1), 1.0)
            return self.trg_embedding(one_hot)
        self.decoder._tokenid_to_embedding = id2emb

        seq = self.decoder.incremental_decode(start_emb, generator_fn,
                                              src_representations_batch, src_mask,
                                              max_steps, k_caches, v_caches, cache_seqlens,
                                              encoder_kv_caches, greedy=greedy)
        return seq
    
    @classmethod
    def create_from_model(
        cls,
        parent_model: object,
        number_of_layers: int,
        d_output: int,
        pmt_len: Optional[int] = None,
        d_model: Optional[int] = None,
        n_heads: Optional[int] = None,
        p_drop: float = 0.1,
        init_xavier: bool = True,
    ) -> "CodingDecoder":
        """
        Create a CodingDecoder by inferring missing dimensions from `parent_model`.

        parent_model is expected to expose attributes:
          - d_model (model hidden size)

        Parameters: the same as __init__, but any None will be inferred where possible.
        """
        # infer pmt_len
        if pmt_len is None:
            if hasattr(parent_model, "pmt_len"):
                pmt_len = int(getattr(parent_model, "pmt_len"))
            else:
                raise ValueError("pmt_len not provided and parent_model has no attribute 'pmt_len'")
            
        # infer d_model
        if d_model is None:
            if hasattr(parent_model, "d_model"):
                d_model = int(getattr(parent_model, "d_model"))
            else:
                raise ValueError("d_model not provided and parent_model has no attribute 'd_model'")
            
        # infer d_model
        if n_heads is None:
            if hasattr(parent_model, "n_heads"):
                n_heads = int(getattr(parent_model, "n_heads"))
            else:
                raise ValueError("n_heads not provided and parent_model has no attribute 'd_model'")

        # now construct
        return cls(pmt_len=pmt_len, d_output=d_output, d_model=d_model, 
                   n_heads=n_heads, number_of_layers=number_of_layers, p_drop=p_drop, init_xavier=init_xavier)