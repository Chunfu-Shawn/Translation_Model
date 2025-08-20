import torch
import random
from typing import List, Optional, Union


class BatchMaskingAdapter:
    def __init__(self, 
                 mask_value: float = -1.0,
                 mask_token_vec: Optional[torch.Tensor] = None):
        self.mask_value = mask_value
        self.mask_token_vec = mask_token_vec  # 1-D tensor of size D or None
        
    @staticmethod
    def _mask_amount(seq_len: int, mask_perc: float):
        return int(round(seq_len * mask_perc))

    def mask_random_single_base_batch(self,
                                       embeddings: torch.Tensor,
                                       mask_perc: float = 0.15,
                                       pad_mask: Optional[torch.Tensor] = None,
                                       generator: Optional[torch.Generator] = None):
        """
        randomly mask single-base (for batch)
        embeddings: (B, L, D)
        pad_mask: optional bool tensor (B, L) with True for valid tokens (non-pad). 
                  if provided we only sample from positions where pad_mask==True.
        """
        B, L, D = embeddings.shape
        device = embeddings.device
        g = generator or torch.Generator(device=device)
        masked = embeddings.clone()
        emb_mask = torch.zeros((B, L), dtype=torch.bool, device=device)

        # prepare mask vector / scalar
        use_vec = (self.mask_token_vec is not None)
        if use_vec:
            mask_vec = self.mask_token_vec.to(device)
            assert mask_vec.ndim == 1 and mask_vec.shape[0] == D

        for i in range(B):
            # candidate positions (where pad_mask True) or all positions
            if pad_mask is not None:
                cand = torch.nonzero(pad_mask[i], as_tuple=False).view(-1)
                if cand.numel() == 0:
                    continue
                seq_len = cand.numel()
                k = self._mask_amount(seq_len, mask_perc)
                if k <= 0:
                    continue
                # sample indices from cand
                perm = torch.randperm(seq_len, generator=g, device=device)[:k]
                pos = cand[perm]
            else:
                seq_len = L
                k = self._mask_amount(seq_len, mask_perc)
                if k <= 0:
                    continue
                pos = torch.randperm(seq_len, generator=g, device=device)[:k]

            # decide replacement types
            r = torch.rand(pos.numel(), generator=g, device=device)
            for idx_j, p in enumerate(pos.tolist()):
                rv = r[idx_j].item()
                if rv < 0.8:
                    if use_vec:
                        masked[i, p, :] = mask_vec
                    else:
                        masked[i, p, :].fill_(self.mask_value)
                elif rv < 0.9:
                    masked[i, p, :] = torch.randn((D,), device=device, generator=g)
                else:
                    # keep original (no change)
                    pass
                emb_mask[i, p] = True

        return masked, emb_mask

    def mask_random_trinucleotide_batch(self,
                                        embeddings: torch.Tensor,
                                        cds_starts: Optional[Union[List[int], torch.Tensor]] = None,
                                        mask_perc: float = 0.15,
                                        pad_mask: Optional[torch.Tensor] = None,
                                        generator: Optional[torch.Generator] = None):
        """
        trinucleotide (codon) mask (for batch)
        cds_starts: list/tensor 长度 B, 取 -1 表示无 frame info
        """
        B, L, D = embeddings.shape
        device = embeddings.device
        g = generator or torch.Generator(device=device)
        masked = embeddings.clone()
        emb_mask = torch.zeros((B, L), dtype=torch.bool, device=device)

        if cds_starts is None:
            cds_starts = [-1] * B
        else:
            if isinstance(cds_starts, torch.Tensor):
                cds_starts = [int(x) for x in cds_starts.cpu().tolist()]
            else:
                cds_starts = [int(x) for x in cds_starts]

        use_vec = (self.mask_token_vec is not None)
        if use_vec:
            mask_vec = self.mask_token_vec.to(device)

        for i in range(B):
            # candidate positions must also respect pad_mask if provided
            if pad_mask is not None:
                valid_positions = torch.nonzero(pad_mask[i], as_tuple=False).view(-1)
                if valid_positions.numel() == 0:
                    continue
                # for codon sampling we still consider frame relative to full L; but if padding at tail, tri_len computed from valid_positions
                # simple approach: compute tri candidates by indices that have full codon inside valid_positions
                # We'll fallback to simpler logic: assume pad only at tail and L is true length for codons
                seq_len = valid_positions.numel()
            else:
                seq_len = L

            cds = cds_starts[i]
            if cds == -1:
                start = 0
                tri_len = seq_len // 3
            else:
                start = (cds - 1) % 3
                tri_len = (seq_len - start) // 3
            if tri_len <= 0:
                continue
            mask_tri_amount = int((seq_len * mask_perc) // 3)
            if mask_tri_amount <= 0:
                continue

            # sample tri indices
            if mask_tri_amount >= tri_len:
                p = torch.arange(tri_len, device=device)
            else:
                p = torch.randperm(tri_len, generator=g, device=device)[:mask_tri_amount]

            for pi in p.tolist():
                s = start + int(pi) * 3
                e = s + 3
                rv = torch.rand(1, generator=g, device=device).item()
                if rv < 0.8:
                    if use_vec:
                        masked[i, s:e, :] = mask_vec.unsqueeze(0).expand(e - s, -1)
                    else:
                        masked[i, s:e, :].fill_(self.mask_value)
                elif rv < 0.9:
                    masked[i, s:e, :] = torch.randn((e - s, D), device=device, generator=g)
                else:
                    pass
                emb_mask[i, s:e] = True

        return masked, emb_mask


    def mask_random_motif_batch(self,
                            occs_list: Optional[Union[List[str], List[List[tuple]]]],
                            embeddings: torch.Tensor,
                            mask_perc: float = 0.15,
                            generator: Optional[torch.Generator] = None):
        """
        Accept either:
        - seqs_or_occs is a list of strings (seqs) -> fallback to automaton
        - OR seqs_or_occs is a list of occ-lists [[(s,e),...], ...] -> use them directly
        """
        B, L, D = embeddings.shape
        device = embeddings.device
        g = generator or torch.Generator(device=device)
        masked = embeddings.clone()
        emb_mask = torch.zeros((B, L), dtype=torch.bool, device=device)
        use_vec = (self.mask_token_vec is not None)
        if use_vec:
            mask_vec = self.mask_token_vec.to(device)

        # determine mode
        if occs_list is None:
            # nothing to do
            return masked, emb_mask

        for i in range(B):
            occs = occs_list[i] if occs_list[i] else []
            if not occs:
                continue
            # same selection logic as before (shuffle occs, greedily pick non-overlap until budget)
            idxs = list(range(len(occs)))
            # shuffle index order deterministically via torch.randperm
            if len(idxs) > 1:
                perm = torch.randperm(len(idxs), generator=g).tolist()
                occs_shuf = [occs[j] for j in perm]
            else:
                occs_shuf = occs
            max_mask_tokens = int(L * mask_perc)
            selected = []
            total = 0
            for (s, e) in occs_shuf:
                if total + (e - s) > max_mask_tokens:
                    continue
                # check overlap
                overlap = False
                for ss, ee in selected:
                    if not (e <= ss or s >= ee):
                        overlap = True
                        break
                if overlap:
                    continue
                selected.append((s, e))
                total += (e - s)
                if total >= max_mask_tokens:
                    break
            for (s, e) in selected:
                rv = torch.rand(1, generator=g, device=device).item()
                if rv < 0.8:
                    if use_vec:
                        masked[i, s:e, :] = mask_vec.unsqueeze(0).expand(e - s, -1)
                    else:
                        masked[i, s:e, :].fill_(self.mask_value)
                elif rv < 0.9:
                    masked[i, s:e, :] = torch.randn((e - s, D), device=device, generator=g)
                else:
                    pass
                emb_mask[i, s:e] = True

        return masked, emb_mask

    
    def get_random_masked_batch(self,
                                embeddings: torch.Tensor,
                                cds_starts: Optional[Union[List[int], torch.Tensor]] = None,
                                occs: Optional[List[List[tuple]]] = None,
                                pad_mask: Optional[torch.Tensor] = None,
                                mask_perc: float = 0.15,
                                generator: Optional[torch.Generator] = None,
                                strategy_probs: Optional[dict] = None):
        """
        Assign randomly a mask strategy for each sample in a batch
        Args:
            embeddings: tensor (B,L,D)
            cds_starts: list/tensor len B, only for trinucleotide mask
            occs: optional, list of per-sample motif occurrences (list of [(s,e),...])
            pad_mask: optional BoolTensor(B,L)
            mask_perc: fraction to mask
            generator: torch.Generator
            strategy_probs: dict giving probabilities for each strategy, e.g. {"single":0.5,"tri":0.25,"motif":0.25}
        Returns:
            masked_emb: (B,L,D), emb_mask: (B,L) bool
        """
        B, L, D = embeddings.shape
        device = embeddings.device
        g = generator or torch.Generator(device=device)
        # default probs
        if strategy_probs is None:
            strategy_probs = {"single": 0.4, "tri": 0.3, "motif": 0.3}
        strategies = list(strategy_probs.keys())
        probs = [strategy_probs[s] for s in strategies]

        # sample a strategy per sample (cpu random)
        choices = random.choices(strategies, probs, k=B)

        # group indices per strategy
        idxs_by_strategy = {s: [] for s in strategies}
        for i, c in enumerate(choices):
            idxs_by_strategy[c].append(i)

        # prepare outputs (clone originals)
        masked_emb = embeddings.clone()
        emb_mask = torch.zeros((B, L), dtype=torch.bool, device=device)

        # helper to index-select embeddings for sub-batches and then scatter results back
        def _scatter_subbatch(res_emb_sub, res_mask_sub, indices):
            # res_emb_sub: (n, L, D) ; indices: list of original indices
            for k, orig_idx in enumerate(indices):
                masked_emb[orig_idx] = res_emb_sub[k]
                emb_mask[orig_idx] = res_mask_sub[k]

        # SINGLE
        if idxs_by_strategy.get('single'):
            idxs = idxs_by_strategy['single']
            sub_emb = embeddings[idxs]                # (n, L, D)
            sub_pad = pad_mask[idxs] if pad_mask is not None else None
            sub_out_emb, sub_out_mask = self.mask_random_single_base_batch(
                sub_emb, mask_perc=mask_perc, pad_mask=sub_pad, generator=g)
            _scatter_subbatch(sub_out_emb, sub_out_mask, idxs)

        # TRI
        if idxs_by_strategy.get('tri'):
            idxs = idxs_by_strategy['tri']
            sub_emb = embeddings[idxs]
            sub_cds = None
            if cds_starts is not None:
                if isinstance(cds_starts, torch.Tensor):
                    sub_cds = [int(x) for x in cds_starts[idxs].cpu().tolist()]
                else:
                    sub_cds = [cds_starts[i] for i in idxs]
            sub_pad = pad_mask[idxs] if pad_mask is not None else None
            sub_out_emb, sub_out_mask = self.mask_random_trinucleotide_batch(
                sub_emb, cds_starts=sub_cds, mask_perc=mask_perc, pad_mask=sub_pad, generator=g)
            _scatter_subbatch(sub_out_emb, sub_out_mask, idxs)

        # MOTIF
        if idxs_by_strategy.get('motif'):
            idxs = idxs_by_strategy['motif']
            sub_emb = embeddings[idxs]
            # supply precomputed occs for these indices
            sub_occs = None
            if occs is not None:
                sub_occs = [occs[i] for i in idxs]
            sub_out_emb, sub_out_mask = self.mask_random_motif_batch(
                sub_occs,  # function should accept either seqs or occs
                sub_emb, mask_perc=mask_perc, generator=g)
            _scatter_subbatch(sub_out_emb, sub_out_mask, idxs)

        return masked_emb, emb_mask
