import numpy as np
from numpy import random


def mask_random_motif(seq, tx_idx, embedding, mask_perc, mask_value, motifs_automaton):
    """
    Fast motif masking for RNA sequences using Aho-Corasick algorithm.
    
    Args:
        seq (str): RNA sequence
        tx_idx (dict): Transcript index information
        embedding (np.ndarray): Original embedding matrix (max_src_len, d_model)
        mask_perc (float): Maximum masking percentage (0-1)
        mask_value (float): Value to use for masking
        motifs (list): List of motif strings to mask
        
    Returns:
        masked_embedding (np.ndarray): Masked embedding matrix
        inverse_embedding_mask (np.ndarray): Boolean mask indicating masked positions
    """
    src_len = len(seq)
    d_model = embedding.shape[1]
    max_src_len = embedding.shape[0]
    masked_embedding = embedding.copy()
    inverse_embedding_mask = np.ones(src_len, dtype=bool)
    # If no motifs provided, return original sequence
    if not motifs_automaton:
        return masked_embedding, inverse_embedding_mask
    else:

        # 1. Find all motif occurrences in the sequence
        seq_upper = seq.upper()
        motif_occurrences = []
        for end_idx, motif in motifs_automaton.iter_long(seq_upper):
            start_idx = end_idx - len(motif) + 1
            motif_occurrences.append((start_idx, end_idx + 1))  # (start, end) where end is exclusive
        
        # 2. Calculate mask limits
        max_mask_tokens = int(src_len * mask_perc)
        if not motif_occurrences or max_mask_tokens == 0:
            return masked_embedding, inverse_embedding_mask
        
        # 3. Select motifs to mask without overlaps
        # Randomize order of motif occurrences
        random.shuffle(motif_occurrences)
        
        for i in range(len(motif_occurrences)):
            total_masked = sum([e - s for s, e in motif_occurrences[i:]])
            # Skip if selected motif would not exceed the mask limit
            if total_masked <= max_mask_tokens:
                motif_occurrences = motif_occurrences[i:]
                break
        
        # 4. Apply masking to the embedding
        for start, end in motif_occurrences:
            perc = random.random()
            if perc < 0.8:
                MASK = mask_value # 80% of the time, replace with [MASK]
            elif perc < 0.9:
                MASK = random.rand(end - start, d_model) # 10% replace with random embedding
            else:
                MASK = embedding[start:end, :] # 10% replace with raw embedding

            # replace with MASK
            masked_embedding[start:end, :] = MASK
            inverse_embedding_mask[start:end] = False
    
    # 5. Handle padding if needed
    if max_src_len > src_len:
        pad_mask = np.full(max_src_len - src_len, True)
        inverse_embedding_mask = np.hstack((inverse_embedding_mask, pad_mask))
    
    return masked_embedding, inverse_embedding_mask