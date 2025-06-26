import numpy as np
from numpy import random


def mask_random_single_base(seq, tx_idx, embedding, mask_perc, mask_value, motifs_automaton):
    """
    Single-base masking for RNA sequences
    
    Args:
        seq (str): RNA sequence
        tx_idx (dict): Transcript index information
        embedding (np.ndarray): Original embedding matrix (max_src_len, d_model)
        mask_perc (float): Maximum masking percentage (0-1)
        mask_value (float): Value to use for masking
        
    Returns:
        masked_embedding (np.ndarray): Masked embedding matrix
        inverse_embedding_mask (np.ndarray): Boolean mask indicating masked positions
    """

    src_len = len(seq)
    max_src_len = embedding.shape[0]
    d_model = embedding.shape[1]
    masked_embedding = embedding.copy()
    inverse_embedding_mask = np.ones(src_len, dtype=bool)

    mask_amount = round(src_len * mask_perc)
    for _ in range(mask_amount):
        i = random.randint(0, src_len - 1)
        perc = random.random()
        if perc < 0.8:
            MASK = mask_value # 80% of the time, replace with [MASK]
        elif perc < 0.9:
            MASK = random.rand(d_model) # replace with random embedding
        else:
            MASK = embedding[i, :] # replace with raw embedding
        # replace with MASK
        masked_embedding[i, :] = MASK
        inverse_embedding_mask[i] = False

    # padding to maximum length
    if max_src_len > src_len:
        pad_mask = np.full(max_src_len - src_len, True)
        inverse_embedding_mask = np.hstack((inverse_embedding_mask, pad_mask))

    return masked_embedding, inverse_embedding_mask