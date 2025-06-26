import numpy as np
from numpy import random


def mask_random_trinucleotide(seq, tx_idx, embedding, mask_perc, mask_value, motifs_automaton):
    src_len = len(seq)
    start = 0 # frame-shift
    mask_tri_amount = int(src_len * mask_perc // 3) # number of trinucleotide needed to mask
    inverse_embedding_mask = np.ones(src_len, dtype=bool) # mask indicator
    max_src_len = embedding.shape[0]
    d_model = embedding.shape[1]
    masked_embedding = embedding.copy()

    # mask random or mask in-frame trinucleotide
    if tx_idx['cds_start'] == -1:
        tri_len = src_len // 3
    else:
        cds_start = tx_idx['cds_start'] - 1
        start = cds_start % 3 # in-frame
        tri_len = (src_len - start) // 3
    
    # mask trinucleotide by index
    i = random.choice(tri_len, mask_tri_amount, replace=False)
    mask_start = start + i * 3
    mask_end = start + i * 3 + 3

    for n in range(mask_tri_amount):
        perc = random.random()
        if perc < 0.8:
            MASK = mask_value # 80% of the time, replace with [MASK]
        elif perc < 0.9:
            MASK = random.rand(3, d_model) # replace with random embedding
        else:
            MASK = embedding[mask_start[n]:mask_end[n], :] # replace with raw embedding
        # replace with MASK
        masked_embedding[mask_start[n]:mask_end[n], :] = MASK
        inverse_embedding_mask[mask_start[n]:mask_end[n]] = False

    # padding to maximum length
    if max_src_len > src_len:
        pad_mask = np.full(max_src_len - src_len, True)
        inverse_embedding_mask = np.hstack((inverse_embedding_mask, pad_mask))

    return masked_embedding, inverse_embedding_mask