import numpy as np
from numpy import random

__author__ = "Chunfu Xiao"
__contributor__="..."
__copyright__ = ""
__credits__ = []
__license__ = ""
__version__="1.0.0"
__maintainer__ = "Chunfu Xiao"
__email__ = "chunfushawn@126.com"


def mask_random_trinucleotide(seq, tx_idx, embedding, mask_perc, mask_value):
    src_len = len(seq)
    start = 0 # frame-shift
    mask_tri_amount = int(src_len * mask_perc // 3) # number of trinucleotide needed to mask
    inverse_embedding_mask = np.array([True for _ in range(src_len)]) # mask indicator
    max_src_len = embedding.shape[0]
    d_model = embedding.shape[1]
    masked_embedding = embedding

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
        if random.random() < 0.8:
            # trinucleotide
            masked_embedding[mask_start[n]:mask_end[n], :] = np.full(d_model, mask_value) # 80% of the time, replace with [MASK]
        else:
            masked_embedding[mask_start[n]:mask_end[n], :] = random.rand(d_model) # replace with random embedding
        inverse_embedding_mask[mask_start[n]:mask_end[n]] = False

    # padding to maximum length
    if max_src_len > src_len:
        mask_p = np.full(max_src_len - src_len, True)
        inverse_embedding_mask = np.hstack((inverse_embedding_mask, mask_p))

    return masked_embedding, inverse_embedding_mask