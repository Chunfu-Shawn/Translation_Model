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

def mask_random_single_base(seq, tx_idx, embedding, mask_perc, mask_value):
    src_len = len(seq)
    max_src_len = embedding.shape[0]
    d_model = embedding.shape[1]
    masked_embedding = embedding
    inverse_embedding_mask = np.array([True for _ in range(src_len)])

    mask_amount = round(src_len * mask_perc)
    for _ in range(mask_amount):
        i = random.randint(0, src_len - 1)

        if random.random() < 0.8:
            masked_embedding[i, :] = np.full(d_model, mask_value) # 80% of the time, replace with [MASK]
        else:
            masked_embedding[i, :] = random.rand(d_model) # replace with random embedding
        inverse_embedding_mask[i] = False

    # padding to maximum length
    if max_src_len > src_len:
        mask_p = np.full(max_src_len - src_len, True)
        inverse_embedding_mask = np.hstack((inverse_embedding_mask, mask_p))

    return masked_embedding, inverse_embedding_mask