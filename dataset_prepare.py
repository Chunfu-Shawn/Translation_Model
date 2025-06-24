import numpy as np
from tqdm import tqdm
from numpy import random
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from torch.utils.data import Dataset
from data_generate_RPF_count import *
from mask_random_single_base import *
from mask_random_trinucleotide import *

__author__ = "Chunfu Xiao"
__contributor__="..."
__copyright__ = ""
__credits__ = []
__license__ = ""
__version__="1.0.0"
__maintainer__ = "Chunfu Xiao"
__email__ = "chunfushawn@126.com"

class BatchDatasetLoader(Dataset):
    MASK = -1
    MASK_PERCENTAGE = 0.15

    def __init__(self, transcript_seq_dict, transcript_index, chroms,
                 batch_size=8, min_length=0, max_length=None, padding_value=-1):
        """
        RNA sequence dataset class, could be grouped and padded by sequence length
        
        :param transcript_seqs: 字典 {transcript_id: sequence_string}
        :param batch_size: 每个批次的大小
        :param min_length: 最小序列长度（过滤短序列）
        :param max_length: 最大序列长度（过滤长序列）
        :param padding_value: 填充值
        """

        self.seq_dict = transcript_seq_dict
        self.tx_idx = transcript_index
        # filter transcript with RPF reads in specific chromosomes
        self.tids = [tid for (tid, v) in transcript_index.items() if (v["chrom"] in chroms)]
        self.length_groups = defaultdict(list)
        self._base_seq_emb = {}
        self._base_pad_mask = {}
        
        # filter and group training transcript
        print("### Data Preparation ###")
        for tid in tqdm(self.tids, desc="Data Processing"):
            seq_len = len(self.seq_dict[tid])
            if seq_len < min_length:
                continue
            if max_length and seq_len > max_length:
                continue
            # seq embedding
            self._base_seq_emb[tid] = self.one_hot_encode(tid)
            # pad mask
            self._base_pad_mask[tid] = np.array([True for _ in range(seq_len)])
            # group
            self.length_groups[seq_len].append(tid)
        self.sorted_length_groups = sorted(self.length_groups.keys())
        # set parameters
        self.batch_size = batch_size
        self.padding_value = padding_value

    def single_base_mask(self, embedding, tid):
        src_len = len(self.seq_dict[tid])
        max_src_len = embedding.shape[0]
        d_model = embedding.shape[1]
        masked_embedding = embedding
        inverse_embedding_mask = np.array([True for _ in range(src_len)])

        mask_amount = round(src_len * self.MASK_PERCENTAGE)
        for _ in range(mask_amount):
            i = random.randint(0, src_len - 1)

            if random.random() < 0.8:
                masked_embedding[i, :] = np.full(d_model, self.MASK) # 80% of the time, replace with [MASK]
            else:
                masked_embedding[i, :] = random.rand(d_model) # replace with random embedding
            inverse_embedding_mask[i] = False

        # padding to maximum length
        if max_src_len > src_len:
            mask_p = np.full(max_src_len - src_len, True)
            inverse_embedding_mask = np.hstack((inverse_embedding_mask, mask_p))

        return masked_embedding, inverse_embedding_mask
    
    def codon_level_mask(self, embedding, tid):
        src_len = len(self.seq_dict[tid])
        max_src_len = embedding.shape[0]
        d_model = embedding.shape[1]
        masked_embedding = embedding
        if self.tx_idx[tid]['cds_start'] == -1:
            inverse_embedding_mask = np.array([True for _ in range(src_len)])

        mask_amount = round(src_len * self.MASK_PERCENTAGE)
        for _ in range(mask_amount):
            i = random.randint(0, src_len - 1)

            if random.random() < 0.8:
                masked_embedding[i, :] = np.full(d_model, self.MASK) # 80% of the time, replace with [MASK]
            else:
                masked_embedding[i, :] = random.rand(d_model) # replace with random embedding
            inverse_embedding_mask[i] = False

        # padding to maximum length
        if max_src_len > src_len:
            mask_p = np.full(max_src_len - src_len, True)
            inverse_embedding_mask = np.hstack((inverse_embedding_mask, mask_p))

        return masked_embedding, inverse_embedding_mask

    def one_hot_encode(self, tid):
        seq = self.seq_dict[tid]
        mapping = dict(zip("ACGT", range(4)))
        seq2 = [mapping[i] for i in seq]
        onehot = np.eye(4)[seq2]

        return onehot
    
    def count_normalized_embedding(self, count_dict, tid, read_len=False):
        seq = self.seq_dict[tid]
        raw_count = np.zeros((len(seq), 10)) # shape (len(seq), 10)
        # save read length information
        if tid in count_dict:
            count = count_dict[tid]
            for read_l in count:
                for pos in count[read_l]:
                    raw_count[pos-1, read_l-25] = count[read_l][pos]
            sum_count = np.sum(raw_count)
            # save read length information
            if read_len:
                nor_count = np.log2(raw_count/(sum_count/np.size(raw_count) + 3) + 1) # shape (len(seq), 10)
            else:
                pos_sum_count = np.sum(raw_count, axis=1, keepdims=True) # shape (len(seq), 1)
                nor_count = np.log2(pos_sum_count/(sum_count/len(seq) + 3) + 1) # shape (len(seq), 1)
        else:
            nor_count = raw_count if read_len else np.zeros((len(seq), 1))

        return nor_count

    def compute_batches(self, count_dict):
        batches = []
        remnant = []
        
        # 对每个长度组进行处理
        for L in tqdm(self.sorted_length_groups, desc="Batch Processing"):
            current_group = remnant + self.length_groups[L]
            # shuffle items for batch
            random.shuffle(current_group)
            remnant = []
            idx = 0
            # 按 batch_size 切分
            while idx + self.batch_size <= len(current_group):
                batch_tids = current_group[idx: idx + self.batch_size]
                idx += self.batch_size
                count_embs = [self.count_normalized_embedding(count_dict, tid, True) for tid in batch_tids]
                seq_embs = [self._base_seq_emb[tid] for tid in batch_tids]
                pad_masks = [self._base_pad_mask[tid] for tid in batch_tids]
                
                # pad seq and count embeddings to L length
                seq_batch = np.stack([
                    np.pad(e,
                        ((0, L - e.shape[0]), (0, 0)),
                        constant_values=self.padding_value
                    ) for e in seq_embs
                ], axis=0)
                count_batch = np.stack([
                    np.pad(e,
                        ((0, L - e.shape[0]), (0, 0)),
                        constant_values=self.padding_value
                    ) for e in count_embs
                ], axis=0)

                # padding mask
                pad_masks_2 = np.stack([
                    np.pad(e, (0, L - len(e)), constant_values=False) for e in pad_masks
                ], axis=0)

                # masked embedding
                combined_batch = np.concatenate((seq_batch, count_batch), axis=2)
                masked, seq_mask = zip(*[
                    random.choice(
                        [mask_random_single_base, mask_random_trinucleotide]
                        )(self.seq_dict[tid], self.tx_idx[tid], combined_batch[i], self.MASK_PERCENTAGE, self.MASK)
                    for i, tid in enumerate(batch_tids)
                    ])
                masked = np.stack(masked, axis=0)
                seq_mask = np.stack(seq_mask, axis=0)
                
                batches.append({
                        'tids': batch_tids, 
                        'length': L, 
                        'seq_embeddings': seq_batch,
                        'count_embeddings': count_batch,
                        'pad_mask': pad_masks_2,
                        'masked_embedding': masked,
                        'seq_mask': seq_mask
                        })
            # 剩余不足 batch_size 的序列，留到下一轮
            remnant = current_group[idx:]

        # drop the last remnant and shuffle all batches
        random.shuffle(batches)
        self.batch_indices = batches

        return batches

if __name__ == "__main__":
    tx_arrays_file = '/home/user/data3/rbase/translation_pred/models/lib/transcript_arrays.pkl'
    RPF_count_file = '/home/user/data3/rbase/translation_pred/models/test/SRR15513158.read_count.pkl'
    tx_seq_file = '/home/user/data3/rbase/translation_pred/models/lib/tx_seq.v48.pkl'
    with open(tx_arrays_file, 'rb') as f:
        tx_arrays = pickle.load(f)
    with open(RPF_count_file, 'rb') as f:
        RPF_count = pickle.load(f)
    with open(tx_seq_file, 'rb') as f:
        tx_seq = pickle.load(f)
    d_model = 4
    batch_size = 8
    print("### Generate batch dataset ###")

    # split dataset
    chrom_train = ["chr" + str(i) for i in range(1,16)] + ["X"]
    chrom_valid = ["chr" + str(i) for i in range(16,21)]
    chrom_test = ["chr" + str(i) for i in range(21,23)] + ["Y"]
    # create dataset
    train_dataset = BatchDatasetLoader(RPF_count, tx_seq, tx_arrays, chrom_train, batch_size, min_length=100)

    max_batch_idx = train_dataset.__len__() - 1
    print(train_dataset.__getitem__(max_batch_idx))
    print(train_dataset.get_batch_seq_embedding(max_batch_idx).shape)
    print(train_dataset.get_batch_count_embedding(max_batch_idx).shape)