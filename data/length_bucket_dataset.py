import numpy as np
from tqdm import tqdm
from numpy import random
from collections import defaultdict
from torch.utils.data import Dataset
from data.RPF_counter import *
from data.masking_adapter_tensor import *

__author__ = "Chunfu Xiao"
__contributor__="..."
__copyright__ = ""
__credits__ = []
__license__ = ""
__version__="1.0.0"
__maintainer__ = "Chunfu Xiao"
__email__ = "chunfushawn@126.com"


class LengthBucketDataset(Dataset):
    def __init__(self, transcript_seq_dict, transcript_index, chroms,
                 batch_size=8, min_length=0, max_length=None, padding_value=-1,
                 mask_perc=0.15, mask_value=-1, motif_file_path=""):
        """
        RNA sequence dataset class, could be grouped and padded by sequence length
        
        :param transcript_seqs: 字典 {transcript_id: sequence_string}
        :param batch_size: 每个批次的大小
        :param min_length: 最小序列长度（过滤短序列）
        :param max_length: 最大序列长度（过滤长序列）
        :param padding_value: 填充值
        """

        self.data = []
        self.seq_dict = transcript_seq_dict
        self.tx_idx = transcript_index
        # filter transcript with RPF reads in specific chromosomes
        self.tids = [tid for (tid, v) in transcript_index.items() if (v["chrom"] in chroms)]
        self.masking_adapter = MaskingAdapter(transcript_seq_dict, mask_perc, mask_value, motif_file_path)
        self._base_seq_emb = {}
        self._base_pad_mask = {}
        self.length_groups = defaultdict(list)
        
        # data cache
        print("### Data Cache ###")
        for tid in tqdm(self.tids, desc="Data Cache", mininterval=1000):
            seq_len = len(self.seq_dict[tid])
            if seq_len < min_length:
                continue
            if max_length and seq_len > max_length:
                continue
            # seq embedding
            self._base_seq_emb[tid] = self.one_hot_encode(tid)
            # pad mask
            self._base_pad_mask[tid] = np.array([True for _ in range(seq_len)])
        
        # set parameters
        self.batch_size = batch_size
        self.padding_value = padding_value

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def one_hot_encode(self, tid):
        seq = self.seq_dict[tid]
        mapping = dict(zip("ACGT", range(4)))
        seq2 = [mapping[i] for i in seq]
        onehot = np.eye(4)[seq2]

        return onehot
    
    def count_normalize_transcript(self, counts: np.ndarray):
        # non-zero counts
        nz = counts[counts > 0]

        # median of non-zero counts
        if nz.size > 0:
            med_nz = np.median(nz)
        else:
            return counts
        
        # asinh(x) ≈ x when x small, ≈ log(2x) when x large
        asinh_counts = np.arcsinh(counts)

        # normlize by median value
        return asinh_counts / (np.arcsinh(med_nz) + 1)
    
    def count_embedding(self, count_dict, tid, read_len=False):
        seq = self.seq_dict[tid]

        # extend to shape (len(seq), 10)
        raw_count = np.zeros((len(seq), 10)) if read_len else np.zeros((len(seq), 1))
        if tid in count_dict:
            count = count_dict[tid]
            for read_l in count:
                for pos in count[read_l]:
                    raw_count[pos-1, read_l-25] = count[read_l][pos]
            # normalize in a transcript
            return self.count_normalize_transcript(raw_count)
        else:
            return raw_count
        
    
    def _length_batches(self, tids):
        # bucketing by length
        print("### Length Bucketing ###")

        # only consider filtered transcript
        for tid in tqdm(tids, desc="Length Bucketing", mininterval=1000):
            seq_len = len(self.seq_dict[tid])
            if tid not in self.tids:
                continue
            self.length_groups[seq_len].append(tid)

        # sorted length (short to long)
        return sorted(self.length_groups.keys())
    

    def generate_batch_dataset(self, count_dict, coding_dict=None, tissue_type=None):
        # length bucketing
        sorted_length_groups = self._length_batches(count_dict.keys())

        # generate batch dataset
        batches = []
        remnant = []
        for L in tqdm(sorted_length_groups, desc="Batch Processing", mininterval=200):
            current_group = remnant + self.length_groups[L]
            # shuffle items for batch
            random.shuffle(current_group)
            remnant = []
            idx = 0
            # 按 batch_size 切分
            while idx + self.batch_size <= len(current_group):
                batch_tids = current_group[idx: idx + self.batch_size]
                idx += self.batch_size
                seq_embs = [self._base_seq_emb[tid] for tid in batch_tids]
                count_embs = [self.count_embedding(count_dict, tid, True) for tid in batch_tids]
                # coding_embs = [coding_dict[tid] for tid in batch_tids]
                pad_masks = [self._base_pad_mask[tid] for tid in batch_tids]
                
                # pad embeddings to L length
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
                # coding_batch = np.stack([
                #     np.pad(e,
                #         ((0, L - e.shape[0]), (0, 0)),
                #         constant_values=self.padding_value
                #     ) for e in coding_embs
                # ], axis=0)

                # padding mask
                pad_masks_2 = np.stack([
                    np.pad(e, (0, L - len(e)), constant_values=False) for e in pad_masks
                ], axis=0)

                # masked embedding
                combined_batch = np.concatenate((seq_batch, count_batch), axis=2)
                masked, pred_mask = zip(*[
                    self.masking_adapter.get_random_mask_function(
                        self.seq_dict[tid], combined_batch[i], self.tx_idx[tid]
                        ) for i, tid in enumerate(batch_tids)
                    ])
                masked = np.stack(masked, axis=0)
                pred_mask = np.stack(pred_mask, axis=0)
                
                batches.append({
                        'tids': batch_tids, 
                        'length': L, 
                        'seq_emb': seq_batch,
                        'count_emb': count_batch,
                        # 'coding_emb': coding_batch,
                        'pad_mask': pad_masks_2,
                        'masked_emb': masked,
                        'pred_mask': pred_mask
                        })
            # 剩余不足 batch_size 的序列，留到下一轮
            remnant = current_group[idx:]

        # drop the last remnant and shuffle all batches
        random.shuffle(batches)
        self.data = batches