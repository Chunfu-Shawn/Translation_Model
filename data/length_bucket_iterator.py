import numpy as np
from tqdm import tqdm
from numpy import random
from collections import defaultdict
from torch.utils.data import IterableDataset
from data.RPF_counter import *
from data.masking_adapter import *

__author__ = "Chunfu Xiao"
__contributor__="..."
__copyright__ = ""
__credits__ = []
__license__ = ""
__version__="1.0.0"
__maintainer__ = "Chunfu Xiao"
__email__ = "chunfushawn@126.com"


class LengthBucketDataset(IterableDataset):
    def __init__(self, transcript_seq_dict, transcript_index, count_dic, chroms,
                 batch_size=8, min_length=0, max_length=50000, padding_value=-1,
                 mask_perc=0.15, mask_value=-1, motif_file_path=""):
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
        self.count_dict = count_dic
        self.length_groups = defaultdict(list)
        # filter transcript with RPF reads in specific chromosomes
        self.tids = [tid for (tid, v) in transcript_index.items() if (v["chrom"] in chroms)]
        
        # filter and group training transcript
        for tid in tqdm(self.tids, desc="Seqeuence Length Processing", mininterval=1000):
            L = len(transcript_seq_dict[tid])
            if L < min_length and L > max_length:
                continue
             # group
            self.length_groups[L].append(tid)
           
        self.sorted_length_groups = sorted(self.length_groups.keys())
        # set parameters
        self.batch_size = batch_size
        self.masking_adapter = MaskingAdapter(transcript_seq_dict, mask_perc, mask_value, motif_file_path)
        self.padding_value = padding_value


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
    
    def count_embedding(self, tid, read_len=False):
        seq = self.seq_dict[tid]

        # extend to shape (len(seq), 10)
        raw_count = np.zeros((len(seq), 10)) if read_len else np.zeros((len(seq), 1))
        if tid in self.count_dict:
            count = self.count_dict[tid]
            for read_l in count:
                for pos in count[read_l]:
                    raw_count[pos-1, read_l-25] = count[read_l][pos]
            # normalize in a transcript
            return self.count_normalize_transcript(raw_count)
        else:
            return raw_count
    

    def __iter__(self):
        remnant = []
        for L in self.sorted_length_groups:
            current_group = remnant + self.length_groups[L]
            tid_list = self.length_groups[L]
            random.shuffle(tid_list)
            remnant = []
            idx = 0
            # 按 batch_size 切分
            while idx + self.batch_size <= len(current_group):
                batch_tids = current_group[idx: idx + self.batch_size]
                idx += self.batch_size
                yield self._make_batch(batch_tids, L)
            remnant = current_group[idx:]


    def _make_batch(self, batch_tids, L):
        seq_embs = []
        count_embs = []
        pad_masks = []

        for tid in batch_tids:
            seq_embs.append(self.one_hot_encode(tid))
            count_embs.append(self.count_embedding(tid, True))
            pad_masks.append(np.array([True for _ in range(L)]))
        
            # pad seq and count embeddings to L length
            seq_pad_embs = np.stack([
                np.pad(e,
                    ((0, L - e.shape[0]), (0, 0)),
                    constant_values=self.padding_value
                ) for e in seq_embs
            ], axis=0)
            count_pad_embs = np.stack([
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
            combined_embs = np.concatenate((seq_pad_embs, count_pad_embs), axis=2)
            masked, pred_mask = zip(*[
                self.masking_adapter.get_random_mask_function(
                    self.seq_dict[tid], combined_embs[i], self.tx_idx[tid]
                    ) for i, tid in enumerate(batch_tids)
                ])
            masked = np.stack(masked, axis=0)
            pred_mask = np.stack(pred_mask, axis=0)
            
            return {
                'tids': batch_tids, 
                'length': L, 
                'seq_emb': seq_pad_embs,
                'count_emb': count_pad_embs,
                'pad_mask': pad_masks_2,
                'masked_emb': masked,
                'pred_mask': pred_mask
                }