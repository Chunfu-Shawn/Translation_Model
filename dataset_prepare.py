import numpy as np
from numpy.random import shuffle
from collections import defaultdict
from torch.utils.data import Dataset
from data_generate_RPF_count import *

__author__ = "Chunfu Xiao"
__contributor__="..."
__copyright__ = ""
__credits__ = []
__license__ = ""
__version__="1.0.0"
__maintainer__ = "Chunfu Xiao"
__email__ = "chunfushawn@126.com"


class PrepareBatchDataset(Dataset):
    def __init__(self, count_dict, transcript_seq_dict, transcript_index, chroms,
                 batch_size=8, min_length=0, max_length=None, padding_value=-1):
        """
        RNA sequence dataset class, could be grouped and padded by sequence length
        
        :param transcript_seqs: 字典 {transcript_id: sequence_string}
        :param batch_size: 每个批次的大小
        :param min_length: 最小序列长度（过滤短序列）
        :param max_length: 最大序列长度（过滤长序列）
        :param padding_value: 填充值
        """

        self.count_dict = count_dict
        self.seq_dict = transcript_seq_dict
        self.tx_index = transcript_index
        # filter transcript with RPF reads in specific chromosomes
        self.tids = [tid for (tid, v) in self.tx_index.items() if (v["chrom"] in chroms and tid in self.count_dict.keys())]
        self.length_groups = defaultdict(list)
        
        # filter and group training transcript
        for tid in self.tids:
            seq_len = len(self.seq_dict[tid])
            if seq_len < min_length:
                continue
            if max_length and seq_len > max_length:
                continue
            self.length_groups[seq_len].append(tid)
        self.sorted_length_groups = sorted(self.length_groups.keys())
        
        # set parameters
        self.batch_size = batch_size
        self.padding_value = padding_value
        
        # precomputate batch index
        self.batch_indices = self._precompute_batches()

    def one_hot_encode(self, seq, pading_max_len=None):
        mapping = dict(zip("ACGT", range(4)))
        seq2 = [mapping[i] for i in seq]
        onehot = np.eye(4)[seq2]
        # padding to maximum length
        if type(pading_max_len) is int and pading_max_len > len(seq):
            pad = np.full((pading_max_len - len(seq), 4), self.padding_value, dtype=np.int8)
            onehot = np.vstack((onehot, pad))
        return onehot

    def count_normalized_embedding(self, count, seq, pading_max_len=None):
        raw_count = np.zeros((len(seq), 10))
        for read_l in count:
            for pos in count[read_l]:
                raw_count[pos-1, read_l-25] = count[read_l][pos]
        pos_sum_count = np.sum(raw_count, axis=1, keepdims=True) # shape (10, len(seq))
        sum_count = np.sum(pos_sum_count)
        nor_count = np.log1p(pos_sum_count/(sum_count/len(seq) + 5))
        # padding to maximum length
        if type(pading_max_len) is int and pading_max_len > len(seq):
            pad = np.full((pading_max_len - len(seq), 1), self.padding_value, dtype=np.int8)
            nor_count = np.vstack((nor_count, pad))
        return nor_count
    
    def _precompute_batches(self):
        """预计算批次索引"""
        batches = []
        remnant = []
        
        # 对每个长度组进行处理
        for L in self.sorted_length_groups:
            current_group = remnant + self.length_groups[L]
            # shuffle items for batch
            np.random.shuffle(current_group)
            remnant = []
            idx = 0
            # 按 batch_size 切分
            while idx + self.batch_size <= len(current_group):
                batch_tids = current_group[idx: idx + self.batch_size]
                idx += self.batch_size
                # one-hot encode
                seq_embedings = np.array([
                    self.one_hot_encode(self.seq_dict[tid], L) for tid in batch_tids
                    ])
                count_embedings = np.array([
                    self.count_normalized_embedding(self.count_dict[tid], self.seq_dict[tid], L) for tid in batch_tids
                    ])
                batches.append({
                        'tids': batch_tids, 
                        'length': L, 
                        'seq_embedings': seq_embedings,
                        'count_embeddings': count_embedings
                        })
            # 剩余不足 batch_size 的序列，留到下一轮
            remnant = current_group[idx:]

        # drop the last remnant
        # shuffle all batches
        np.random.shuffle(batches)
        return batches

    def __len__(self):
        return len(self.batch_indices)
    
    def __getitem__(self, idx):
        batch_info = self.batch_indices[idx]
        
        return batch_info
    
    def get_batch_seq_embedding(self, idx):
        batch_info = self.batch_indices[idx]
        seq_embedings = batch_info['seq_embedings']
        
        return seq_embedings
    
    def get_batch_count_embedding(self, idx):
        batch_info = self.batch_indices[idx]
        count_embeddings = batch_info['count_embeddings']
        
        return count_embeddings

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
    seq = "GACTGTACGTACGTT"
    count = {
        25:{
            1:1, 2:1, 3:10, 5:2
        },
        29:{
            6:5, 9:6, 10:2,
        },
        34:{
            2:1, 5:1, 9:2
        }
    }
    # split dataset
    chrom_train = ["chr" + str(i) for i in range(1,16)] + ["X"]
    chrom_valid = ["chr" + str(i) for i in range(16,21)]
    chrom_test = ["chr" + str(i) for i in range(21,23)] + ["Y"]
    # create dataset
    train_dataset = PrepareBatchDataset(RPF_count, tx_seq, tx_arrays, chrom_train, batch_size, min_length=100)
    val_dataset = PrepareBatchDataset(RPF_count, tx_seq, tx_arrays, chrom_valid, batch_size, min_length=100)
    test_dataset = PrepareBatchDataset(RPF_count, tx_seq, tx_arrays, chrom_test, batch_size, min_length=100)

    max_batch_idx = train_dataset.__len__() - 1
    print(train_dataset.__getitem__(max_batch_idx))
    print(train_dataset.get_batch_seq_embedding(max_batch_idx).shape)
    print(train_dataset.get_batch_count_embedding(max_batch_idx).shape)