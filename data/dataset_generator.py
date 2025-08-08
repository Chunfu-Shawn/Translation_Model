import sys
sys.path.append("/home/user/data3/rbase/translation_pred/models/src")
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from data.RPF_counter import *
from data.masking_adapter_v2 import *

__author__ = "Chunfu Xiao"
__contributor__="..."
__copyright__ = ""
__credits__ = []
__license__ = ""
__version__="1.0.0"
__maintainer__ = "Chunfu Xiao"
__email__ = "chunfushawn@126.com"

class TranslationDataset(Dataset):
    def __init__(self, tids, 
                 embs, masked_embs, emb_masks,
                 coding_embs, tissue_nums):
        super().__init__()
        self.tids = tids
        self.embs = embs
        self.masked_embs = masked_embs
        self.emb_masks = emb_masks
        self.coding_embs = coding_embs
        self.tissue_nums = tissue_nums
        self.lengths = [emb[0].shape[0] for emb in embs] # emb[0] represent seq embeddings

    def __getitem__(self, idx):
        # cat the embeddings of seq and count
        embs = torch.cat(self.embs[idx], axis=-1)
        masked_embs = torch.cat(self.masked_embs[idx], axis=-1)
        # stack the masks (True indicate masked embs)
        emb_masks = torch.stack(self.emb_masks[idx], axis=1)
        return embs, masked_embs, emb_masks, self.coding_embs[idx], self.tissue_nums[idx]
    
    def __len__(self):
        return len(self.embs)
    
    def get_identifier(self, idx):
        return self.tids[idx]

# generate embeddings and return Dataset class
class DatasetGenerator():
    def __init__(self, transcript_seq_dict, transcript_meta, transcript_cds, chroms, all_tissue_types,
                 min_length=0, max_length=None, mask_perc=[0.15, 0.15], mask_value=-1, motif_file_path=""):
        """
        RNA sequence dataset class, could be grouped and padded by sequence length
        
        :param transcript_seqs: 字典 {transcript_id: sequence_string}
        :param batch_size: 每个批次的大小
        :param min_length: 最小序列长度（过滤短序列）
        :param max_length: 最大序列长度（过滤长序列）
        :param padding_value: 填充值
        """

        self.data = {}
        self.seq_dict = transcript_seq_dict
        self.tx_meta = transcript_meta
        self.tx_cds = transcript_cds
        # filter transcript with RPF reads in specific chromosomes
        self.tids = [tid for (tid, v) in self.tx_meta.items() if (v["chrom"] in chroms)]
        self.masking_adapter = MaskingAdapter(transcript_seq_dict, mask_value, motif_file_path, mask_perc)
        self.nt_mapping = dict(zip("ACGT", range(4)))
        self.all_tissue_types = all_tissue_types
        self.tissue_mapping = dict(
            zip(
                self.all_tissue_types, 
                range(len(self.all_tissue_types))
                )
            )
        self.filtered_tids = []
        self._base_seq_emb = {}
        
        # data cache
        print("### Data Cache ###")
        for tid in tqdm(self.tids, desc="Data Cache", mininterval=1000):
            seq_len = len(self.seq_dict[tid])
            if seq_len < min_length:
                continue
            if max_length and seq_len > max_length:
                continue
            self.filtered_tids.append(tid)
            # seq embedding
            self._base_seq_emb[tid] = self.one_hot_encode(tid)

    def one_hot_encode(self, tid):
        seq = self.seq_dict[tid]
        seq2 = [self.nt_mapping[i] for i in seq]
        onehot = np.eye(4)[seq2]
        return onehot


    def tissue_type_encode(self, tissue_type):

        return self.tissue_mapping[tissue_type]
        

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
        return (asinh_counts + 1) / (np.arcsinh(med_nz) + 1)
    
    def count_embedding(self, count_dict, tid, read_len=False, offset=12):
        # add general offset of 12 to align RPF footprint and sequence

        seq = self.seq_dict[tid]
        # extend to shape (len(seq), 10)
        raw_count = np.zeros((len(seq), 10)) if read_len else np.zeros((len(seq), 1))
        if tid in count_dict:
            count = count_dict[tid]
            for read_l in count:
                for pos in count[read_l]:
                    raw_count[pos-1, read_l-25] = count[read_l][pos]
            # normalize in a transcript
            norm_count = self.count_normalize_transcript(raw_count)
            
            # shift backrward for 12 nt
            return np.concatenate((np.zeros((12, 10)), norm_count[:-12,:]), axis=0)
        else:
            return raw_count
    
    def generate_dataset(self, count_dict, coding_dict=None, tissue_type=None):
        tids = []
        embs = []
        masked_embs = []
        emb_masks = []
        coding_embs = []
        tissue_idxs = []

        # for target transcripts
        for tid in tqdm(self.filtered_tids, desc="Dataset generating", mininterval=200):
            # only for imformative transcripts
            if tid not in count_dict.keys():
                continue
            
            seq_emb = self._base_seq_emb[tid]
            count_emb = self.count_embedding(count_dict, tid, True, 12)
            # combined_emb = np.concatenate((seq_emb, count_emb), axis=-1)

            # mask embedding for learning
            ## True indicate this is a token need to be blocked and predicted
            masked_emb, emb_mask = self.masking_adapter.get_random_mask_function(
                self.seq_dict[tid], 
                [seq_emb, count_emb],
                self.tx_cds[tid]
                )
            
            # tissue embedding
            tissue_idx = self.tissue_type_encode(tissue_type)

            # add data
            tids.append(tid)
            embs.append([torch.from_numpy(e).float() for e in [seq_emb, count_emb]])
            masked_embs.append([torch.from_numpy(e).float() for e in masked_emb])
            emb_masks.append([torch.from_numpy(e).bool() for e in emb_mask])
            coding_embs.append(torch.from_numpy(coding_dict[tid]).float())
            tissue_idxs.append(torch.tensor(tissue_idx, dtype=torch.long))

        # return Dataset class
        return TranslationDataset(tids, embs, masked_embs, emb_masks,
                                  coding_embs, tissue_idxs)
        