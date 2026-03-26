import sys
sys.path.append("/home/user/data3/rbase/translation_model/models/src")
import numpy as np
import torch
from torch.utils.data import Dataset
from data.RPF_counter import *

# optinal
try:
    import h5py
    HAS_H5 = True
except Exception:
    HAS_H5 = False


__author__ = "Chunfu Xiao"
__contributor__="..."
__copyright__ = ""
__credits__ = []
__license__ = ""
__version__="1.1.0"
__maintainer__ = "Chunfu Xiao"
__email__ = "xiaochunfu@126.com"

class CodingDataset(Dataset):
    """
    Support two mothods to create:
    - directly provide dataset_dict (from memory)
    - 使用类方法 from_pickle / from_h5 从磁盘加载
    """
    def __init__(self, dataset_dict):
        # 期望 dataset_dict 含 keys:
        # "tids","embs","masked_embs","emb_masks","cell_type_idxs"
        self.uuids = dataset_dict["uuids"]
        self.cell_types = dataset_dict["cell_types"]
        self.seq_embs = dataset_dict["seq_embs"]     # list of (L, 4)
        self.coding_embs = dataset_dict["coding_embs"] # list of (L, 2)
        self.count_embs = dataset_dict["count_embs"] # list of (L, 1)
        self.lengths = [int(emb.shape[0]) for emb in self.seq_embs]
        self.n_samples = len(self.lengths)
    
    @classmethod
    def from_pickle(cls, path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            # 如果存的是 numpy arrays，就可以直接给构造函数使用
            return cls(data)
        except FileNotFoundError:
            print(f"### Error: No such file: {path} ! ###")
       
    @classmethod
    def from_h5(cls, path, lazy=False):

        if not HAS_H5:
            raise ImportError("h5py not available. Install it or use pickle format.")
        if not lazy:
            # 读取所有样本到内存（safe but memory heavy）
            data = {
                "uuids": [],
                "cell_types": [],
                "seq_embs": [],
                "coding_embs": [],
                "count_embs": []
            }
            try:
                with h5py.File(path, "r") as f:
                    for uuid in f["samples"].keys():
                        grp = f["samples"][uuid]
                        tid = grp.attrs.get("tid", -1)
                        # [:] is requested, data eager
                        data["uuids"].append(str(uuid))
                        data["cell_types"].append(str(grp.attrs.get("cell_type", None)))
                        data["seq_embs"].append(f["sequences"][tid][:]) # (L, 4)
                        data["coding_embs"].append(grp["coding_emb"][:]) # (L, 2)
                        data["count_embs"].append(grp["count_emb"][:]) # (L, 1)
                return cls(data)
            except FileNotFoundError:
                print(f"### Error: No such file: {path} ! ###")
        else:
            # Lazy model：将 h5 文件路径记录下来，按需读取（注意：DataLoader 多 worker 时需特殊处理）
            obj = object.__new__(cls)
            obj._h5_path = path
            obj._h5_handle = None
            obj._lazy = True

            # 读取索引 (idxs + lengths）到内存
            uuids = []
            lengths = []
            try:
                with h5py.File(path, "r") as f:
                    n_samples = f.attrs.get("n_samples", -1)
                    for uuid in f["samples"].keys():
                        uuids.append(str(uuid))
                        lengths.append(int(f["samples"][uuid]["coding_emb"].shape[0]))
            except FileNotFoundError:
                print(f"### Error: No such file: {path} ! ###")
            obj.uuids = uuids
            obj.lengths = lengths
            obj.n_samples = n_samples
            # placeholders; actual data read in __getitem__
            obj.seq_embs = None
            obj.coding_embs = None
            obj.count_embs = None
            obj.cell_types = None
            return obj

    def _open_h5(self):
        # 打开文件句柄（为 DataLoader worker-safe，可以在每个 worker 初始化时调用）
        if getattr(self, "_lazy", False):
            if self._h5_handle is None:
                self._h5_handle = h5py.File(self._h5_path, "r")
    
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # return tensor
        if getattr(self, "_lazy", False):
            # lazy h5 读取
            self._open_h5()
            uuid = self.uuids[idx]
            grp = self._h5_handle["samples"][uuid]
            tid = grp.attrs.get("tid", -1)
            cell_type = str(grp.attrs.get('cell_type', None))
            seq_emb = torch.from_numpy(self._h5_handle["sequences"][tid][:]).float() # (L, 4)
            count_emb = torch.from_numpy(grp["count_emb"][:]).float() # (L, 1)
            coding_emb = torch.from_numpy(grp["coding_emb"][:]).float() # (L, 2)
                                   
            return uuid, cell_type, seq_emb, count_emb, coding_emb

        # otherwise load from memory（self.dataset)
        uuid = str(self.uuids[idx])
        cell_type = str(self.cell_types[idx])
        seq_emb = torch.from_numpy(self.seq_embs[idx]).float()
        count_emb = torch.from_numpy(self.count_embs[idx]).float()
        coding_emb = torch.from_numpy(self.coding_embs[idx]).float()

        return uuid, cell_type, seq_emb, count_emb, coding_emb
    
    def get_identifier(self, idx):
        return self.uuids[idx]