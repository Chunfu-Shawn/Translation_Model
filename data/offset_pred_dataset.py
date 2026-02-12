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

class OffsetPredDataset(Dataset):
    """
    Support two mothods to create:
    - directly provide dataset_dict (from memory)
    - 使用类方法 from_pickle / from_h5 从磁盘加载
    """
    def __init__(self, dataset_dict):
        # 期望 dataset_dict 含 keys:
        # "tids","embs","masked_embs","emb_masks","coding_embs","cell_type_idxs"
        self.read_length = dataset_dict["read_length"]
        self.enzyme_id = dataset_dict["enzyme_id"]
        self.flanking_seq = dataset_dict["flanking_seq"]     # list of (L, 4)
        self.offset = dataset_dict["offset"]
        self.n_samples = len(self.read_length)
    
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
                "read_length": None,
                "enzyme_id": None,
                "flanking_seq": None,
                "offset": None
            }
            try:
                with h5py.File(path, "r") as f:
                    grp = f["samples"]
                    # [:] is requested, data eager
                    data["read_length"] = np.array(grp["read_length"][:])
                    data["enzyme_id"] = np.array(grp["enzyme_id"][:])
                    data["flanking_seq"] = np.array(grp["flanking_seq"][:])
                    data["offset"] = np.array(grp["offset"][:])
                return cls(data)
            except FileNotFoundError:
                print(f"### Error: No such file: {path} ! ###")
            except Exception as e:
                print(f"### Error reading H5: {e} ###")
        else:
            # Lazy model：将 h5 文件路径记录下来，按需读取（注意：DataLoader 多 worker 时需特殊处理）
            obj = object.__new__(cls)
            obj._h5_path = path
            obj._h5_handle = None
            obj._lazy = True

            # 读取n_samples到内存
            try:
                with h5py.File(path, "r") as f:
                    n_samples = f.attrs.get("n_samples", -1)
            except FileNotFoundError:
                print(f"### Error: No such file: {path} ! ###")
            obj.n_samples = n_samples
            # placeholders; actual data read in __getitem__
            obj.embs = None
            obj.masked_embs = None
            obj.emb_masks = None
            obj.coding_embs = None
            obj.cell_type_idxs = None
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
            grp = self._h5_handle["samples"]
            read_length = int(grp['read_length'][idx])
            enzyme_id = int(grp['enzyme_id'][idx])
            flanking_seq = np.array(grp['flanking_seq'][idx], dtype=np.float32)
            offset = int(grp['offset'][idx])

            return read_length, enzyme_id, flanking_seq, offset

        # otherwise load from memory（self.dataset）
        read_length = int(self.read_length[idx])
        enzyme_id = int(self.enzyme_id[idx])
        flanking_seq = np.array(self.flanking_seq[idx], dtype=np.float32)
        offset = int(self.offset[idx])

        return read_length, enzyme_id, flanking_seq, offset
