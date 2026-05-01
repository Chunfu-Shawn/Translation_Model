import sys
sys.path.append("/home/user/data3/rbase/translation_model/models/src")
import numpy as np
import torch
import pickle
import json
from torch.utils.data import Dataset

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
__version__="1.2.0"
__maintainer__ = "Chunfu Xiao"
__email__ = "xiaochunfu@126.com"


class TranslationDataset(Dataset):
    """
    Support two mothods to create:
    - directly provide dataset_dict (from memory)
    - Load from disk via from_pickle / from_h5
    """
    def __init__(self, dataset_dict):
        self.uuids = dataset_dict["uuids"]
        self.species = dataset_dict["species"]
        self.cell_types = dataset_dict["cell_types"]
        self.cell_expr_dict = dataset_dict.get("cell_expr_dict", {})
        self.meta_info = dataset_dict["meta_info"]
        self.seq_embs = dataset_dict["seq_embs"]     # list of (L, 4)
        self.count_embs = dataset_dict["count_embs"] # list of (L, 1)
        self.lengths = [int(emb.shape[0]) for emb in self.seq_embs]
        self.n_samples = len(self.uuids)
        self.cell_type_counts = dataset_dict.get("cell_type_counts", {})
    
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
                "species": [],
                "cell_types": [],
                "cell_expr_dict": {},
                "meta_info": [],
                "seq_embs": [],
                "count_embs": []
            }
            try:
                with h5py.File(path, "r") as f:
                    try:
                        data["cell_type_counts"] = json.loads(f.attrs.get("cell_type_counts", "{}"))
                    except:
                        data["cell_type_counts"] = {}

                    if "cell_exprs" in f:
                        for ct in f["cell_exprs"].keys():
                            data["cell_expr_dict"][str(ct)] = f["cell_exprs"][ct][:]

                    for uuid in f["samples"].keys():
                        grp = f["samples"][uuid]
                        tid = grp.attrs.get("tid", -1)
                        # [:] is requested, data eager
                        data["uuids"].append(str(uuid))
                        data["species"].append(str(grp.attrs.get("species", None)))
                        data["cell_types"].append(str(grp.attrs.get("cell_type", None)))
                        data["meta_info"].append(
                            {
                                "cds_start_pos": np.int16(grp.attrs.get("cds_start_pos", -1)),
                                "cds_end_pos": np.int16(grp.attrs.get("cds_end_pos", -1)),
                                "motif_occ": list(grp.attrs.get("motif_occ", None)),
                                "rpf_depth": np.float32(grp.attrs.get("rpf_depth", -1)),
                                "rpf_coverage": np.float32(grp.attrs.get("rpf_coverage", -1)),
                                "te_scale": np.float32(grp.attrs.get("te_scale", None))
                            })
                        data["seq_embs"].append(f["sequences"][tid][:]) # (L, 4)
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
            species = []
            cell_types = []
            cell_expr_dict = {}

            try:
                with h5py.File(path, "r") as f:
                    obj.n_samples = f.attrs.get("n_samples", -1)
                    obj.cell_type_counts = json.loads(f.attrs.get("cell_type_counts", "{}"))

                    # [Optimized] Even in lazy mode, we eager-load this tiny dictionary to RAM 
                    # to prevent disk I/O bottlenecks during DataLoader iteration
                    if "cell_exprs" in f:
                        for ct in f["cell_exprs"].keys():
                            cell_expr_dict[str(ct)] = f["cell_exprs"][ct][:]

                    for uuid in f["samples"].keys():
                        grp = f["samples"][uuid]
                        uuids.append(str(uuid))
                        lengths.append(int(grp["count_emb"].shape[0]))
                        species.append(str(grp.attrs.get("species", None)))
                        cell_types.append(str(grp.attrs.get("cell_type", None)))
            except FileNotFoundError:
                print(f"### Error: No such file: {path} ! ###")

            obj.uuids = uuids
            obj.lengths = lengths
            obj.species = species
            obj.cell_types = cell_types
            obj.cell_expr_dict = cell_expr_dict # [Optimized] Store in memory

            # placeholders; actual data read in __getitem__
            obj.seq_embs = None
            obj.count_embs = None
            obj.meta_info = None
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
            species = self.species[idx]
            grp = self._h5_handle["samples"][uuid]
            tid = grp.attrs.get("tid", -1)
            cell_type = str(grp.attrs.get('cell_type', None))
            expr_arr = self.cell_expr_dict.get(cell_type, np.array([]))
            expr_vector = torch.from_numpy(expr_arr).float()

            meta_info = {
                "cds_start_pos": np.int16(grp.attrs.get("cds_start_pos", -1)),
                "cds_end_pos": np.int16(grp.attrs.get("cds_end_pos", -1)),
                "motif_occ": list(grp.attrs.get("motif_occ", None)),
                "rpf_depth": np.float32(grp.attrs.get("rpf_depth", -1)),
                "rpf_coverage": np.float32(grp.attrs.get("rpf_coverage", -1)),
                "te_scale": np.float32(grp.attrs.get("te_scale", None))
                }
            seq_emb = torch.from_numpy(self._h5_handle["sequences"][tid][:]).float() # (L, 4)
            count_emb = torch.from_numpy(grp["count_emb"][:]).float() # (L, 1)
                                   
            return uuid, species, cell_type, expr_vector, meta_info, seq_emb, count_emb

        # otherwise load from memory（self.dataset)
        uuid = str(self.uuids[idx])
        species = str(self.species[idx])
        cell_type = str(self.cell_types[idx])
        expr_arr = self.cell_expr_dict.get(cell_type, np.array([]))
        expr_vector = torch.from_numpy(expr_arr).float()
        meta_info = dict(self.meta_info[idx])
        seq_emb = torch.from_numpy(self.seq_embs[idx]).float()
        count_emb = torch.from_numpy(self.count_embs[idx]).float()

        return uuid, species, cell_type, expr_vector, meta_info, seq_emb, count_emb
    
    def get_identifier(self, idx):
        return self.uuids[idx]