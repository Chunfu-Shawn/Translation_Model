import sys, os, time
from typing import Literal
import numpy as np
import pickle
import pandas as pd
from itertools import islice
from tqdm import tqdm
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from collections import defaultdict
sys.path.append("/home/user/data3/rbase/translation_model/models/src")
from data.RPF_counter_v3 import double_nested_zero_defaultdict

__author__ = "Chunfu Xiao"
__version__="1.1.0"
__email__ = "xiaochunfu@126.com"


class PSitePredictor():
    def __init__(self,
                 tree_index_file: str, 
                 transcript_seq_file: str, 
                 transcript_meta_file: str, 
                 transcript_cds_file: str, 
                 chrom_groups: dict = {
                     "train": ["chr" + str(i) for i in range(1,18)] + ["X"],
                     "valid": ["chr" + str(i) for i in range(18,23)] + ["Y"]},
                 all_endonucleases: list = ["RNase I", "MNase"],
                 read_length_limits: tuple = (21, 40),
                 n_thread: int = 50):

        # load data
        self.tree_index, _ = self._load_pickle_file(tree_index_file)
        self.seq_dict, _ = self._load_pickle_file(transcript_seq_file)
        self.tx_meta, _ = self._load_pickle_file(transcript_meta_file)
        self.tx_cds, _ = self._load_pickle_file(transcript_cds_file)

        self.nt_mapping = dict(zip("ACGTN", range(5))) # N means unknown
        self.endonuclease_mapping = dict(zip(all_endonucleases, range(len(all_endonucleases)))) # endonuclease type
        self.endonuclease_1hot = np.eye(len(all_endonucleases))
        self.read_length_range = range(read_length_limits[0], read_length_limits[1] + 1)
        self.read_len_1hot = np.eye(len(self.read_length_range))
        self.n_thread = n_thread

        # Cache sequence onehot
        self._base_seq_emb = {}
        for tid in tqdm(self.seq_dict, desc=f"Cache transcript sequences", mininterval=1000):
            self._base_seq_emb[tid] = self._one_hot_encode(tid)

        # Cache read length -> onehot
        self._read_len_onehot_map = {}
        for i, rl in enumerate(self.read_length_range):
            self._read_len_onehot_map[rl] = self.read_len_1hot[i, :].astype(np.float32)

        # Cache enzyme id -> onehot
        self._enzyme_onehot_map = {eid: self.endonuclease_1hot[eid, :].astype(np.float32) for eid in range(len(all_endonucleases))}
    
        self.tids_train = []
        self.tids_valid = []
        self.cds_start_pos = {}
        self.cds_end_pos = {}

        self.predictor = xgb.Booster(params={"nthread": self.n_thread})
        
        # transcript filter
        n_tx = {"no_start_stop_codon": 0, "overlap_other_gene": 0, "valid": 0}
        for tid, value in tqdm(self.tx_cds.items(), desc=f"Filter transcripts", mininterval=1000):
            
            # print(f"--- {tid} ---")
            # 1. having start or stop codon
            if not value["start_codon"] or not value["stop_codon"]:
                n_tx["no_start_stop_codon"] += 1
                continue

            # 2. filter out start/stop codon overlapping with other genes
            chrom = self.tx_meta[tid]["chrom"]
            strand = self.tx_meta[tid]["strand"]

            ol_transcripts = []
            tis_1base = value["cds_starts"][0] if strand == "+" else value["cds_ends"][0] # start site (genomic coordianate)
            ol_transcripts.extend([iv.data for iv in self.tree_index[chrom][tis_1base]])
            tts_1base = value["cds_ends"][-1] if strand == "+" else value["cds_starts"][-1] # start site (genomic coordianate)
            ol_transcripts.extend([iv.data for iv in self.tree_index[chrom][tts_1base]])

            # transcript (gene) overlapping start or stop codon
            ol_transcripts = set(ol_transcripts)
            ol_genes = set(self.tx_meta[tx]["gene_id"] for tx in ol_transcripts)
            if len(ol_genes) > 1: # start codon overlap other gene
                n_tx["overlap_other_gene"] += 1
                continue

            # finally
            if chrom in chrom_groups["train"]:
                self.tids_train.append(tid)
            elif chrom in chrom_groups["valid"]:
                self.tids_valid.append(tid)
            self.cds_start_pos[tid] = self.tx_cds[tid]["cds_start_pos"] - 1 # 0-based for TIS
            self.cds_end_pos[tid] = self.tx_cds[tid]["cds_end_pos"] - 3 # 0-based for last codon before TTS
            n_tx["valid"] += 1

        print("Number of transcripts for training: ", n_tx)

    def _load_pickle_file(self, file_path):
        (_, filename) = os.path.split(file_path)
        name = filename.split('.')[0]
        with open(file_path, 'rb') as f:
            dict = pickle.load(f)
        
        return dict, name

    def _save_pickle_file(self, data, path):
        # ensure directory exist
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _one_hot_encode(self, tid):
        seq = self.seq_dict[tid].upper()
        seq2 = [self.nt_mapping[i] for i in seq]
        onehot = np.eye(5)[seq2, :4] # [0, 0, 0, 0] indicates N
        return onehot
    
    def _load_predictor(self, model_path):
        if os.path.exists(model_path):
            self.predictor.load_model(model_path)
        else:
            raise FileNotFoundError(f"Model file {model_path} not exists.")
    
    def _prepare_train_datasets(self,
                                tids: list,
                                count_dict: dict,
                                train_ref: Literal["TIS", "TTS"],
                                offset_limits: tuple,
                                five_end_flanking: bool = True,
                                three_end_flanking: bool = True,
                                width: int = 8):
        flanking_list = []
        offset_list = []
        read_len_list = []
        for tid in tqdm(tids, desc=f"Generate train dataset around {train_ref} with offsets {offset_limits}", mininterval=200):
            # only for imformative transcripts
            if tid not in count_dict:
                continue

            # extract flanking sequences
            seq_emb = self._base_seq_emb[tid]
            seq_len = seq_emb.shape[0]
            tis_tts_pos = self.cds_start_pos[tid] if train_ref == "TIS" else self.cds_end_pos[tid] # 0-based
            count = count_dict[tid]

            for pos in count:
                five_end = pos - 1 # 0-based
                # read enrichment in TIS and 3 nt upstream of TTS 
                offset = tis_tts_pos - five_end if train_ref == "TIS" else tis_tts_pos - five_end
                # 1. select reads around TIS or TTS
                if not (offset >= offset_limits[0] and offset <= offset_limits[1]):
                    continue
                for read_l in count[pos]:
                    if read_l not in self._read_len_onehot_map:
                        continue
                    three_end = five_end + read_l - 1 # 0-based
                    # 2. ensure 2 *(window * 2 + 1)
                    if not (five_end - width >= 0 and three_end + width + 1 <= seq_len):
                        continue
                    # multuple RPFs simultaneously
                    cnt = count[pos][read_l]
                    read_len_list.extend([read_l] * cnt)
                    five_end_flanking_seq = seq_emb[five_end - width: five_end + width + 1, :]
                    three_end_flanking_seq = seq_emb[three_end - width: three_end + width + 1, :]
                    # flanking sequences
                    if five_end_flanking and three_end_flanking:
                        flanking_seq = [np.concatenate((five_end_flanking_seq, three_end_flanking_seq), axis=0)] * cnt
                    elif five_end_flanking:
                        flanking_seq = [five_end_flanking_seq] * cnt
                    else:
                        flanking_seq = [three_end_flanking_seq] * cnt
                    flanking_list.extend(flanking_seq)
                    offset_list.extend([offset] * cnt)
                            
        # construct feature matrix
        n_sample = len(flanking_list)
        X_seq = np.array(flanking_list, dtype=np.float32).reshape(n_sample, -1) # (n_reads, seq_feat)
        
        # read len one-hot
        read_len_onehot = np.vstack([self._read_len_onehot_map[rl] for rl in read_len_list]).astype(np.float32)

        X = np.hstack([X_seq, read_len_onehot])
        y = np.array(offset_list, dtype=np.int64) - offset_limits[0]

        return X, y
    
    def _prepare_batch_features(self, count_dict, tids, five_end_flanking=True, three_end_flanking=True):
        """
        Prepare feature matrix for batch data (some tids)
        return: X (n_reads, feat_dim) and metadata (tid_list, pos_list, read_len_list)
        if no qualified reads, return (None, None, None, None)
        """
        width = 8
        flanking_list = []
        tid_list = []
        pos_list = []
        read_len_list = []

        for tid in tids:
            seq_emb = self._base_seq_emb.get(tid)
            if seq_emb is None:
                continue
            seq_len = seq_emb.shape[0]
            count_for_tid = count_dict.get(tid, {})
            # traverse read length and positions
            for pos, read_l_dict in count_for_tid.items():
                five_end = pos - 1 # 0-based
                for read_l in read_l_dict:
                    if read_l not in self._read_len_onehot_map:
                        continue
                    three_end = five_end + read_l - 1 # 0-based
                    # ensure 2 *(window * 2 + 1)
                    if not (five_end - width >= 0 and three_end + width + 1 <= seq_len):
                        continue
                    # flanking sequences
                    five_end_flanking_seq = seq_emb[five_end - width: five_end + width + 1, :]
                    three_end_flanking_seq = seq_emb[three_end - width: three_end + width + 1, :]
                    if five_end_flanking and three_end_flanking:
                        flanking_seq = np.concatenate((five_end_flanking_seq, three_end_flanking_seq), axis=0)
                    elif five_end_flanking:
                        flanking_seq = five_end_flanking_seq
                    else:
                        flanking_seq = three_end_flanking_seq
                    flanking_list.append(flanking_seq.reshape(-1))  # 直接扁平化，便于 vstack
                    tid_list.append(tid)
                    pos_list.append(pos)
                    read_len_list.append(read_l)

        if len(flanking_list) == 0:
            return None, None, None, None

        # construct feature matrix
        n_sample = len(flanking_list)
        X_seq = np.array(flanking_list, dtype=np.float32).reshape(n_sample, -1) # (n_reads, seq_feat)
        # read len one-hot
        read_len_onehot = np.vstack([self._read_len_onehot_map[rl] for rl in read_len_list]).astype(np.float32)

        X = np.hstack([X_seq, read_len_onehot])

        return X, tid_list, pos_list, read_len_list
    

    def train_p_site_predictor(
            self,
            count_dict: dict,
            output_dir: str,
            train_ref: Literal["TIS", "TTS"],
            offset_limits: tuple,
            five_end_flanking: bool = True,
            three_end_flanking: bool = True
            ):
        """
        Train P-site predictor using reads around TIS or TTS
        """

        offset_min = offset_limits[0]

        # 1) dataset for training and valid
        X_train, y_train = self._prepare_train_datasets(self.tids_train, 
                                                        count_dict,
                                                        train_ref, offset_limits,
                                                        five_end_flanking, three_end_flanking)
        X_val, y_val = self._prepare_train_datasets(self.tids_valid, 
                                                    count_dict,
                                                    train_ref, offset_limits,
                                                    five_end_flanking, three_end_flanking)
        d_train = xgb.DMatrix(X_train, label=y_train)
        d_val = xgb.DMatrix(X_val, label=y_val)
        num_classes = int(y_train.max()) + 1

        # 2) model initialize and train
        params = {
            'booster': 'gbtree',
            "eta": 0.15,
            "objective": "multi:softprob",
            "num_class": num_classes,
            "eval_metric": "mlogloss",
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": 'hist',
            "max_depth": 9,
            "min_child_weight": 5,
            "gamma": 0.4,
            "seed": 42,
            "nthread": self.n_thread
        }

        evallist = [(d_train, "train"), (d_val, "eval")]
        self.predictor = xgb.train(params, d_train,
                        num_boost_round=500,
                        evals=evallist,
                        early_stopping_rounds=50,
                        verbose_eval=10)

        # 3) predict & evaluate
        y_pred_prob = self.predictor.predict(d_val)
        y_pred = np.argmax(y_pred_prob, axis=1)
        print("Accuracy:", accuracy_score(y_val, y_pred))
        print(classification_report(y_val, y_pred))
        cm = confusion_matrix(y_val, y_pred)
        pd.DataFrame(cm, index=range(offset_min, offset_min + num_classes), 
                    columns=range(offset_min, offset_min + num_classes)).to_csv(output_dir + "/confusion_matrix.csv", 
                                                                                header=True)
        print("Confusion matrix:\n", cm)

        # 4) save & load
        self.predictor.save_model(output_dir + "/xgboosting_model.best.json")
    
        
    def predict_read_specific_p_site(
            self,
            count_dict_path: str,
            endonuclease: Literal["RNase I", "MNase"],
            offset_limits: tuple,
            output_dir: str,
            model_path: str = None,
            mode: Literal["train", "load"] = "train",
            train_ref: Literal["TIS", "TTS"] = "TIS",
            five_end_flanking: bool = True,
            three_end_flanking: bool = True,
            batch_transcripts: int = 1000,
            max_reads_per_batch: int = 300000
            ):
        """
        predict P sites for batch read data
        - batch_transcripts: 每次从字典取多少个 transcript 做一次特征收集
        - max_reads_per_batch: 当累积到多少 reads 时强制执行一次预测（避免 OOM
        """
        
        p_site_dict = defaultdict(double_nested_zero_defaultdict)
        count_dict, sample_name = self._load_pickle_file(count_dict_path)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if mode == "train":
            self.train_p_site_predictor(
                count_dict, output_dir, train_ref, offset_limits
                )
        elif mode == "load":
            self._load_predictor(model_path)
        else:
            raise ValueError(f"Mode type not found: {mode}")

        # 用 iterator 遍历 keys 并按 batch_transcripts 分组
        keys_iter = iter(count_dict.keys())
        total_tids = len(count_dict)
        processed_tids = 0
        
        with tqdm(total=total_tids, desc="Predicting P-site", mininterval=1) as pbar:
            while True:
                tids_batch = list(islice(keys_iter, batch_transcripts))
                if not tids_batch:
                    break

                # 构建特征并尽可能多做预测，若读数过多则分割
                X, tid_list, pos_list, read_len_list = self._prepare_batch_features(
                    count_dict, tids_batch, five_end_flanking, three_end_flanking)
                if X is None:
                    processed_tids += len(tids_batch)
                    pbar.update(len(tids_batch))
                    continue

                n_samples = X.shape[0]
                # 如果单次 batch 太大，拆成子块按 max_reads_per_batch
                start_idx = 0
                while start_idx < n_samples:
                    end_idx = min(start_idx + max_reads_per_batch, n_samples)
                    subX = X[start_idx:end_idx, :]
                    dmat = xgb.DMatrix(subX)
                    offset_pred_prob = self.predictor.predict(dmat)  # 对子块一次预测
                    offsets = np.argmax(offset_pred_prob, axis=1) + offset_limits[0]

                    # map the prediction to corresponding tid/pos/read_l
                    for i_local, global_idx in enumerate(range(start_idx, end_idx)):
                        tid = tid_list[global_idx]
                        read_l = read_len_list[global_idx]
                        pos = pos_list[global_idx]
                        offset = int(offsets[i_local])
                        seq_len = self._base_seq_emb[tid].shape[0]
                        p_pos = pos + offset
                        if p_pos <= seq_len:
                            p_site_dict[tid][p_pos][read_l] += count_dict[tid][pos][read_l]

                    start_idx = end_idx

                processed_tids += len(tids_batch)
                pbar.update(len(tids_batch))

        # save
        self._save_pickle_file(p_site_dict, os.path.join(output_dir, sample_name + ".p_site_count.pkl"))

        return p_site_dict

if __name__ == "__main__":
    lib_path = "/home/user/data3/rbase/translation_model/models/lib"
    tree_index_file = lib_path + "/genome_index_tree.pkl"
    tx_seq_file = lib_path + '/tx_seq.v48.pkl'
    tx_meta_file = lib_path + '/transcript_meta.pkl'
    tx_cds_file = lib_path + '/transcript_cds.pkl'
    result_dir = "/home/user/data3/rbase/translation_model/results/preprocess/WM902B"

    time_s = time.time()
    preditor = PSitePredictor(
        tree_index_file, tx_seq_file, tx_meta_file, tx_cds_file,
        all_endonucleases = ["RNase I", "MNase"],
        read_length_limits = (21, 40),
        n_thread = 60
    )

    time_m = time.time()
    p_site_count = preditor.predict_read_specific_p_site(
        count_dict_path = "/home/user/data3/yaoc/WM902B.read_count.pkl",
        endonuclease = "MNase",
        output_dir = "/home/user/data3/rbase/translation_model/results/preprocess/WM902B/xgboost",
        mode = "train",
        train_ref = "TTS",
        offset_limits = (8, 16)
    )
    time_e = time.time()
    print(f"[TIMING] load={time_m-time_s:.4f}s train and predict={time_e-time_m:.4f}s")