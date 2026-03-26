import sys, os
import numpy as np
import pickle
import pandas as pd
from typing import Literal, Optional, Dict, List, Tuple
from itertools import islice
from threading import Thread
from multiprocessing import Queue, cpu_count
from queue import Queue
import traceback
from tqdm import tqdm
import lightgbm as lgb
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from collections import defaultdict
sys.path.append("/home/user/data3/rbase/translation_model/models/src")
from data.RPF_counter_v3 import double_nested_zero_defaultdict
from model.eval_RPF_density_TIS_TTS import build_length_position_matrix, build_length_frame_matrix, find_peaks_in_range_all_lengths
from plot.plot_reads_density_TIS_TTS import plot_start_stop_read_length
from plot.plot_reads_periodicity import plot_length_periodicity
from plot.plot_offset_confusion_matrix import plot_offset_confusion_matrix_heatmap


__author__ = "Chunfu Xiao"
__version__="1.1.0"
__email__ = "xiaochunfu@126.com"

    
class PSitePredictor():
    """
    Trains a LightGBM model to predict P-site offsets from Ribosome Profiling data
    and applies it to genome-wide reads.
    """
    def __init__(self,
                 tree_index_file: str, 
                 transcript_seq_file: str, 
                 transcript_meta_file: str, 
                 transcript_cds_file: str, 
                 chrom_groups: Optional[Dict[str, List[str]]] = None,
                 read_length_limits: Tuple[int, int] = (21, 40),
                 n_thread: int = -1):

        print("--- Initializing PSitePredictor ---")

        # Set default chromosome groups if not provided
        if chrom_groups is None:
            chrom_groups = {
                "train": [f"chr{i}" for i in range(1, 18)] + ["X"],
                "valid": [f"chr{i}" for i in range(18, 23)] + ["Y"]
            }

        # Load annotation and sequence data
        self.tree_index, _ = self._load_pickle_file(tree_index_file)
        self.seq_dict, _ = self._load_pickle_file(transcript_seq_file)
        self.tx_meta, _ = self._load_pickle_file(transcript_meta_file)
        self.tx_cds, _ = self._load_pickle_file(transcript_cds_file)
        self.count_dict = None
        self.sample_name = None

        # Configure parameters
        self.nt_mapping = dict(zip("ACGTN", range(5))) # N means unknown
        self.read_length_limits = range(min(read_length_limits), max(read_length_limits) + 1)
        self.n_thread = cpu_count() if n_thread == -1 else min(n_thread, cpu_count())

        # Initialize state attributes
        self.offset_map = None
        self.offset_fix = None
        self.offset_limits = None
        self._feature_names_ = None
        self.predictor = None

        # --- Pre-processing and Caching for performance ---
        self._cache_sequences() # Cache sequence -> onehot
        self._cache_read_length_emb() # Cache read length -> onehot
        self._filter_transcripts(chrom_groups)

    def _load_pickle_file(self, file_path: str) -> Tuple[any, str]:
        """Loads a pickle file and returns its content and base name."""
        print(f"- Loading pickle file: {file_path}")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        name = os.path.basename(file_path).split('.')[0]

        return data, name

    def _save_pickle_file(self, data: any, path: str):
        """Saves data to a pickle file, creating directories if needed."""
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"## Data saved to {path}")

    def _one_hot_encode(self, seq):
        seq2 = np.array([self.nt_mapping[i] for i in seq.upper()])
        onehot = np.eye(5)[seq2, :4] # [0, 0, 0, 0] indicates N
        return onehot
    
    def _cache_sequences(self):
        """Converts transcript nucleotide sequences to integer arrays and caches them."""
        print("- Caching and encoding transcript sequences...")
        self._base_seq_emb = {}
        for tid, seq in tqdm(self.seq_dict.items(), mininterval=5):
            self._base_seq_emb[tid] = self._one_hot_encode(seq)
    
    def _cache_read_length_emb(self):
        self.read_len_onehot_map = {}
        self.read_len_1hot = np.eye(len(self.read_length_limits))
        for i, rl in enumerate(self.read_length_limits):
            self.read_len_onehot_map[rl] = self.read_len_1hot[i, :].astype(np.float32)
    
    def _filter_transcripts(self, chrom_groups: Dict[str, List[str]]):
        """
        Filters transcripts to include only those suitable for training/validation.
        - Must have annotated start and stop codons.
        - TIS/TTS should not overlap with other genes.
        """
        print("- Filtering transcripts based on criteria...")
        self.tids_train: List[str] = []
        self.tids_valid: List[str] = []
        self.cds_start_pos: Dict[str, int] = {}
        self.cds_end_pos: Dict[str, int] = {}
        
        for tid, value in self.tx_cds.items():
            if not value.get("start_codon") or not value.get("stop_codon"):
                continue

            chrom = self.tx_meta[tid]["chrom"]
            strand = self.tx_meta[tid]["strand"]
            
            # Check for gene overlaps at TIS and TTS genomic coordinates
            tis_1base = value["cds_starts"][0] if strand == "+" else value["cds_ends"][0]
            tts_1base = value["cds_ends"][-1] if strand == "+" else value["cds_starts"][-1]
            
            overlapping_genes = set()
            for pos in [tis_1base, tts_1base]:
                overlapping_genes.update(
                    self.tx_meta[iv.data]["gene_id"] for iv in self.tree_index[chrom].at(pos)
                )

            if len(overlapping_genes) <= 1:
                if chrom in chrom_groups["train"]:
                    self.tids_train.append(tid)
                elif chrom in chrom_groups["valid"]:
                    self.tids_valid.append(tid)
                
                self.cds_start_pos[tid] = value["cds_start_pos"] - 1 # 0-based TIS
                self.cds_end_pos[tid] = value["cds_end_pos"] - 3   # 0-based start of last codon

        print(f"## Transcripts filtered: {len(self.tids_train)} for training, {len(self.tids_valid)} for validation.")
    
    def load_model(self, model_path: str, offset_limits: Tuple[int, int]):
        """
        Loads a pre-trained model.
        IMPORTANT: You must provide the same offset_min and offset_max used during training.
        """
        print(f"--- Loading model from {model_path} ---")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.predictor = joblib.load(model_path)
        
        # --- Explicitly set offset range for correct prediction ---
        # if offset_limits was provided, ignore offset_map
        if offset_limits:
            self.offset_limits = range(min(offset_limits), max(offset_limits) + 1)
        else:
            raise ValueError("Either offset_limits or offset_map should be provided !")
        
        if hasattr(self.predictor, 'feature_name_'):
            self._feature_names_ = self.predictor.feature_name_()
        else:
            print("Warning: Could not automatically determine feature names from model.")
        
        print("✔ Model loaded successfully.")
    
    def load_read_count_data(self, 
                             count_dict_path: str,
                             endonuclease = Literal["RNase I", "MNase"]):
        """
        Loads a read count data (a dictionary in a format: {tid: {pos: {read_len: count}}}).
        """
        print(f"\n--- Loading count data ---")
        if endonuclease not in ("RNase I", "MNase"):
            raise ValueError("Bad [endonuclease] value, please provide 'RNase I' or 'MNase'!")
        if not os.path.exists(count_dict_path):
            raise FileNotFoundError(f"Data file not found: {count_dict_path}")
        
        self.endonuclease = endonuclease
        self.count_dict, self.sample_name = self._load_pickle_file(count_dict_path)

    def evaluate_read_density(self,
                              count_dict: dict,
                              output_dir: str,
                              left: int = 40,
                              right: int = 60,
                              min_len: int = 21,
                              max_len: int = 40,
                              prefix: str = "read"):
        if not count_dict:
            raise ModuleNotFoundError("Read count data not loaded ! Please first run function: load_read_count_data()")
        
        print("- Evaluating read density around TIS and TTS...")
        # generate matrix of read counts around start and stop codon
        lengths, mat_starts, rel_pos_start = build_length_position_matrix(
            count_dict, self.tx_cds, output_dir,
            "start", left, right, min_len, max_len, prefix)
        lengths, mat_stops, rel_pos_stop = build_length_position_matrix(
            count_dict, self.tx_cds, output_dir,
            "stop", left, right, min_len, max_len, prefix)
        
        # plot read density
        plot_start_stop_read_length(
            mat_starts, mat_stops, rel_pos_start, rel_pos_stop, lengths, 
            os.path.join(output_dir, prefix + "_perc.start_stop_heatmap.read_len.pdf"), figsize=(8, 5))
        
        return mat_starts, mat_stops, rel_pos_start, rel_pos_stop, lengths
    
    def evaluate_three_nucleotide_periodicity(
            self,
            count_dict: dict,
            output_dir: str,
            min_len: int = 21,
            max_len: int = 40,
            prefix: str = "read"):
        
        if not count_dict:
            raise ModuleNotFoundError("Read count data not loaded ! Please first run function: load_read_count_data()")
        
        print("- Evaluating 3-nt periodicity around TIS and TTS...")
        
        # generate matrix of read length-specifc 3-nt periodity
        lengths, mat, totals = build_length_frame_matrix(
            count_dict, self.tx_cds, output_dir, min_len, max_len, prefix)
        plot_length_periodicity(
            lengths, mat, totals,
            os.path.join(output_dir, prefix + "_perc.length_frame_heatmap.pdf"), figsize=(3, 4))
        
        # periodicity per read length
        prd_read_len = (mat.max(axis=1) + 1) / (mat.sum(axis=1) + 1)
        
        return prd_read_len, lengths, totals
    
    def eval_p_site(
            self,
            p_site_dict: dict,
            output_dir: str,
            prefix: str = "p_site"
    ):
        print("\n--- Evaluating P site distribution and periodicity ---")
        # evaluate read density around TIS and TTS
        self.evaluate_read_density(p_site_dict, output_dir, prefix=prefix)
        self.evaluate_three_nucleotide_periodicity(p_site_dict, output_dir, prefix=prefix)
    
    def calculate_offset_map_fix(
            self,
            output_dir: str,
            train_ref: Literal["TIS", "TTS"],
            search_range: Tuple[int, int] = None,
            span: int = 5,
            prd_cutoff: float = 0.55
            ) -> Tuple[Dict, Dict]:
        
        if not self.count_dict:
            raise ModuleNotFoundError("Read count data not loaded ! Please first run function: load_read_count_data()")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        print("\n--- Calculating read length-specific offset ---")

        # evaluate read density around TIS and TTS
        mat_starts, mat_stops, rel_pos_start, rel_pos_stop, lengths = self.evaluate_read_density(
            self.count_dict, output_dir, prefix="read")
        prd_read_len, lengths, totals = self.evaluate_three_nucleotide_periodicity(
            self.count_dict, output_dir, prefix="read")

        # caculate potential read-length-specific offset
        if train_ref == "TIS":
            mat = mat_starts
            rel_pos = rel_pos_start
            search_range = (-13, -11) if not search_range else search_range
        elif train_ref == "TTS":
            mat = mat_stops
            rel_pos = rel_pos_stop
            search_range = (-16, -14) if not search_range else search_range

        best_pos_dict, inf_read_lens = find_peaks_in_range_all_lengths(
            mat, rel_pos, lengths, totals, 
            output_dir, search_range, smoothing_sigma=0.2)
        
        # set read_length_limits to select informative reads
        self.read_length_limits = inf_read_lens
        self._cache_read_length_emb()
        
        # generate offset limits or fixed value
        w = span // 2
        self.offset_map = {
            rl: (-pos - w, -pos + w) if train_ref == "TIS" else (-pos - 3 - w, -pos - 3 + w)
                for rl, pos in best_pos_dict.items()
        }
        self.offset_fix = {
            rl: -pos if train_ref == "TIS" else -pos - 3
                for rl, pos in best_pos_dict.items() if prd_read_len[np.where(np.array(lengths) == rl)] >= prd_cutoff and rl in inf_read_lens
        }
        print(f"- Fixed offset for some read lengths: {self.offset_fix}")

        return self.offset_map, self.offset_fix
    
    def _prepare_train_datasets(self,
                                tids: list,
                                train_ref: Literal["TIS", "TTS"],
                                five_end_flanking: bool = True,
                                three_end_flanking: bool = True,
                                width: int = 8):
        flanking_list = []
        offset_list = []
        read_len_list = []

        
        print(f"Offset limits: {self.offset_limits}")
        print(f"Read length-specific offset limits: {self.offset_map}")

        for tid in tqdm(tids, desc=f"Generate train dataset around {train_ref}", mininterval=200):
            # only for imformative transcripts
            if tid not in self.count_dict:
                continue

            # extract flanking sequences
            seq_emb = self._base_seq_emb.get(tid)
            seq_len = seq_emb.shape[0]
            tis_tts_pos = self.cds_start_pos[tid] if train_ref == "TIS" else self.cds_end_pos[tid] # 0-based
            count = self.count_dict[tid]

            for pos in count:
                five_end = pos - 1 # 0-based
                # read enrichment in TIS and 3 nt upstream of TTS 
                offset = tis_tts_pos - five_end
                # 1. select reads around TIS or TTS
                if offset not in self.offset_limits:
                    continue
                for read_l in count[pos]:
                    # 2. select reads with informative length
                    if read_l not in self.read_length_limits or read_l in self.offset_fix:
                        continue

                    # 3. if no offset_limits and offset_map provided, select reads with explicit read length-specific offset
                    if self.offset_map:
                        if not (offset >= self.offset_map[read_l][0] and offset <= self.offset_map[read_l][1]):
                            continue
                    three_end = five_end + read_l - 1 # 0-based

                    # 4. ensure 2 *(window * 2 + 1)
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
        X_seq = np.array(flanking_list, dtype=np.int8).reshape(n_sample, -1) # (n_reads, seq_feat)

        # read len one-hot
        X_rl = np.vstack([self.read_len_onehot_map[rl] for rl in read_len_list]).astype(np.float32)

        # seq ana read length features
        X_seq_df = pd.DataFrame(X_seq, columns = [f'seq_{i}' for i in range(X_seq.shape[1])])
        X_rl_df = pd.DataFrame(X_rl, columns = [f'rl_{i}' for i in range(X_rl.shape[1])])
        X_df = pd.concat([X_seq_df, X_rl_df], axis=1)

        self._feature_names_ = list(X_df.columns)

        # only these offset to train
        self.offset_limits = sorted(set(offset_list))

        y = np.array([np.where(np.array(self.offset_limits) == offset)[0][0] for offset in offset_list])


        return X_df, y
    
    def _prepare_batch_datasets(self, 
                                tids: List[str], 
                                five_end_flanking=True, 
                                three_end_flanking=True,
                                width = 8):
        """
        Prepare feature matrix for batch data (some tids)
        return: X (n_reads, feat_dim) and metadata (tid_list, pos_list, read_len_list)
        if no qualified reads, return (None, None, None, None)
        """
        flanking_list = []
        tid_list = []
        pos_list = []
        read_len_list = []

        for tid in tids:
            seq_emb = self._base_seq_emb.get(tid)
            if seq_emb is None:
                continue
            seq_len = seq_emb.shape[0]
            count_for_tid = self.count_dict.get(tid, {})
            # traverse read length and positions
            for pos, read_l_dict in count_for_tid.items():
                five_end = pos - 1 # 0-based
                for read_l in read_l_dict:
                    if read_l not in self.read_length_limits or read_l in self.offset_fix:
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
                    flanking_list.append(flanking_seq)  # 直接扁平化，便于 vstack
                    tid_list.append(tid)
                    pos_list.append(pos)
                    read_len_list.append(read_l)

        if len(flanking_list) == 0:
            return None, None, None, None

        # construct feature matrix
        n_sample = len(flanking_list)
        X_seq = np.array(flanking_list, dtype=np.int8).reshape(n_sample, -1) # (n_reads, seq_feat)

        # features
        if self._feature_names_ is None:
            raise RuntimeError("Feature names are not set. Run training first.")

        # read len one-hot
        X_rl = np.vstack([self.read_len_onehot_map[rl] for rl in read_len_list]).astype(np.float32)

        # seq ana read length features
        X_seq_df = pd.DataFrame(X_seq, columns = [f'seq_{i}' for i in range(X_seq.shape[1])])
        X_rl_df = pd.DataFrame(X_rl, columns = [f'rl_{i}' for i in range(X_rl.shape[1])])
        X_df = pd.concat([X_seq_df, X_rl_df], axis=1)
        
        X_df = X_df[self._feature_names_]

        return X_df, tid_list, pos_list, read_len_list

    def _evaluate(self, y_true: np.ndarray, X_val: pd.DataFrame, output_dir: str):
        """Evaluates the model on a validation set and saves reports."""
        if y_true is None or X_val is None or X_val.empty:
            print("No validation data for evaluation.")
            return
        
        print("\n--- Evaluating Model Performance ---")
        y_pred = self.predictor.predict(X_val)
        
        num_classes = len(self.offset_limits)
        report_labels = range(num_classes)
        target_names = [f"{i}" for i in self.offset_limits]

        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print(classification_report(y_true, y_pred, labels=report_labels, target_names=target_names, zero_division=0))
        
        # confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=report_labels)
        cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
        cm_df.to_csv(os.path.join(output_dir, "confusion_matrix.csv"))
        print("Confusion Matrix:\n", cm)

        # plot
        plot_offset_confusion_matrix_heatmap(
            cm_df, os.path.join(output_dir, "confusion_matrix.p_site_pred.pdf"), 
            cmap="Greys", vmin=0, vmax=20000, figsize=(4,3))

    def train(
            self,
            output_dir: str,
            train_ref: Literal["TIS", "TTS"] = None,
            offset_limits: Tuple[int, int] = None,
            search_range : Tuple[int, int] = None,
            span: int = None,
            five_end_flanking: bool = True,
            three_end_flanking: bool = True,
            lgb_params: dict = None,
            ):
        """
        Train P-site predictor using reads around TIS or TTS with LightGBM.
        User-supplied lgb_params override defaults; missing keys are filled with defaults.
        Some parameters that must match data (e.g. num_class) are enforced.
        """
        print("\n--- Starting Model Training ---")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # set train reference if not provided
        if not train_ref:
            if self.endonuclease == "RNase I":
                train_ref = "TIS"
            elif self.endonuclease == "MNase":
                train_ref = "TTS"
        # set read peak search range
        if not search_range:
            if self.endonuclease == "RNase I":
                search_range = (-14, -10)
            elif self.endonuclease == "MNase":
                search_range = (-18, -14)
        else:
            search_range = (min(search_range), max(search_range))
        # set training set span of read peak
        if not span:
            if self.endonuclease == "RNase I":
                span = 3
            elif self.endonuclease == "MNase":
                span = 5
        else:
            search_range = (min(search_range), max(search_range))
        print(f"Training set params: train_ref={train_ref} search_range={search_range} span={span}")

        # calculate read length-specific offset ranges (set self.offset_map)
        self.calculate_offset_map_fix(output_dir, train_ref, search_range, span)

        # if offset_limits was explicitly provided, use it; otherwise derive from offset_map
        if offset_limits:
            self.offset_limits = range(min(offset_limits), max(offset_limits) + 1)
        elif self.offset_map:
            self.offset_limits = range(
                min([v[0] for v in self.offset_map.values()]), 
                max([v[1] for v in self.offset_map.values()]) + 1)
        else:
            raise ValueError("Either offset_limits or offset_map should be provided !")

        # ---------------------------
        # 1) prepare datasets
        # ---------------------------
        # may reset self.offset_list
        X_train, y_train = self._prepare_train_datasets(self.tids_train,
                                                        train_ref, five_end_flanking, three_end_flanking)
        X_val, y_val = self._prepare_train_datasets(self.tids_valid,
                                                    train_ref, five_end_flanking, three_end_flanking)

        # check data
        print("Train/Val shapes:", getattr(X_train, "shape", None), getattr(X_val, "shape", None),
          getattr(y_train, "shape", None), getattr(y_val, "shape", None))
        if X_train is None or X_train.shape[0] == 0:
            raise RuntimeError("No training samples found. Check count_dict and offset_limits.")

        # determine number of classes
        num_classes = int(np.max(y_train)) + 1

        # ---------------------------
        # 2) set default params and merge with user params 
        # ---------------------------
        # defaults (will be used when user did not provide a key)
        DEFAULT_LGB_PARAMS = {'num_leaves': 32, 'max_depth': 8} if self.endonuclease == "RNase I" else {'num_leaves': 41, 'max_depth': 11}
        DEFAULT_LGB_PARAMS['objective'] = 'multiclass'
        DEFAULT_LGB_PARAMS['num_class'] = num_classes      # will be overwritten below to ensure correctness
        DEFAULT_LGB_PARAMS['learning_rate'] = 0.1
        DEFAULT_LGB_PARAMS['n_estimators'] = 2000
        DEFAULT_LGB_PARAMS['min_child_samples'] = 50
        DEFAULT_LGB_PARAMS['subsample'] = 0.8
        DEFAULT_LGB_PARAMS['colsample_bytree'] = 0.8
        DEFAULT_LGB_PARAMS['reg_alpha'] = 0.0
        DEFAULT_LGB_PARAMS['reg_lambda'] = 1.0
        DEFAULT_LGB_PARAMS['random_state'] = 42
        DEFAULT_LGB_PARAMS['n_jobs'] = self.n_thread
        DEFAULT_LGB_PARAMS['class_weight'] = 'balanced'
        DEFAULT_LGB_PARAMS['verbosity'] = -1

        # Extract fit-only options from user params (if provided)
        user_params = dict(lgb_params) if lgb_params is not None else {}
        # pop fit-only keys so they don't get passed to the constructor unexpectedly
        fit_categorical = user_params.pop('categorical_feature', None)
        early_stopping_rounds = user_params.pop('early_stopping_rounds', 100)
        log_period = user_params.pop('log_period', 10)

        # Merge defaults with user params: user params take precedence
        merged = {**DEFAULT_LGB_PARAMS, **user_params}

        # Ensure num_class matches the actual labels; warn if user provided a conflicting value
        if 'num_class' in user_params and int(user_params['num_class']) != num_classes:
            print(f"Warning: user-specified num_class={user_params['num_class']} "
                f"does not match labels-derived num_classes={num_classes}. Overriding to {num_classes}.")

        # Final merged params will be passed to LGBMClassifier constructor
        # Keep a copy for logging/debugging
        print("LightGBM params (merged):")
        for k in sorted(merged.keys()):
            # mask large structures if any
            print(f"  {k}: {merged[k]}")

        # create model
        self.predictor = lgb.LGBMClassifier(**merged)

        # ---------------------------
        # 3) fit the model (with early stopping callbacks)
        # ---------------------------
        if X_val is not None and X_val.shape[0] > 0:
            # early stopping & logging, add evaluation printing every 10 rounds 
            # prepare callbacks: early stopping and periodic logging
            callbacks = [
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=log_period)
            ]
            # decide categorical_feature for fit: prefer explicit fit_categorical provided by user,
            # otherwise fallback to self._feature_names_ (if available).
            fit_cat = fit_categorical if fit_categorical is not None else getattr(self, "_feature_names_", None)

            # call fit; pass categorical_feature if available
            fit_kwargs = dict(eval_set=[(X_val, y_val)], eval_metric='multi_logloss', callbacks=callbacks)
            if fit_cat is not None:
                fit_kwargs['categorical_feature'] = fit_cat

            self.predictor.fit(X_train, y_train, **fit_kwargs)
        else:
            # no validation set available: simple fit
            self.predictor.fit(X_train, y_train)

        # ---------------------------
        # 4) evaluate on validation set (if present)
        # ---------------------------
        if X_val is None or X_val.shape[0] == 0:
            print("No validation data found; skipping validation metrics.")
        else:
            # delegate to your evaluation helper
            self._evaluate(y_val, X_val, output_dir)

        # ---------------------------
        # 5) save model
        # ---------------------------
        model_path = os.path.join(output_dir, "lgbm_model.joblib")
        joblib.dump(self.predictor, model_path)
        print(f"✔ LightGBM model saved to {model_path}")
            
    def predict(
            self,
            output_dir: str,
            offset_limits: tuple = None,
            model_path: str = None,
            five_end_flanking: bool = True,
            three_end_flanking: bool = True,
            batch_transcripts: int = 2000,
            max_reads_per_batch: int = 10000000
            ) -> Dict:
        """
        Predicts P-sites for all transcripts using a multi-process producer-consumer pattern.
        """

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        p_site_dict = defaultdict(double_nested_zero_defaultdict)

        # check predictor
        if not self.predictor:
            if model_path:
                self.load_model(model_path, offset_limits)
            else:
                raise ModuleNotFoundError("Model not found, please train first or give a model path!")

        print("\n--- Identify P-sites for specific read length with high periodicity ---")

        # predict P-site using fixed offset
        for tid, raw_counts in self.count_dict.items():
            any_val = next(iter(raw_counts.values()))
            if isinstance(any_val, dict):
                # nested by length
                for pos, d in raw_counts.items():
                    for read_l, cnt in d.items():
                        if read_l in self.offset_fix:
                            p_pos = pos + self.offset_fix[read_l]
                            p_site_dict[tid][p_pos][read_l] += cnt
            else:
                # raw_counts already collapsed (no length info) -> skip (can't attribute length)
                continue

        print("\n--- Predicting P-sites for all transcripts ---")
        # --- pipeline parameters 
        producer_queue_maxsize = 2

        # Prepare queue and control vars
        q = Queue(maxsize=producer_queue_maxsize)
        producer_exc = []

        # Producer function runs in a separate process
        def _producer(keys_iter, 
                      batch_transcripts: int, 
                      five_end_flanking: bool, 
                      three_end_flanking: bool):
            try:
                while True:
                    tids_batch = list(islice(keys_iter, batch_transcripts))
                    if not tids_batch:
                        break
                    X_t, tid_list_t, pos_list_t, read_len_list_t = self._prepare_batch_datasets(
                        tids_batch, five_end_flanking, three_end_flanking)
                    # put in line
                    q.put((X_t, tid_list_t, pos_list_t, read_len_list_t, len(tids_batch)))
                q.put(None)
            except Exception:
                # record traceback
                tb = traceback.format_exc()
                producer_exc.append((e, tb))
                q.put(None)

        # start producer process
        keys_iter = iter(self.count_dict.keys())
        total_tids = len(self.count_dict)
        producer_thr = Thread(target=_producer, 
                              args=(keys_iter, batch_transcripts, five_end_flanking, three_end_flanking), 
                              daemon=True)
        producer_thr.start()

        # Consumer loop: get the batch in line，predict and update p_site_dict
        with tqdm(total=total_tids, desc="Predicting P-site", mininterval=1) as pbar:
            while True:
                item = q.get()
                if item is None:
                    break

                X, tid_list, pos_list, read_len_list, n_tids_batch = item

                pbar.update(n_tids_batch)

                if X is None:
                    continue

                # split into manageable sub-blocks to avoid OOM
                n_samples = X.shape[0]
                start_idx = 0
                while start_idx < n_samples:
                    end_idx = min(start_idx + max_reads_per_batch, n_samples)
                    if hasattr(X, "iloc"):
                        subX = X.iloc[start_idx:end_idx, :]
                    else:
                        subX = X[start_idx:end_idx, :]

                    # convert to DataFrame with feature names if model was trained with names
                    if getattr(self, "_feature_names_", None) is not None:
                        # ensure correct feature columns
                        if subX.shape[1] != len(self._feature_names_):
                            try:
                                subX = pd.DataFrame(subX, columns=self._feature_names_)
                            except Exception:
                                raise RuntimeError(f"Feature dimension mismatch: model expects {len(self._feature_names_)} features, got {subX.shape[1]}")
                        else:
                            # ensure DataFrame with same columns
                            if not hasattr(subX, "columns"):
                                subX = pd.DataFrame(subX, columns=self._feature_names_)
                            else:
                                subX.columns = self._feature_names_

                    # model predict
                    pred_labels = self.predictor.predict(subX)
                    offsets = np.array(self.offset_limits)[pred_labels]

                    for i_local, global_idx in enumerate(range(start_idx, end_idx)):
                        tid = tid_list[global_idx]
                        read_l = read_len_list[global_idx]
                        pos = pos_list[global_idx]
                        offset = int(offsets[i_local])
                        seq_len = self._base_seq_emb[tid].shape[0]
                        p_pos = pos + offset
                        if p_pos <= seq_len and read_l not in self.offset_fix:
                            try:
                                cnt = self.count_dict[tid][pos][read_l]
                            except Exception:
                                try:
                                    cnt = self.count_dict[tid][read_l][pos]
                                except Exception:
                                    cnt = 1
                            p_site_dict[tid][p_pos][read_l] += cnt

                    start_idx = end_idx

        # wait producer thread end
        producer_thr.join()

        # if error in producer
        if producer_exc:
            e, tb = producer_exc[0]
            raise RuntimeError(f"Producer thread failed: {e}\nTraceback:\n{tb}")

        print("\n--- Save P-site counts ---")
        out_path = os.path.join(output_dir, "p_site_count.pkl")
        self._save_pickle_file(p_site_dict, out_path)

        return p_site_dict
