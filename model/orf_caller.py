import os
import re
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from Bio.Seq import Seq

# =================================================================
# [NEW] 安全清理 ID 的辅助函数
# =================================================================
def safe_clean_id(raw_id: str) -> str:
    """
    安全清理 ID：
    1. 去除 FASTA 头部可能含有的管道符及后面的内容（例如 '>ENST0001|protein_coding' -> 'ENST0001'）
    2. 仅当 ID 以 'ENST' 或 'ENSG' 开头时，才去除 '.' 及后面的版本号
    """
    clean_id = str(raw_id).split('|')[0]
    if (clean_id.startswith('ENST') or clean_id.startswith('ENSG')) and '.' in clean_id:
        clean_id = clean_id.split('.')[0]
    return clean_id


# =================================================================
# Util: Fast Fasta Parser
# =================================================================
def read_fasta(file_paths: Union[str, List[str]]) -> Dict[str, str]:
    if isinstance(file_paths, str):
        file_paths = [file_paths]
        
    seq_dict = {}
    total_files = len(file_paths)
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"[Warning] Fasta file not found: {file_path}. Skipping...")
            continue
            
        curr_id = ""
        curr_seq = []
        file_seq_count = 0
        print(f"Reading Fasta File: {file_path}")
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith('>'):
                    if curr_id:
                        seq_dict[curr_id] = "".join(curr_seq).replace('U', 'T')
                        file_seq_count += 1
                    
                    # [MODIFIED] 使用安全清理函数
                    raw_id = line[1:].split()[0]
                    curr_id = safe_clean_id(raw_id)
                    curr_seq = []
                else:
                    curr_seq.append(line.upper())
            if curr_id:
                seq_dict[curr_id] = "".join(curr_seq).replace('U', 'T')
                file_seq_count += 1
                
        print(f"  -> Loaded {file_seq_count} sequences from this file.")
        
    print(f"✅ Successfully loaded a total of {len(seq_dict)} unique sequences.")
    return seq_dict

# =================================================================
# Core Algorithm: Multi-dimensional Prefix Sum ORF Caller
# (FastSignalDrivenORFCaller 的代码保持不变，此处省略以节省空间)
# =================================================================
class FastSignalDrivenORFCaller:
    def __init__(self, start_codons=['ATG', 'CTG', 'GTG'], stop_codons=['TAA', 'TAG', 'TGA'], min_len=30, mode='balanced'):
        self.start_codons, self.stop_codons, self.min_len, self.mode = start_codons, stop_codons, min_len, mode.lower()
        self.stop_re = re.compile(f"(?=({'|'.join(stop_codons)}))")
        self.start_re = re.compile(f"(?=({'|'.join(start_codons)}))")

    def extract_all_candidates(self, sequence: str) -> List[dict]:
        candidates = []
        starts_by_frame = {0: [], 1: [], 2: []}
        stops_by_frame = {0: [], 1: [], 2: []}

        for match in self.start_re.finditer(sequence):
            pos = match.start()
            starts_by_frame[pos % 3].append(pos)

        for match in self.stop_re.finditer(sequence):
            pos = match.start()
            stops_by_frame[pos % 3].append(pos)

        for frame in range(3):
            starts = starts_by_frame[frame]
            stops = stops_by_frame[frame]
            if not starts or not stops: continue

            stops_with_dummy = [-3] + stops
            start_idx = 0
            num_starts = len(starts)

            for i in range(len(stops_with_dummy) - 1):
                prev_stop = stops_with_dummy[i]
                curr_stop = stops_with_dummy[i + 1]
                while start_idx < num_starts and starts[start_idx] <= prev_stop:
                    start_idx += 1
                while start_idx < num_starts and starts[start_idx] < curr_stop:
                    start_pos = starts[start_idx]
                    orf_len = curr_stop - start_pos + 3
                    if orf_len >= self.min_len:
                        candidates.append({
                            'start': start_pos, 'stop': curr_stop,
                            'length': orf_len, 'start_codon': sequence[start_pos:start_pos+3]
                        })
                    start_idx += 1 
        return candidates

    def fast_nms(self, cands: List[dict], iou_threshold: float = 0.3) -> List[dict]:
        keep = []
        cands.sort(key=lambda x: x['score'], reverse=True)
        for i, cand in enumerate(cands):
            if cand.get('suppressed', False): continue
            keep.append(cand)
            s1, e1, l1 = cand['start'], cand['stop'], cand['length']
            for j in range(i + 1, len(cands)):
                if cands[j].get('suppressed', False): continue
                s2, e2, l2 = cands[j]['start'], cands[j]['stop'], cands[j]['length']
                overlap_l = max(0, min(e1, e2) - max(s1, s2))
                if overlap_l > 0:
                    iou = overlap_l / (l1 + l2 - overlap_l)
                    if iou > iou_threshold:
                        cands[j]['suppressed'] = True
        for k in keep:
            k.pop('suppressed', None)
        return keep

    def extract_features(self, sequence: str, signal_array: np.ndarray, intensity_threshold: float = 0.01) -> List[dict]:
        cands = self.extract_all_candidates(sequence)
        if not cands: return []

        seq_len = len(signal_array)
        cumsum_sig = np.zeros(seq_len + 1, dtype=np.float32)
        np.cumsum(signal_array, out=cumsum_sig[1:])
        
        valid_cands_pre = []
        for cand in cands:
            s, e, length = cand['start'], cand['stop'], cand['length']
            total_sig = float(cumsum_sig[e] - cumsum_sig[s])
            mean_intensity = total_sig / length
            if mean_intensity <= intensity_threshold: continue
            cand['mean_intensity'] = mean_intensity
            cand['total_sig'] = total_sig  
            valid_cands_pre.append(cand)

        if not valid_cands_pre: return []

        active_codons = (signal_array > intensity_threshold / 10.0).astype(np.float32) 
        cumsum_active_frames = [np.zeros(seq_len + 1, dtype=np.float32) for _ in range(3)]
        cumsum_frames = [np.zeros(seq_len + 1, dtype=np.float32) for _ in range(3)]
        
        for f in range(3):
            f_active = np.zeros(seq_len, dtype=np.float32)
            f_active[f::3] = active_codons[f::3]
            np.cumsum(f_active, out=cumsum_active_frames[f][1:])
            frame_sig = np.zeros(seq_len, dtype=np.float32)
            frame_sig[f::3] = signal_array[f::3]  
            np.cumsum(frame_sig, out=cumsum_frames[f][1:])
            
        flank_size = 30
        extracted_cands = []
        for cand in valid_cands_pre:
            s, e, length = cand['start'], cand['stop'], cand['length']
            total_sig, mean_intensity = cand['total_sig'], cand['mean_intensity']
            f_s = s % 3
            in_frame_sig = float(cumsum_frames[f_s][e] - cumsum_frames[f_s][s])
            periodicity = in_frame_sig / (total_sig + 1e-9)
            uniformity = float(cumsum_active_frames[f_s][e] - cumsum_active_frames[f_s][s]) / (length / 3.0)
            
            up_s, dn_s = max(0, s - flank_size), min(seq_len, s + flank_size)
            step_up_contrast = float(cumsum_sig[dn_s] - cumsum_sig[s]) / (float(cumsum_sig[s] - cumsum_sig[up_s]) + float(cumsum_sig[dn_s] - cumsum_sig[s]) + 1e-9)
            
            up_e, dn_e = max(0, e - flank_size), min(seq_len, e + flank_size)
            drop_off = float(cumsum_sig[e] - cumsum_sig[up_e]) / (float(cumsum_sig[e] - cumsum_sig[up_e]) + float(cumsum_sig[dn_e] - cumsum_sig[e]) + 1e-9)
            
            length_bonus = np.log10(min(length, 900) + 1) * 0.5 
            if self.mode == 'long': length_bonus = np.log10(length + 1)
            elif self.mode == 'short': length_bonus = np.log2(min(length, 450) + 1) * 0.5
            
            translation_score = length_bonus * mean_intensity # * (uniformity + 1e-3) * periodicity * step_up_contrast * drop_off 
            cand.update({'tri_nucleotide_periodicity': periodicity, 'uniformity_of_signal': uniformity,
                         'step_up_contrast': step_up_contrast, 'drop_off': drop_off, 'score': float(translation_score)})
            extracted_cands.append(cand)
        return extracted_cands

    def collapse_and_nms(self, cands: List[dict], iou_threshold: float = 0.3) -> List[dict]:
        cands_by_stop = {}
        for cand in cands:
            e = cand['stop']
            if e not in cands_by_stop: cands_by_stop[e] = []
            cands_by_stop[e].append(cand)
        resolved_cands = []
        for e, group in cands_by_stop.items():
            atg_cands = [c for c in group if c['start_codon'] == 'ATG']
            best_cand = max(atg_cands if atg_cands else group, key=lambda x: x['score'])
            resolved_cands.append(best_cand)
        return self.fast_nms(resolved_cands, iou_threshold=iou_threshold)

# =================================================================
# Main Pipeline
# =================================================================
class TranslationSignalORFCaller:
    def __init__(self, fasta_files: Union[str, List[str]], pkl_file: str, cell_type: str,
                 tpm_csv_path: Optional[str] = None, tpm_level: str = 'gene', mapping_csv_path: Optional[str] = None):
        self.fasta_files = fasta_files
        self.pkl_file = pkl_file
        self.cell_type = cell_type
        self.tpm_level = tpm_level.lower()
        self.has_tpm = tpm_csv_path is not None and os.path.exists(tpm_csv_path)
        
        print("\n[1/4] Loading Fasta File(s)...")
        self.seq_dict = read_fasta(self.fasta_files)
        with open(self.pkl_file, 'rb') as f:
            self.preds_data = pickle.load(f)
        if self.cell_type not in self.preds_data:
            raise ValueError(f"Cell type '{self.cell_type}' not found in PKL.")
            
        # Mapping: Transcript -> Gene
        self.tx2gene = {}
        if mapping_csv_path and os.path.exists(mapping_csv_path):
            print(f"[3/4] Loading Mapping from {mapping_csv_path}...")
            m_df = pd.read_csv(mapping_csv_path, sep='\t')
            g_col, t_col = 'Gene stable ID', 'Transcript stable ID'
            if g_col in m_df.columns and t_col in m_df.columns:
                for _, r in m_df.iterrows():
                    # [MODIFIED] 使用安全清理函数
                    g_id = safe_clean_id(str(r[g_col]))
                    t_id = safe_clean_id(str(r[t_col]))
                    self.tx2gene[t_id] = g_id
                print(f"  -> Successfully loaded mapping for {len(self.tx2gene)} transcripts.")
        else:
            if self.tpm_level == 'gene' and self.has_tpm:
                print("  [Warning] 'tpm_level' is set to 'gene', but no mapping table was provided. Will fallback to using Transcript ID as Gene ID.")
        
        # TPM Matrix Loading
        print(f"[4/4] Loading TPM Matrix (Level: {self.tpm_level})...")
        self.tpm_dict = {}
        if self.has_tpm:
            t_df = pd.read_csv(tpm_csv_path, index_col=0)
            
            # [MODIFIED] 使用 Pandas 的 apply 方法配合 safe_clean_id 清理 DataFrame 的 Index
            t_df.index = t_df.index.to_series().astype(str).apply(safe_clean_id)
            
            # 由于去除了版本号，同一个 ID 可能出现重复行 (例如 ENSG001.1 和 ENSG001.2 被去除了版本号)
            # 对重复的 ID 表达量取均值 (或者最大值)，这里选择均值
            t_df = t_df.groupby(t_df.index).mean()
            
            if self.cell_type in t_df.columns:
                self.tpm_dict = t_df[self.cell_type].to_dict()
                print(f"  -> Loaded TPM for {len(self.tpm_dict)} {self.tpm_level}s in '{self.cell_type}'.")
            else:
                print(f"  [Warning] Cell type '{self.cell_type}' not found in TPM matrix. Disabling TPM integration.")
                self.has_tpm = False

    def _plot_metrics_density(self, raw_df: pd.DataFrame, thresholds: dict, out_dir: str, mane_quantile: float):
        """Plots the density distributions of the core metrics."""
        print("\n[Visualizer] Generating Density Plots for the core metrics...")
        
        metrics_to_plot = [
            'mean_intensity', 'tri_nucleotide_periodicity', 
            'uniformity_of_signal', 'step_up_contrast', 'drop_off'
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        palette = {'MANE Annotated ORFs': '#e74c3c', 'Novel Potential ORFs': '#2980b9', 'All Predicted ORFs': '#27ae60'}

        for i, metric in enumerate(metrics_to_plot):
            if metric not in raw_df.columns: continue
            ax = axes[i]
            
            thresh_val = thresholds.get(metric, 0)
            thresh_label = f"Threshold (Bottom {mane_quantile*100:.0f}% + Hard):\n{thresh_val:.3f}"
                
            p01, p99 = raw_df[metric].quantile(0.01), raw_df[metric].quantile(0.99)
            plot_df = raw_df[(raw_df[metric] >= p01) & (raw_df[metric] <= p99)]
            
            max_plot_samples = 15000
            if len(plot_df) > max_plot_samples:
                plot_df = plot_df.groupby('Group', group_keys=False).apply(
                    lambda x: x.sample(min(len(x), max_plot_samples//2), random_state=42)
                )

            sns.kdeplot(
                data=plot_df, x=metric, hue='Group', fill=True, 
                common_norm=False, palette=palette, alpha=0.4, linewidth=2.5, ax=ax
            )
            
            ax.axvline(x=thresh_val, color='#e74c3c', linestyle='--', linewidth=2)
            y_max = ax.get_ylim()[1]
            ax.text(thresh_val, y_max * 0.85, f' {thresh_label}', 
                    color='#e74c3c', fontsize=11, fontweight='bold', ha='left')
            
            title_name = metric.replace('_', ' ').title()
            ax.set_title(f"{title_name} Distribution", fontsize=14, pad=10)
            ax.set_xlabel(title_name, fontsize=12)
            ax.set_ylabel("Density", fontsize=12)
            sns.despine(ax=ax)

        axes[5].set_visible(False)
        plt.suptitle(f"Translation Metrics: {self.cell_type}", fontsize=18, y=1.02)
        plt.tight_layout()
        
        plot_path = os.path.join(out_dir, f"orf_metrics_density.{self.cell_type}.pdf")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  -> Plots successfully saved to: {plot_path}")

    def run(self, mane_orfs_path=None, out_dir="./results", start_codons=['ATG', 'CTG', 'GTG'],
            min_len=30, offset_tolerance=6, mode='balanced', use_mane_filter=True, mane_quantile=0.05,
            plot_density=True, hard_thresh_intensity=0.01, hard_thresh_periodicity=0.40,
            hard_thresh_uniformity=0.20, hard_thresh_step_up=0.3, hard_thresh_drop_off=0.3) -> pd.DataFrame:
        
        os.makedirs(out_dir, exist_ok=True)
        cell_preds = self.preds_data[self.cell_type]
        caller = FastSignalDrivenORFCaller(start_codons=start_codons, min_len=min_len, mode=mode)
        all_raw_records = []
        
        print(f"\n[Phase 1] Extracting candidates & integrating LogTPM...")
        for tid, pred_raw in tqdm(cell_preds.items()):
            # [MODIFIED] 使用安全清理函数
            clean_tid = safe_clean_id(tid)
            
            if clean_tid not in self.seq_dict: continue
            
            sequence = self.seq_dict[clean_tid]
            pred_signal = np.expm1(pred_raw.reshape(-1).astype(np.float32))
            v_len = min(len(sequence), len(pred_signal))
            cands = caller.extract_features(sequence[:v_len], pred_signal[:v_len], hard_thresh_intensity/10)
            
            gene_id = self.tx2gene.get(clean_tid, clean_tid)
            query_id = clean_tid if self.tpm_level == 'transcript' else gene_id
            
            if self.has_tpm:
                tpm_val = float(self.tpm_dict.get(query_id, 0.0))
                if pd.isna(tpm_val) or tpm_val < 0:
                    tpm_val = 0.0
                log_tpm = np.log2(tpm_val + 1.0)
            else:
                tpm_val = np.nan
                log_tpm = 1.0
            
            for cand in cands:
                cand.update({
                    'Tid': clean_tid, 
                    'Gene_ID': gene_id, 
                    'Cell_Type': self.cell_type,
                    'tpm': tpm_val, 
                    'log2_tpm_plus_1': log_tpm,
                    'translation_score': cand['score'],
                    'expr_score': cand['score'] * log_tpm
                })
                all_raw_records.append(cand)
                
        if not all_raw_records: return pd.DataFrame()
        raw_df = pd.DataFrame(all_raw_records)
        raw_df['orf_index'] = raw_df.index

        hard_thresholds = {
            'mean_intensity': hard_thresh_intensity, 'tri_nucleotide_periodicity': hard_thresh_periodicity,
            'uniformity_of_signal': hard_thresh_uniformity, 'step_up_contrast': hard_thresh_step_up, 'drop_off': hard_thresh_drop_off
        }
        
        final_thresholds = {}
        if use_mane_filter and mane_orfs_path and os.path.exists(mane_orfs_path):
            print(f"\n[Phase 2] Mapping candidates to MANE annotations (Tolerance: ±{offset_tolerance} nt)...")
            mane_df = pd.read_csv(mane_orfs_path)
            
            gt_dict = {}
            for row in mane_df.itertuples(index=False):
                t_id = str(row.PacBio_ID)
                if t_id not in gt_dict: gt_dict[t_id] = []
                gt_dict[t_id].append((row.CDS_Start_0based, row.CDS_End_0based))
                
            annotated_indices = set()
            for row in raw_df.itertuples(index=False):
                t_id = row.Tid
                if t_id in gt_dict:
                    for g_start, g_stop in gt_dict[t_id]:
                        if abs(row.start - g_start) <= offset_tolerance and abs(row.stop - g_stop) <= offset_tolerance:
                            annotated_indices.add(row.orf_index)
                            break 
                            
            raw_df['Group'] = 'Novel Potential ORFs'
            raw_df.loc[raw_df['orf_index'].isin(annotated_indices), 'Group'] = 'MANE Annotated ORFs'
            
            num_mane = (raw_df['Group'] == 'MANE Annotated ORFs').sum()
            print(f"  -> Found {num_mane} MANE Annotated ORFs.")

            print("\n[Phase 3] Calculating Thresholds (MANE Quantile + Hard Baseline)...")
            mane_data = raw_df[raw_df['Group'] == 'MANE Annotated ORFs']
            
            for metric, h_thresh in hard_thresholds.items():
                if metric not in raw_df.columns: continue
                mane_thresh_val = mane_data[metric].quantile(mane_quantile) if num_mane > 0 else 0
                final_val = max(mane_thresh_val, h_thresh)
                final_thresholds[metric] = final_val
                print(f"  -> {metric}: MANE {mane_quantile*100}% = {mane_thresh_val:.3f} | Hard = {h_thresh:.3f} ==> Final = {final_val:.3f}")
        else:
            print("\n[Phase 2 & 3] MANE Filter is DISABLED. Using absolute Hard Thresholds.")
            raw_df['Group'] = 'All Predicted ORFs'
            final_thresholds = hard_thresholds
            for metric, val in final_thresholds.items():
                print(f"  -> Applying absolute threshold for {metric}: {val:.3f}")

        if plot_density:
            self._plot_metrics_density(raw_df, final_thresholds, out_dir, mane_quantile)

        print("\n[Phase 4] Filtering Candidates by Thresholds...")
        high_conf_mask = pd.Series(True, index=raw_df.index)
        for metric, thresh in final_thresholds.items():
            high_conf_mask &= (raw_df[metric] >= thresh)
            
        high_conf_df = raw_df[high_conf_mask].copy()
        print(f"  -> Total ORFs before filter : {len(raw_df)}")
        print(f"  -> Total ORFs after filter  : {len(high_conf_df)}")
        
        annotated_csv_path = os.path.join(out_dir, f"all_candidates_labeled.{self.cell_type}.csv")
        raw_df.drop(columns=['orf_index']).to_csv(annotated_csv_path, index=False)
        
        print(f"\n[Phase 5] Stop-Codon Collapse (Score Driven) and NMS...")
        final_records = []
        for _, group in tqdm(high_conf_df.groupby('Tid')):
            final_records.extend(caller.collapse_and_nms(group.to_dict('records'), iou_threshold=0.7))
        
        final_df = pd.DataFrame(final_records)
        cols = ['Tid', 'Gene_ID', 'Cell_Type', 'start', 'stop', 'length', 'start_codon', 
                'translation_score', 'tpm', 'log2_tpm_plus_1', 'expr_score', 'mean_intensity', 
                'tri_nucleotide_periodicity', 'uniformity_of_signal', 'step_up_contrast', 'drop_off']
        
        if not final_df.empty:
            final_df = final_df[cols].sort_values(by=['expr_score'], ascending=False)
            final_df.to_csv(os.path.join(out_dir, f"high_confidence_orfs.{self.cell_type}.{mode}_mode.csv"), index=False)
            print(f"\n🎉 ORF Calling Completed! Found {len(final_df)} Final High-Confidence ORFs.")

            # Phase 6: Protein FASTA
            print("\n[Phase 6] Extracting and translating DNA sequences to Protein FASTA...")
            f_path = os.path.join(out_dir, f"high_confidence_proteins.{self.cell_type}.{mode}_mode.fasta")
            
            extracted_count = 0
            with open(f_path, 'w') as f_out:
                for r in final_df.itertuples(index=False):
                    seq_dna = self.seq_dict[r.Tid][r.start:r.stop+3]
                    seq_dna = seq_dna[:len(seq_dna) - (len(seq_dna) % 3)]
                    
                    try:
                        prot = str(Seq(seq_dna).translate(to_stop=True))
                        if prot:
                            tpm_display = r.tpm if not pd.isna(r.tpm) else "N/A"
                            hdr = f">{r.Tid}|{r.Gene_ID}|{r.Cell_Type}|{r.start}:{r.stop}|ExprScore:{r.expr_score:.3f}|TPM:{tpm_display}"
                            split_prot = [prot[i:i+80] for i in range(0, len(prot), 80)]
                            f_out.write(f"{hdr}\n" + "\n".join(split_prot) + "\n")
                            extracted_count += 1
                    except Exception as e:
                        print(f"  [Warning] Translation failed for {r.Tid}: {e}")
                        
            print(f"✅ Saved FASTA to: {f_path}")
        return final_df