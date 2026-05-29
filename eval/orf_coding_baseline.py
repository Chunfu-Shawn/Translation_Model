import os
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from tqdm import tqdm

# =================================================================
# [NEW] 安全清理 ID 的辅助函数 (复用之前的稳健逻辑)
# =================================================================
def safe_clean_id(raw_id: str) -> str:
    """仅去除 ENST/ENSG 的版本号，保留 PacBio/MSTRG 的完整结构"""
    clean_id = str(raw_id).split('|')[0]
    if (clean_id.startswith('ENST') or clean_id.startswith('ENSG')) and '.' in clean_id:
        clean_id = clean_id.split('.')[0]
    return clean_id


# =================================================================
# Util: Fast Fasta Parser
# =================================================================
def read_fasta(file_path: str) -> Dict[str, str]:
    """Read Fasta file and return a {tid: sequence} dictionary. (Turbo Version)"""
    seq_dict = {}
    curr_id = ""
    curr_seq = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('>'):
                if curr_id:
                    seq_dict[curr_id] = "".join(curr_seq).replace('U', 'T')
                
                # [MODIFIED] 使用安全 ID 清理
                raw_id = line[1:].split()[0]
                curr_id = safe_clean_id(raw_id)
                curr_seq = []
            else:
                curr_seq.append(line.upper())
        if curr_id:
            seq_dict[curr_id] = "".join(curr_seq).replace('U', 'T')
            
    print(f"Loaded {len(seq_dict)} sequences from {file_path}")
    return seq_dict

# =================================================================
# Core Algorithm: Pure Sequence-Based Baseline ORF Caller
# =================================================================
class BaselineSequenceORFCaller:
    def __init__(self, 
                 start_codons: List[str] = ['ATG', 'CTG', 'GTG', 'TTG', 'ACG'], 
                 stop_codons: List[str] = ['TAA', 'TAG', 'TGA'], 
                 min_len: int = 30):
        
        self.start_codons = start_codons
        self.stop_codons = stop_codons
        self.min_len = min_len
        
        # 定义基线打分权重 (ATG 权重最高，依此类推)
        self.codon_weights = {
            'ATG': 1.0,
            'CTG': 0.8,
            'GTG': 0.6,
            'TTG': 0.4,
            'ACG': 0.2
        }
        
        self.stop_re = re.compile(f"(?=({'|'.join(stop_codons)}))")
        self.start_re = re.compile(f"(?=({'|'.join(start_codons)}))")

    def extract_and_score_candidates(self, sequence: str) -> List[dict]:
        """Extract all valid ORFs and assign baseline sequence scores."""
        candidates = []
        stop_positions = {0: [], 1: [], 2: []}
        
        for match in self.stop_re.finditer(sequence):
            pos = match.start()
            stop_positions[pos % 3].append(pos)
            
        for match in self.start_re.finditer(sequence):
            start_pos = match.start()
            frame = start_pos % 3
            
            for stop_pos in stop_positions[frame]:
                if stop_pos > start_pos:
                    orf_len = stop_pos - start_pos + 3 
                    if orf_len >= self.min_len:
                        start_codon = sequence[start_pos:start_pos+3]
                        
                        # 基线打分逻辑: 起始密码子权重 * 长度对数
                        weight = self.codon_weights.get(start_codon, 0.1)
                        length_bonus = np.log10(orf_len + 1)
                        baseline_score = weight * length_bonus
                        
                        candidates.append({
                            'start': start_pos,
                            'stop': stop_pos,
                            'length': orf_len,
                            'start_codon': start_codon,
                            'score': float(baseline_score) 
                        })
                    break 
        return candidates

    def fast_nms(self, cands: List[dict], iou_threshold: float = 0.3) -> List[dict]:
        """Non-Maximum Suppression to resolve spatial overlaps."""
        keep = []
        # NMS 始终依赖序列基线得分 (局部去重最优解)
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

    def collapse_and_nms(self, cands: List[dict], iou_threshold: float = 0.3) -> List[dict]:
        """Collapse identical stop codons (prioritizing ATG, then length) and apply NMS."""
        if not cands: return []
        
        cands_by_stop = {}
        for cand in cands:
            e = cand['stop']
            if e not in cands_by_stop:
                cands_by_stop[e] = []
            cands_by_stop[e].append(cand)
            
        resolved_cands = []
        for e, group in cands_by_stop.items():
            if len(group) == 1:
                resolved_cands.append(group[0])
            else:
                # 规则 1: 优先选择 ATG
                atg_cands = [c for c in group if c['start_codon'] == 'ATG']
                if atg_cands:
                    # 规则 2: 多个 ATG 中选最长的
                    best_cand = max(atg_cands, key=lambda x: x['length'])
                else:
                    # 如果没有 ATG，在其余组合中选最长的
                    best_cand = max(group, key=lambda x: x['length'])
                    
                resolved_cands.append(best_cand)

        return self.fast_nms(resolved_cands, iou_threshold=iou_threshold)

# =================================================================
# Main Pipeline: Batch Processing & Saving
# =================================================================
class BaselineORFIdentifier:
    def __init__(self, 
                 fasta_file: str,
                 cell_type: str,
                 tpm_csv_path: Optional[str] = None,
                 tpm_level: str = 'gene',
                 mapping_csv_path: Optional[str] = None):
                 
        self.fasta_file = fasta_file
        self.cell_type = cell_type
        self.tpm_level = tpm_level.lower()
        self.has_tpm = tpm_csv_path is not None and os.path.exists(tpm_csv_path)
        
        print("Loading Fasta File for Baseline Analysis...")
        self.seq_dict = read_fasta(self.fasta_file)
        
        # =================================================================
        # [NEW] 建立 Transcript -> Gene Mapping (复用机制)
        # =================================================================
        self.tx2gene = {}
        if mapping_csv_path and os.path.exists(mapping_csv_path):
            print(f"Loading Gene-Transcript Mapping from {mapping_csv_path}...")
            try:
                m_df = pd.read_csv(mapping_csv_path, sep='\t')
                g_col, t_col = 'Gene stable ID', 'Transcript stable ID'
                if g_col in m_df.columns and t_col in m_df.columns:
                    for _, r in m_df.iterrows():
                        g_id = safe_clean_id(str(r[g_col]))
                        t_id = safe_clean_id(str(r[t_col]))
                        self.tx2gene[t_id] = g_id
            except Exception as e:
                print(f"  [Error] Failed to load Mapping CSV: {e}")
                
        # =================================================================
        # [NEW] 加载 TPM 矩阵
        # =================================================================
        print(f"Loading TPM Expression Matrix (Level: {self.tpm_level})...")
        self.tpm_dict = {}
        if self.has_tpm:
            try:
                t_df = pd.read_csv(tpm_csv_path, index_col=0)
                t_df.index = t_df.index.to_series().astype(str).apply(safe_clean_id)
                t_df = t_df.groupby(t_df.index).mean() # 去重合并
                
                if self.cell_type in t_df.columns:
                    self.tpm_dict = t_df[self.cell_type].to_dict()
                    print(f"  -> Loaded TPM for {len(self.tpm_dict)} {self.tpm_level}s in '{self.cell_type}'.")
                else:
                    print(f"  [Warning] Cell type '{self.cell_type}' not found. Disabling TPM integration.")
                    self.has_tpm = False
            except Exception as e:
                print(f"  [Error] Failed to load TPM matrix: {e}")
                self.has_tpm = False

    def run(self, 
            out_dir: str = "./results/baseline", 
            start_codons: List[str] = ['ATG', 'CTG', 'GTG', 'TTG', 'ACG'],
            target_tids: Optional[list] = None, 
            min_len: int = 30) -> pd.DataFrame:
        
        os.makedirs(out_dir, exist_ok=True)
        caller = BaselineSequenceORFCaller(start_codons=start_codons, min_len=min_len)
        all_records = []
        
        if target_tids is not None:
            # [MODIFIED] 清理 target_tids
            target_set = set(safe_clean_id(t) for t in target_tids)
            filtered_seq_dict = {}
            for tid, seq in self.seq_dict.items():
                if tid in target_set:
                    filtered_seq_dict[tid] = seq
            
            print(f"Filtered Fasta: Keeping {len(filtered_seq_dict)} sequences matching target Tids "
                  f"(out of {len(self.seq_dict)} total).")
            seq_dict = filtered_seq_dict
            
            if not seq_dict:
                print("Warning: No matching sequences found! Please check if your Tids match the Fasta headers.")
                return None
        else:
            seq_dict = self.seq_dict

        print(f"\nStarting Sequence-Based Baseline Calling for {len(seq_dict)} transcripts...")
        for tid, sequence in tqdm(seq_dict.items()):
            
            # 1. 提取并使用序列特征打分
            cands = caller.extract_and_score_candidates(sequence)
            
            # 2. 停止密码子折叠 & NMS 去重 (依靠内部的 sequence score 去重)
            final_cands = caller.collapse_and_nms(cands, iou_threshold=0.3)
            
            # =================================================================
            # 提取表达量 TPM 并生成 transcription_score 基线
            # =================================================================
            gene_id = self.tx2gene.get(tid, tid)
            query_id = tid if self.tpm_level == 'transcript' else gene_id
            
            if self.has_tpm:
                tpm_val = float(self.tpm_dict.get(query_id, 0.0))
                if pd.isna(tpm_val) or tpm_val < 0: tpm_val = 0.0
                log_tpm = np.log2(tpm_val + 1.0)
            else:
                tpm_val = np.nan
                log_tpm = 0.0 # 若无TPM，表达基线得分为 0
            
            # 3. 记录并解绑分数
            for cand in final_cands:
                cand['Tid'] = tid
                # [MODIFIED] 明确分数的用途
                cand['seq_score'] = cand['score'] # 保留序列打分
                cand['transcription_score'] = log_tpm      # 全新的纯表达量打分
                cand['tpm'] = tpm_val
                all_records.append(cand)
                
        if not all_records:
            print("No valid ORFs were found across all transcripts.")
            return pd.DataFrame()
            
        final_df = pd.DataFrame(all_records)
        
        # =================================================================
        # [MODIFIED] 补充导出的列名，默认以 transcription_score 倒序，作为表达量基线的呈现
        # =================================================================
        cols = ['Tid', 'start', 'stop', 'length', 'start_codon', 'seq_score', 'transcription_score', 'tpm']
        final_df = final_df[cols].sort_values(by=['transcription_score', 'seq_score'], ascending=[False, False])
        
        save_path = os.path.join(out_dir, f"baseline_called_orfs.{self.cell_type}.csv")
        final_df.to_csv(save_path, index=False)
        
        print(f"\n🎉 Baseline Calling Completed! Found {len(final_df)} ORFs.")
        print(f"Results saved to: {save_path}")
        
        return final_df