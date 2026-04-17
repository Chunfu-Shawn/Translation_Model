import os
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from tqdm import tqdm

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
                curr_id = line[1:].split()[0].split('|')[0] 
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
                            'score': float(baseline_score) # 必须包含 score 列供下游评估
                        })
                    break 
        return candidates

    def fast_nms(self, cands: List[dict], iou_threshold: float = 0.3) -> List[dict]:
        """Non-Maximum Suppression to resolve spatial overlaps."""
        keep = []
        # 按基线得分降序排列
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
    def __init__(self, fasta_file: str):
        self.fasta_file = fasta_file
        print("Loading Fasta File for Baseline Analysis...")
        self.seq_dict = read_fasta(self.fasta_file)

    def run(self, 
            out_dir: str = "./results/baseline", 
            start_codons: List[str] = ['ATG', 'CTG', 'GTG', 'TTG', 'ACG'],
            target_tids: Optional[list] = None, 
            min_len: int = 30) -> pd.DataFrame:
        
        os.makedirs(out_dir, exist_ok=True)
        caller = BaselineSequenceORFCaller(start_codons=start_codons, min_len=min_len)
        all_records = []
        
        if target_tids is not None:
            target_set = set(target_tids)
            filtered_seq_dict = {}
            for tid, seq in self.seq_dict.items():
                # 同样清理 Fasta 头里的版本号或管道符
                clean_tid = str(tid).split('|')[0] #.split('.')[0]
                if clean_tid in target_set:
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
            
            # 1. 提取并打分
            cands = caller.extract_and_score_candidates(sequence)
            
            # 2. 停止密码子折叠 & NMS 去重
            final_cands = caller.collapse_and_nms(cands, iou_threshold=0.3)
            
            # 3. 记录
            for cand in final_cands:
                cand['Tid'] = tid
                all_records.append(cand)
                
        if not all_records:
            print("No valid ORFs were found across all transcripts.")
            return pd.DataFrame()
            
        final_df = pd.DataFrame(all_records)
        
        # 保留关键列，完美兼容下游的 Precision@K 和 AUC 评估代码
        cols = ['Tid', 'start', 'stop', 'length', 'start_codon', 'score']
        final_df = final_df[cols].sort_values(by=['Tid', 'score'], ascending=[True, False])
        
        save_path = os.path.join(out_dir, "sequence_baseline_called_orfs.csv")
        final_df.to_csv(save_path, index=False)
        
        print(f"\n🎉 Baseline Sequence Calling Completed! Found {len(final_df)} ORFs.")
        print(f"Results saved to: {save_path}")
        
        return final_df