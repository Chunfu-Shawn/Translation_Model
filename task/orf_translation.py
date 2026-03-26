import os
import pickle
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from plotnine import *
from typing import List, Dict

from task.translation_metrics import compute_pif, compute_uniformity, compute_dropoff


class TranslationEvaluator:
    def __init__(self, pkl_path: str, out_dir: str):
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)
        self.out_dir = out_dir
        self.results_df = None

    def calculate_metrics(self, cell_types=None, window=15, suffix="", eps=0.1):
        records = []
        for uuid, content in tqdm(self.data.items(), desc="Evaluating Known CDS"):
            pred = content['pred']
            cds = content['cds_info']
            
            # 解析 UUID (如: transcriptID-cellType)
            parts = str(uuid).split("-")
            tid, cell_type = parts[0], parts[1] if len(parts) > 1 else "Unknown"

            if cell_types and cell_type not in cell_types:
                continue
            if cds is None:
                continue

            cds_start = cds['start'] - 1 
            cds_end = cds['end']
            if (cds_end - cds_start) < 30: 
                continue

            # 还原 log 变换用于 PIF 和 Uniformity
            cds_sig_real = np.expm1(pred[cds_start: cds_end])
            f1_sig_real = cds_sig_real[0::3]

            # 调用解耦后的工具函数
            pif = compute_pif(cds_sig_real)
            uni = compute_uniformity(f1_sig_real)
            drop = compute_dropoff(pred, cds_start, cds_end, window, eps)

            records.append({'tid': tid, 'cell_type': cell_type, 'PIF': pif, 'Uniformity': uni, 'Drop_off': drop})
        
        self.results_df = pd.DataFrame(records)
        os.makedirs(self.out_dir, exist_ok=True)
        self.results_df.to_csv(os.path.join(self.out_dir, f"translation_eval{suffix}.csv"), index=False)
        return self.results_df

    def plot_distributions(self, perc=0.95, suffix=""):
        file_suffix = f".{suffix}" if suffix else ""

        if self.results_df is None: return
        
        # 转换数据为长表格式，方便 plotnine 绘图
        df_long = self.results_df.melt(
            id_vars=['tid', "cell_type"], 
            value_vars=['PIF', 'Uniformity', 'Drop_off'],
            var_name='Metric', value_name='Score'
        )
        df_long['Metric'] = pd.Categorical(
            df_long['Metric'], 
            categories=['PIF', 'Uniformity', 'Drop_off'], 
            ordered=True
        )

        # 计算每个指标的 95% Cutoff (排除表现最差的 5% 样本)
        cutoffs = df_long.groupby('Metric')['Score'].quantile(1-perc).reset_index()
        cutoffs.columns = ['Metric', 'Cutoff']

        cutoffs['Label'] = cutoffs['Cutoff'].apply(lambda x: f"Cutoff: {x:.3f}")

        # 使用 plotnine 绘图
        p = (
            ggplot(df_long, aes(x='Score'))
            + geom_density(aes(fill='Metric'), alpha=0.6, show_legend=False)
            + geom_vline(data=cutoffs, mapping=aes(xintercept='Cutoff'), 
                         linetype='dashed', color='red', size=0.8)
            + geom_text(data=cutoffs, 
                        mapping=aes(x='Cutoff', y=0.5, label='Label'),
                        va='bottom', ha='right', color='red', size=9, nudge_x=-0.02)
            + facet_wrap('~Metric', scales='free')
            + scale_fill_brewer(type='qual', palette='Set2')
            + theme_classic()
            + theme(
                figure_size=(12, 4),
                panel_spacing=0.02,
                strip_background=element_blank(),
                strip_text=element_text(size=12),
                axis_title=element_text(size=12)
            )
            + labs(
                title="Model Performance Metrics Across Transcripts",
                subtitle="Red line indicates the 5th percentile (retaining 95% of best performing samples)",
                x="Score Value",
                y="Density"
            )
        )
        # 确保输出目录存在
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
            
        out_path = os.path.join(self.out_dir, f"translation_eval{file_suffix}.pdf")
        p.save(out_path, dpi=300)
        return p


class ORFFinderEvaluator:
    def __init__(self, pred_pkl: str, seq_pkl: str, out_dir: str = "./"):
        with open(pred_pkl, 'rb') as f:
            self.pred_data = pickle.load(f)
        with open(seq_pkl, 'rb') as f:
            self.seq_data = pickle.load(f)
        self.out_dir = out_dir

    def find_all_orfs(self, sequence: str, start_codons: List[str], stop_codons: List[str], min_len: int) -> List[Dict]:
        orfs = []
        seq_len = len(sequence)
        for start_motif in start_codons:
            for match in re.finditer(f'(?=({start_motif}))', sequence):
                s_pos = match.start()
                for i in range(s_pos + 3, seq_len - 2, 3):
                    if sequence[i:i+3] in stop_codons:
                        o_len = i + 3 - s_pos
                        if o_len >= min_len:
                            orfs.append({'start': s_pos + 1, 'end': i + 3, 'length': o_len})
                        break
        return orfs

    def evaluate_potential_orfs(self, start_codons = ["ATG"], stop_codons = ["TGA", "TAA", "TAG"],
            min_len=60, thresholds=(0.5, 0.7, 0.7), suffix=""):
        file_suffix = f".{suffix}" if suffix else ""
        p_thr, u_thr, d_thr = thresholds
        results = []
        
        for uuid, content in tqdm(self.pred_data.items(), desc="Scanning Potential ORFs"):
            # 解析 UUID (如: transcriptID-cellType)
            parts = str(uuid).split("-")
            tid, cell_type = parts[0], parts[1] if len(parts) > 1 else "Unknown"

            if tid not in self.seq_data: 
                continue
            
            rna_seq = self.seq_data[tid]
            pred = content['pred'] # 原始预测值 (log 空间)
            
            # 1. 搜索所有潜在 ORF
            potential_orfs = self.find_all_orfs(rna_seq, start_codons, stop_codons, min_len)
            
            for orf in potential_orfs:
                s_idx, e_idx = orf['start'] - 1, orf['end']
                
                # --- 新增：起始密码子必须有信号校验 ---
                # 检查预测的 P-site 密度在起始位置是否大于 0
                # if pred[s_idx] <= 0:
                #     continue

                # 2. 指标计算 (使用解耦后的工具函数)
                # 注意：PIF 和 Uniformity 需还原 log 变换
                cds_sig_real = np.expm1(pred[s_idx:e_idx])
                pif = compute_pif(cds_sig_real)
                uni = compute_uniformity(cds_sig_real[0::3])
                
                # Drop-off 保持原逻辑 (使用 log 空间的 pred)
                drop = compute_dropoff(pred, s_idx, e_idx)

                results.append({
                    'tid': tid.split('.')[0], 
                    'cell_type': cell_type,
                    'start_pos': orf['start'], 
                    'end_pos': orf['end'],
                    'length': orf['length'],
                    'PIF': pif, 
                    'Uniformity': uni, 
                    'Drop_off': drop,
                    'is_passed': (pif >= p_thr and uni >= u_thr and drop >= d_thr)
                })
        
        if not results:
            print("No ORFs found meeting the criteria.")
            return pd.DataFrame()

        # 3. 数据处理与去重
        df = pd.DataFrame(results)
        
        # --- 新增：嵌套 ORF 去重逻辑 ---
        # 按照 tid, cell_type, 和终止位置 (end_pos) 分组，保留最长 (length) 的 ORF
        # 先按长度降序排列，然后根据这三个字段去重，保留第一行（即最长行）
        df_deduped = df.sort_values(by=['tid', 'cell_type', 'end_pos', 'length'], ascending=[True, True, True, False])
        df_deduped = df_deduped.drop_duplicates(subset=['tid', 'cell_type', 'end_pos'], keep='first')

        # 保存完整评估记录（去重后）
        out_csv = os.path.join(self.out_dir, f"potential_orfs_metrics{file_suffix}.csv")
        df_deduped.to_csv(out_csv, index=False)
        
        # 返回筛选通过的高置信度集合
        passed_df = df_deduped[df_deduped['is_passed'] == True]
        print(f"Total Unique ORFs scanned: {len(df_deduped)}")
        print(f"Passed Thresholds: {len(passed_df)}")
        
        return passed_df