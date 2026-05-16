import os
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from eval.calculate_te import *

class TranslationEfficiencyAnalyzer:
    def __init__(self, transcript_cds_pkl, dataset=None, preds_pkl_path=None, unlog_data=True, use_pred=True):
        """
        Args:
            transcript_cds_pkl: 包含转录本元数据 (如 cds_start_pos, cds_end_pos) 的 PKL 文件路径 (必需项)。
            dataset: TranslationDataset 实例 (与 preds_pkl_path 二选一)。
            preds_pkl_path: 预测结果的 pkl 文件路径 (与 dataset 二选一，优先使用)。
            unlog_data: 是否对数据进行 expm1 还原。
            use_pred: 当读取 pkl 时作为标识。
        """
        self.dataset = dataset
        self.preds_pkl_path = preds_pkl_path
        self.unlog_data = unlog_data
        self.use_pred = use_pred
        self.preds_dict = None
        self.tx_cds = None

        # 1. 验证输入参数
        if dataset is None and preds_pkl_path is None:
            raise ValueError("必须至少提供 dataset 或 preds_pkl_path 中的一个！")

        # 2. 强制加载 Transcript Meta
        print(f"Loading transcript metadata from {transcript_cds_pkl}...")
        with open(transcript_cds_pkl, 'rb') as f:
            self.tx_cds = pickle.load(f)
        # clean tid
        self.tx_cds = {tid.split(".")[0]: values for tid, values in self.tx_cds.items()}

        # 3. 如果提供了预测文件，加载预测结果
        if preds_pkl_path is not None:
            print(f"Loading predictions from {preds_pkl_path}...")
            with open(preds_pkl_path, 'rb') as f:
                self.preds_dict = pickle.load(f)
            print("Loaded prediction records.")

    def run(self, out_dir="./results", suffix=""):
        results = []
        os.makedirs(out_dir, exist_ok=True)
        
        # ==========================================
        # 优先级 1: 遍历 Predictions PKL
        # ==========================================
        if self.preds_dict is not None:
            print("Mode: PKL Predictions (Priority)")
            for cell_type, preds in self.preds_dict.items():
                for transcript_id, count_emb in tqdm(preds.items(), desc=f"Processing {cell_type}"):
                    uuid = f"{transcript_id}-{cell_type}-pred"
                    self._extract_and_run(results, uuid, transcript_id, cell_type, count_emb)
                    
        # ==========================================
        # 优先级 2: 遍历 Dataset Ground Truth
        # ==========================================
        else:
            print("Mode: Dataset Ground Truth")
            for i in tqdm(range(len(self.dataset)), desc="Processing Dataset"):
                try:
                    sample = self.dataset[i]
                    uuid = str(sample[0])
                    parts = uuid.split("-")
                    
                    if len(parts) < 2: 
                        continue
                        
                    transcript_id = parts[0]
                    cell_type = parts[1]
                    count_emb = sample[6]
                    
                    self._extract_and_run(results, uuid, transcript_id, cell_type, count_emb)
                except Exception as e:
                    print(f"Error processing index {i} (UUID: {uuid}): {e}")
                    continue

        # ==========================================
        # 保存与返回
        # ==========================================
        df = pd.DataFrame(results)
        if df.empty:
            print("Warning: No valid records were processed. Returning empty DataFrame.")
            return df
            
        print("\nHead of Result DataFrame:")
        print(df.head())
        
        file_suffix = f".{suffix}" if suffix else ""
        save_path = os.path.join(out_dir, f"translation_efficiency_metrics{file_suffix}.csv")
        df.to_csv(save_path, index=False)
        print(f"Metrics saved to: {save_path}")
        
        return df

    def _extract_and_run(self, results, uuid, transcript_id, cell_type, count_emb):
        """
        统一的信息提取入口：负责从 tx_cds 中查找边界，并触发计算。
        """
        # 带有版本号的回退处理
        lookup_tid = transcript_id
        tid_no_version = transcript_id.split('.')[0]
        if tid_no_version in self.tx_cds:
            lookup_tid = tid_no_version
        else:
            return  # 元数据中完全找不到该转录本，直接跳过
                
        meta = self.tx_cds[lookup_tid]
        
        cds_start = int(meta.get('cds_start_pos', -1))
        cds_end = int(meta.get('cds_end_pos', -1))
        
        if cds_start == -1 or cds_end == -1:
            return  # 没有 CDS 信息的转录本跳过
        else:
            cds_start = cds_start - 1  # 调整为 0-based 索引
            
        self._process_and_append(results, uuid, transcript_id, cell_type, count_emb, cds_start, cds_end)

    def _process_and_append(self, results_list, uuid, transcript_id, cell_type, count_emb, cds_start, cds_end):
        """
        内部核心逻辑：处理 Density 数组并计算翻译指标
        """
        # 转为 Numpy
        if isinstance(count_emb, torch.Tensor):
            density = count_emb.detach().cpu().numpy()
        else:
            density = count_emb

        # 还原 Log
        if self.unlog_data:
            density = np.expm1(density.astype(np.float32))
        
        # 维度压缩兼容性处理 (L, 10) -> (L,) 或 (L, 1) -> (L,)
        if len(density.shape) > 1 and density.shape[1] > 1:
            density_profile = np.sum(density, axis=1)
        elif len(density.shape) > 1 and density.shape[1] == 1:
            density_profile = density.flatten()
        else:
            density_profile = density

        if np.sum(density_profile[cds_start: cds_end]) == 0:
            return None

        # --- 计算指标 ---
        # 注意：需要确保外部已导入以下函数
        te_morf_ratio = calculate_morf_signal_ratio(density_profile, cds_start, cds_end)
        te_morf_mean_ratio = calculate_morf_mean_signal_ratio(density_profile, cds_start, cds_end)
        te_morf_mean_signal = calculate_morf_mean_signal(density_profile, cds_start, cds_end)
        te_total_sum = calculate_sum_signal(density_profile, cds_start, cds_end)
        te_total_mean = calculate_mean_signal(density_profile)

        # --- 记录结果 ---
        results_list.append({
            'UUID': uuid,
            'Tid': transcript_id,
            'Cell_Type': cell_type,
            'mORF_Sum_Ratio': te_morf_ratio,
            'mORF_Mean_Ratio': te_morf_mean_ratio,
            'mORF_Mean_Density': te_morf_mean_signal,
            'mORF_Ribo_Load': te_total_sum,
            'Global_Mean_Density': te_total_mean,
            'Transcript_Length': len(density_profile)
        })