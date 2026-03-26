import os
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from eval.calculate_te import *

class TranslationEfficiencyAnalyzer:
    def __init__(self, dataset=None, preds_pkl_path=None, unlog_data=True, use_pred=True):
        """
        Args:
            dataset: TranslationDataset 实例。由于新版 pkl 缺失 cds_info，此项现在为必需项。
            preds_pkl_path: 预测结果的 pkl 文件路径。
            unlog_data: 是否对数据进行 np.expm1 还原。
            use_pred: 当读取 pkl 时作为标识。
        """
        self.dataset = dataset
        self.preds_pkl_path = preds_pkl_path
        self.unlog_data = unlog_data
        self.use_pred = use_pred
        self.preds_dict = None

        if preds_pkl_path is not None:
            print(f"Loading data from {preds_pkl_path}...")
            with open(preds_pkl_path, 'rb') as f:
                self.preds_dict = pickle.load(f)
            print("Loaded records from PKL.")
            
        # 强制要求提供 dataset，因为需要从中获取 cds_info
        if dataset is None:
            raise ValueError("由于新的 PKL 格式不再包含 cds_info，必须提供 dataset 来获取元数据！")

    def run(self, out_dir="./results", suffix=""):
        results = []
        os.makedirs(out_dir, exist_ok=True)
        
        mode_str = "PKL Predictions" if self.preds_dict is not None else "Dataset Ground Truth"
        print(f"Start processing {len(self.dataset)} samples using {mode_str}...")
        
        for i in tqdm(range(len(self.dataset))):
            try:
                # 1. 从 Dataset 提取元数据 (无论使用真值还是预测值，都需要边界)
                sample = self.dataset[i]
                uuid = str(sample[0])
                parts = uuid.split("-")
                
                if len(parts) < 2: 
                    continue
                    
                transcript_id = parts[0]
                cell_type = parts[1]
                
                meta = sample[2]
                te_val = meta.get('te_val', np.nan)
                cds_start = int(meta.get('cds_start_pos', -1))
                cds_end = int(meta.get('cds_end_pos', -1))
                
                if cds_start == -1 or cds_end == -1:
                    continue  # 没有 CDS 信息的转录本跳过
                else:
                    cds_start = cds_start - 1

                # 2. 决定 count_emb 的来源 (PKL 或 Dataset)
                if self.preds_dict is not None:
                    if cell_type not in self.preds_dict:
                        continue
                        
                    predictions = self.preds_dict[cell_type]
                    
                    # 带有版本号的回退处理
                    lookup_tid = transcript_id
                    if lookup_tid not in predictions:
                        tid_no_version = transcript_id.split('.')[0]
                        if tid_no_version in predictions:
                            lookup_tid = tid_no_version
                        else:
                            continue
                            
                    # 取出预测数组
                    count_emb = predictions[lookup_tid]
                else:
                    # 使用 Dataset 中的真实值
                    count_emb = sample[4]
                
                # 3. 核心计算
                self._process_and_append(results, uuid, transcript_id, cell_type, count_emb, cds_start, cds_end, te_val)
                
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

    def _process_and_append(self, results_list, uuid, transcript_id, cell_type, count_emb, cds_start, cds_end, te_val):
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
        te_total_mean = calculate_mean_signal(density_profile)

        # --- 记录结果 ---
        results_list.append({
            'UUID': uuid,
            'Tid': transcript_id,
            'Cell_Type': cell_type,
            'TE': te_val,
            'mORF_Sum_Ratio': te_morf_ratio,
            'mORF_Mean_Ratio': te_morf_mean_ratio,
            'mORF_Mean_Density': te_morf_mean_signal,
            'Global_Mean_Density': te_total_mean,
            'Transcript_Length': len(density_profile)
        })