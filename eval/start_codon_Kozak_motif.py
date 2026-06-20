import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from plotnine import *
from scipy.stats import ranksums
from scipy.stats import pearsonr, spearmanr
import itertools

# --- 1. Global Constants & Helper Functions (Kozak Logic) ---

KOZAK_WEIGHTS = {
    '-6': {'A': 0, 'C': -0.03, 'G': 0.05, 'T': -0.02},
    '-5': {'A': -0.06, 'C': 0.04,  'G': -0.01, 'T': 0.03},
    '-4': {'A': 0.07, 'C': 0.08,  'G': -0.06, 'T': -0.09}, 
    '-3': {'A': 0.14, 'C': -0.08,  'G': 0.14, 'T': -0.29},  # Critical
    '-2': {'A': 0.03, 'C': 0.06,  'G': -0.12, 'T': -0.01},
    '-1': {'A': 0.02, 'C': 0.04,  'G': 0.02, 'T': 0.02}, 
    '+4': {'A': -0.02, 'C': -0.09,  'G': 0.12, 'T': -0.02},  # Critical
    '+5': {'A': -0.01, 'C': 0.04,  'G': 0.05, 'T': -0.07},
}

START_CODON_WEIGHTS = {
    'ATG': 1.0,
    'CTG': 0.3,
    'GTG': 0.2,
    'TTG': 0.05
}

def calculate_kozak_score(sequence, start_idx):
    """
    Quantify Kozak sequence strength based on weight matrix.
    Now includes the absolute weight of the Start Codon itself.
    """
    if start_idx < 6 or start_idx + 5 >= len(sequence):
        return None
    
    score = 0
    # Calculate -6 to -1
    for i, pos_name in enumerate(['-6', '-5', '-4', '-3', '-2', '-1']):
        base = sequence[start_idx - 6 + i]
        score += KOZAK_WEIGHTS[pos_name].get(base, 0)
    
    # Calculate +4 to +5
    for i, pos_name in enumerate(['+4', '+5']):
        base = sequence[start_idx + 3 + i]
        score += KOZAK_WEIGHTS[pos_name].get(base, 0)
        
    # Accumulate Start Codon's own influence
    start_codon = sequence[start_idx:start_idx+3]
    score += START_CODON_WEIGHTS.get(start_codon, 0)
    
    return score

def classify_kozak_context(sequence, start_idx):
    """
    Rule-based Kozak strength classification.
    """
    if start_idx < 3 or start_idx + 4 >= len(sequence):
        return None
        
    pos_minus_3 = sequence[start_idx - 3]
    pos_plus_4  = sequence[start_idx + 3]
    
    is_plus4_G = (pos_plus_4 == 'G')
    is_minus3_R = (pos_minus_3 in ['A', 'G'])
    
    if is_plus4_G and is_minus3_R:
        return "Strong (+4G, -3R)"
    elif is_minus3_R:
        return "Moderate (-3R)"     
    elif is_plus4_G:
        return "Moderate (+4G)"   
    else:
        return "Weak"             

# --- 2. Feature Extraction Function ---

# =================================================================
# [MODIFIED] Added target_cell_type parameter to filter data
# =================================================================
def extract_initiation_features(preds, seqs, target_cell_type=None):
    results = []
    target_codons = {'ATG', 'CTG', 'TTG', 'GTG'}
    
    print(f"Scanning full transcripts ...")
    if target_cell_type:
        print(f"Filtering for target cell type: {target_cell_type}")
    
    for cell_type, tid_dict in preds.items():
        # =================================================================
        # [NEW] Skip irrelevant cell types
        # =================================================================
        if target_cell_type is not None and cell_type != target_cell_type:
            continue
            
        print(f"  -> Processing cell type: {cell_type}")
        
        for tid, pred_signal in tqdm(tid_dict.items(), leave=False):
            clean_tid = str(tid).split('.')[0] if str(tid).startswith('ENST') else str(tid).split('|')[0]
            seq_key = tid if tid in seqs else clean_tid
            if seq_key not in seqs: 
                continue
            
            seq_str = seqs[seq_key].upper()
            seq_len = len(seq_str)
            
            if hasattr(pred_signal, 'cpu'):
                pred_signal = pred_signal.cpu().numpy()
            
            pred_arr = np.expm1(np.array(pred_signal).reshape(-1).astype(np.float32))
            limit = min(len(pred_arr), seq_len)
            global_mean = np.mean(pred_arr[:limit]) + 1e-6
            
            for i in range(6, limit - 5):
                codon = seq_str[i : i+3]
                if codon in target_codons:
                    k_class = classify_kozak_context(seq_str, i)
                    if k_class is None: continue
                    
                    k_score = calculate_kozak_score(seq_str, i) # This is already the Total Score
                    p_site_intensity = pred_arr[i] / (np.sum(pred_arr[i-3:i+3]) + global_mean)
                    
                    results.append({
                        'Tid': clean_tid,
                        'Cell_type': cell_type,
                        'Start codon': codon,
                        'Kozak class': k_class,
                        'Kozak score': k_score,
                        'Normalized_Density': p_site_intensity
                    })

    if not results:
        print("Warning: No motifs extracted. Check your data.")
        return pd.DataFrame()

    meta_df = pd.DataFrame(results)
    
    meta_df['Kozak class'] = pd.Categorical(
        meta_df['Kozak class'], 
        categories=["Weak", "Moderate (+4G)", "Moderate (-3R)", "Strong (+4G, -3R)"], 
        ordered=True
    )
    meta_df['Start codon'] = pd.Categorical(
        meta_df['Start codon'],
        categories=['ATG', 'CTG', 'GTG', 'TTG'],
        ordered=True
    )
    
    print(f"Extracted {len(meta_df)} motifs from transcripts.")
    return meta_df

# =================================================================
# Plotting functions remain unchanged
# =================================================================
def plot_global_kozak_correlation_scatter(meta_df, out_dir, suffix=""):
    """
    Scatter plot spanning all start codons to show global correlation 
    between Total Kozak Score (including Start Codon) and Prediction.
    """
    if meta_df.empty: return
    
    # Sample to accelerate rendering
    plot_df = meta_df.sample(10000, random_state=42) if len(meta_df) > 10000 else meta_df

    df_valid = plot_df.dropna(subset=['Kozak score', 'Normalized_Density'])
    if len(df_valid) < 2: return
    
    # Calculate global Spearman R
    r, p = spearmanr(df_valid['Kozak score'], df_valid['Normalized_Density'])
    p_text = f"{p:.2e}" if p < 0.001 else f"{p:.3f}"
    label_text = f"Global Spearman R = {r:.3f}\nP = {p_text}"
    
    anno_df = pd.DataFrame({
        'x': [df_valid['Kozak score'].min()], 
        'y': [df_valid['Normalized_Density'].max()], 
        'label': [label_text]
    })
    
    p = (
        ggplot(plot_df, aes(x='Kozak score', y='Normalized_Density'))
        + geom_point(alpha=0.2, size=1.5, color='#2c3e50', shape='.') 
        
        # Orange trendline
        + geom_smooth(method='lm', color='#E67E22', size=1.5, linetype='dashed', se=False)
        
        + geom_text(
            data=anno_df,
            mapping=aes(x='x', y='y', label='label'),
            ha='left', va='top', 
            size=12, 
            color='black',
            nudge_x=0.05,
            nudge_y=-0.05, 
            inherit_aes=False
        )
        + theme_classic()
        + theme(
            figure_size=(6, 5), 
            axis_title=element_text(size=12)
        )
        + labs(
            x="Total Kozak Score (incl. Start Codon)",
            y="Relative translation signal peak"
        )
    )
    
    plot_path = os.path.join(out_dir, f"global_kozak_score_scatter.{suffix}.pdf")
    p.save(plot_path, width=6, height=5, dpi=300, verbose=False)
    print(f"Saved global correlation scatter plot to {plot_path}")


def plot_kozak_correlation_scatter(meta_df, out_dir, suffix=""):
    """
    Scatter plot faceted by Start Codon. (Preserved original functionality)
    """
    if meta_df.empty: return

    if len(meta_df) > 10000:
        plot_df = meta_df.sample(10000, random_state=42)
    else:
        plot_df = meta_df

    cor_data = []
    for codon, group in plot_df.groupby('Start codon'):
        sub = group.dropna(subset=['Kozak score', 'Normalized_Density'])
        if len(sub) < 2: continue
        
        r, p = spearmanr(sub['Kozak score'], sub['Normalized_Density'])
        p_text = f"{p:.2e}" if p < 0.001 else f"{p:.3f}"
        label_text = f"R = {r:.3f}\nP = {p_text}"
        
        cor_data.append({
            'Start codon': codon,
            'Label': label_text,
            'x': sub['Kozak score'].min(),
            'y': sub['Normalized_Density'].max()
        })
    
    cor_df = pd.DataFrame(cor_data)
    if 'Start codon' in plot_df.columns and isinstance(plot_df['Start codon'].dtype, pd.CategoricalDtype):
         cor_df['Start codon'] = pd.Categorical(cor_df['Start codon'], categories=plot_df['Start codon'].cat.categories)

    p = (
        ggplot(plot_df, aes(x='Kozak score', y='Normalized_Density'))
        + geom_point(alpha=0.1, size=1.5, color='#2c3e50', shape='.') 
        + geom_smooth(method='lm', color='#E67E22', size=1, linetype='dashed', fill='#E67E22', alpha=0.2) 
        + geom_text(
            data=cor_df,
            mapping=aes(x='x', y='y', label='Label'),
            ha='left', va='top', 
            size=10, 
            color='black',
            nudge_x=0.05, 
            nudge_y=-0.05, 
            inherit_aes=False
        )
        + facet_wrap('Start codon', scales='fixed', nrow=1)
        + theme_bw()
        + theme(
            figure_size=(14, 4), 
            strip_background=element_blank(),
            strip_text=element_text(size=12),
            axis_title=element_text(size=12)
        )
        + labs(
            x="Quantified Kozak Score",
            y="Relative translation signal peak"
        )
    )
    
    plot_path = os.path.join(out_dir, f"kozak_score_scatter.{suffix}.pdf")
    p.save(plot_path, width=14, height=4, dpi=300, verbose=False)


def analyze_and_plot_initiation(meta_df, out_dir, suffix=""):
    """
    Perform statistical tests and plot Kozak feature distribution boxplot.
    """
    if meta_df.empty:
        print("DataFrame is empty, skipping plot and stats.")
        return

    os.makedirs(out_dir, exist_ok=True)
    
    df_plot = meta_df.copy()
    df_plot['Log_Density'] = np.log2(df_plot['Normalized_Density'] + 1e-3)
    
    new_order = ["Strong (+4G, -3R)", "Moderate (-3R)", "Moderate (+4G)", "Weak"]
    df_plot['Kozak class'] = pd.Categorical(df_plot['Kozak class'], categories=new_order, ordered=True)
    y_limit = df_plot['Normalized_Density'].quantile(0.99)
    
    df_valid = df_plot.dropna(subset=['Kozak score', 'Normalized_Density'])
    if len(df_valid) > 2:
        r_val, p_val = spearmanr(df_valid['Kozak score'], df_valid['Normalized_Density'])
        p_text = f"{p_val:.1e}" if p_val < 0.001 else f"{p_val:.3f}"
        corr_label = f"Spearman R = {r_val:.3f}\nP = {p_text}"
    else:
        corr_label = ""
        
    anno_df = pd.DataFrame({'x': [4.4], 'y': [y_limit * 0.95], 'label': [corr_label]})

    # --- A. Plotting ---
    p = (
        ggplot(df_plot, aes(x='Start codon', y='Normalized_Density'))
        
        + geom_boxplot(
            aes(color='Kozak class'), 
            fill='white',               
            alpha=1.0,
            width=0.7,
            size=0.8,                   
            outlier_shape=None,         
            outlier_alpha=0,
            outlier_size=0,
            position=position_dodge(width=0.8)
        )

        + geom_smooth(
            aes(group=1), 
            method='lm', 
            color='#FF4500', 
            linetype='dashed', 
            size=1.5, 
            se=False,
            alpha=0.8
        )
        
        + geom_text(
            data=anno_df,
            mapping=aes(x='x', y='y', label='label'),
            inherit_aes=False,
            ha='right', va='top',       
            size=11, 
            color='black'
        )
        
        + theme_classic()
        + theme(
            axis_text_x=element_text(size=12, color='black'),
            axis_text_y=element_text(size=11, color='black'),
            legend_position="top",
            legend_title=element_blank()
        )
        + labs(
            y="Relative translation signal peak"
        )
        + scale_color_manual(values=["#08306B", "#2171B5", "#6BAED6", "#B0B0B0"]) 
        + coord_cartesian(ylim=(0, y_limit))
    )
    
    plot_filename = f"start_codon_kozak_boxplot.{suffix}.pdf" if suffix else "start_codon_kozak_boxplot.pdf"
    plot_path = os.path.join(out_dir, plot_filename)
    p.save(plot_path, width=4, height=5, verbose=False)
    print(f"Saved elegant boxplot to {plot_path}")

    # --- B. Statistics ---
    stats_results = []
    
    codons = ['ATG', 'CTG', 'GTG', 'TTG']
    for c1, c2 in itertools.combinations(codons, 2):
        group1 = df_plot[df_plot['Start codon'] == c1]['Log_Density']
        group2 = df_plot[df_plot['Start codon'] == c2]['Log_Density']
        
        if len(group1) > 0 and len(group2) > 0:
            stat, pval = ranksums(group1, group2)
            stats_results.append({
                'Type': 'Between_Codons',
                'Group1': c1,
                'Group2': c2,
                'Statistic': stat,
                'P_Value': pval
            })

    kozak_levels = ["Strong (+4G, -3R)", "Moderate (-3R)", "Moderate (+4G)", "Weak"]
    for codon in codons:
        subset = df_plot[df_plot['Start codon'] == codon]
        for k1, k2 in itertools.combinations(kozak_levels, 2):
            group1 = subset[subset['Kozak class'] == k1]['Log_Density']
            group2 = subset[subset['Kozak class'] == k2]['Log_Density']
            
            if len(group1) > 0 and len(group2) > 0:
                stat, pval = ranksums(group1, group2)
                stats_results.append({
                    'Type': f'Within_{codon}',
                    'Group1': k1,
                    'Group2': k2,
                    'Statistic': stat,
                    'P_Value': pval
                })

    stats_df = pd.DataFrame(stats_results)
    stats_filename = f"wilcox_stats_results.{suffix}.csv" if suffix else "wilcox_stats_results.csv"
    stats_path = os.path.join(out_dir, stats_filename)
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved statistics to {stats_path}")

# --- 4. Main Execution Function ---

# =================================================================
# [MODIFIED] Added target_cell_type parameter to main function
# =================================================================
def evaluate_start_codon_kozak_motif(pred_pkl, seq_pkl, out_dir="./results/initiation_eval", suffix="", target_cell_type=None):
    """
    Main entry function to evaluate Kozak motif representation.
    """
    print(f"Loading predictions: {pred_pkl}")
    with open(pred_pkl, 'rb') as f:
        preds = pickle.load(f)
        
    print(f"Loading sequences: {seq_pkl}")
    with open(seq_pkl, 'rb') as f:
        seqs = pickle.load(f)
        
    # Pass target_cell_type to extraction logic
    df_features = extract_initiation_features(preds, seqs, target_cell_type=target_cell_type)
    
    if not df_features.empty:
        os.makedirs(out_dir, exist_ok=True)
        csv_filename = f"initiation_features_fullscan.{suffix}.csv" if suffix else "initiation_features_fullscan.csv"
        csv_path = os.path.join(out_dir, csv_filename)
        df_features.to_csv(csv_path, index=False)
        print(f"Features saved to {csv_path}")
        
        analyze_and_plot_initiation(df_features, out_dir, suffix=suffix)
        plot_kozak_correlation_scatter(df_features, out_dir, suffix=suffix)
        plot_global_kozak_correlation_scatter(df_features, out_dir, suffix=suffix)