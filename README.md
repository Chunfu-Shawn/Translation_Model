# TRACE: Translation Resolution Across Cell Environments

A Transformer-based model that decodes full-length transcriptomes into translatomes — predicting per-position ribosome density profiles purely from mRNA sequence and cellular context. TRACE integrates multi-omics data — transcript sequence, gene expression, and species identity — through adaptive layer normalization (AdaLN-Zero) to resolve translation regulation across cell types and species.

## Overview

TRACE takes as input:
- **mRNA sequence** (one-hot encoded nucleotides)
- **Cellular transcriptome profile** (continuous expression vector, ~40k genes)
- **Species label** (discrete identifier for evolutionary context)

And decodes the full-length transcript into a translatome — the per-position ribosome density profile — enabling:
- Translation efficiency (TE) estimation
- Ribosome pausing site identification
- Cross-species and cross-cell-type coding ORF prediction

## Model Architecture

```
                    ┌──────────────────────────────┐
                    │  Species Embedding (d=16)     │
                    └──────────────┬───────────────┘
                                   │
┌─────────────┐    ┌───────────────▼───────────────┐
│ mRNA Seq    │    │  Expression Projector          │
│ (one-hot)   │    │  (d_expr + d_species → d_cell) │
└──────┬──────┘    └───────────────┬───────────────┘
       │                           │
       │                           │   compact_style
       │                           │   (adaptive_dim)
       │                           │
┌──────▼──────────────────────────▼┐
│  Linear Embedding                │
│  (seq → d_model)                │
└──────────────┬───────────────────┘
               │
       ┌───────▼───────┐
       │  AdaLN-Zero   │
       │  Transformer   │  ← compact_style modulates
       │  Encoder       │     each sublayer via adaLN
       │  (N layers)    │
       └───────┬───────┘
               │
       ┌───────▼───────┐
       │  Pluggable    │
       │  Prediction   │  ← TranslationProfileHead,
       │  Heads         │    PsiteDensityHead, etc.
       └───────────────┘
```

Key architectural features:
- **AdaLN-Zero**: Each transformer sublayer is modulated by a compact style vector derived from the concatenated expression + species features, with a learned gating parameter initialized to zero for stable training.
- **Rotary Position Embedding (RoPE)**: Applied to query/key in self-attention, with NTK-aware scaling for long sequences.
- **Flash Attention**: Automatic dispatch to FlashAttention-2 when available, with graceful fallback to standard PyTorch attention.
- **Pluggable Heads**: Modular prediction heads (density, coding, decoupled shape/scale) that can be added/removed at runtime.

## Project Structure

```
translation_model/
├── src/
│   ├── model/
│   │   ├── translation_base_model.py   # Core model: TranslationBaseModel
│   │   ├── model_modules.py            # AdaLN-Zero encoder, attention, embedding
│   │   ├── mask_heads.py               # Prediction heads (PsiteDensityHead, etc.)
│   │   ├── position_embedding.py       # RoPE (LlamaRotaryEmbeddingExt)
│   │   ├── flash_multi_headed_attention.py  # FlashAttention-2 wrapper
│   │   ├── translation_predictor.py    # Inference utilities
│   │   ├── coding_decoder.py           # ORF coding potential decoder
│   │   └── orf_caller.py              # ORF identification
│   ├── data/
│   │   ├── translation_dataset_generator.py  # H5 dataset generation pipeline
│   │   ├── translation_dataset.py            # PyTorch Dataset with lazy loading
│   │   ├── RPF_counter_v3.py                 # Ribo-seq read counting
│   │   └── cell_env_expr_array_generate.py   # Expression vector generation
│   ├── train/
│   │   ├── model_pretrain.py                # Pretraining trainer
│   │   ├── model_finetune.py                # Fine-tuning trainer
│   │   ├── masking_adapter.py               # BERT-style masking (curriculum)
│   │   ├── distributed_balanced_bucket_sampler.py  # Length-bucketed DDP sampler
│   │   └── distributed_bucket_sampler.py
│   ├── config/
│   │   ├── model_config_expr.py        # ModelConfig dataclass
│   │   └── *.yaml                      # Hyperparameter configs
│   ├── eval/                            # Evaluation & visualization scripts
│   ├── plot/                            # Plotting utilities
│   ├── process/                         # Data processing scripts
│   ├── run.pretrain.py                  # Pretraining entry point
│   ├── run.fine_tune.py                 # Fine-tuning entry point
│   ├── utils.py                         # Shared utilities
│   └── lora_utils.py                    # LoRA adapter utilities
├── data/                                # Data directory (not tracked)
├── figures/                             # Paper figures
├── results/                             # Experiment results
└── environment.yml                      # Conda environment
```

## Setup

### Environment

```bash
conda env create -f src/environment.yml
conda activate ribo_model
```

Key dependencies:
- Python 3.11
- PyTorch 2.6.0 (CUDA 12.4)
- flash-attn 2.8.0
- h5py, pyahocorasick, pysam, scipy, scikit-learn

### Hardware

- Training: Multi-GPU (tested on 4–8× A100/H100)
- Inference: Single GPU (>=16GB VRAM recommended for 384d model)

## Data Pipeline

### 1. Generate Expression Vectors

```python
from data.cell_env_expr_array_generate import *
# Produces a {cell_type: np.ndarray} dictionary saved as .pt or .pkl
```

### 2. Generate H5 Datasets

```python
from data.translation_dataset_generator import DatasetGenerator

generator = DatasetGenerator(
    transcript_seq_file="path/to/seq_dict.pkl",
    transcript_meta_file="path/to/tx_meta.pkl",
    transcript_cds_file="path/to/tx_cds.pkl",
    chrom_groups={"train": ["chr1-8"], "valid": ["chr9"], "test": ["chr10-X"]},
    species="human",
    motif_file_path="path/to/motifs.txt"
)

generator.generate_save_dataset(
    dataset_config=[
        {"cell_type": "heart", "read_count": "heart_ribo.pkl", "rna_count": "heart_rna.pkl"},
        {"cell_type": "liver", "read_count": "liver_ribo.pkl", "rna_count": "liver_rna.pkl"},
    ],
    depth=0.1, coverage=0.1, rpm=1.0,
    expr_dict_path="cell_expr_dict.pt",
    out_path="dataset"
)
# Produces: dataset.train.h5, dataset.valid.h5, dataset.test.h5
```

### 3. Load Dataset

```python
from data.translation_dataset import TranslationDataset

dataset = TranslationDataset.from_h5("dataset.train.h5", lazy=True)
# lazy=True: on-demand disk I/O (recommended for large datasets)
# lazy=False: load everything into RAM
```

## Training

### Pretraining

```bash
torchrun --nproc_per_node=4 src/run.pretrain.py
```

During pretraining, an auxiliary RPF count signal is fed alongside the sequence and progressively masked (BERT-style curriculum from 10% to 100%), teaching the model to reconstruct density from sequence alone. At deployment, the count input is set to zeros and the model predicts purely from transcriptome.

The pretraining trainer supports:
- **Curriculum masking**: Linearly increasing mask ratio over the auxiliary count signal, so the model gradually learns to predict density without it
- **Multi-strategy masking**: Single-base, trinucleotide, and motif-aware masking
- **Species/cell-type masking**: Randomly dropping species or expression context (15% each)
- **Expression noise injection**: Gaussian noise (std=0.1) added to expression vectors during training
- **Joint micro + macro loss**: Token-level SmoothL1 + Frame-aware MSE + Pairwise ranking loss
- **Mixed precision (BF16)**: Automatic with `torch.amp`
- **Gradient accumulation**: Configurable steps
- **Early stopping**: Patience-based with checkpoint saving

### Configuration

Model hyperparameters are defined in YAML files under `src/config/`:

```yaml
# Example: base_model_expr_384d_16h_12l_128env_32ad.yaml
d_seq: 4
d_count: 1
d_model: 384
d_expr: 40000
d_cell_env: 128
n_heads: 16
number_of_layers: 12
d_ff: 2048
adaptive_dim: 32
p_drop: 0.1
all_species: ["human", "macaque", "mouse"]
d_species: 16
```

### Fine-tuning

Fine-tuning supports LoRA adapters for parameter-efficient transfer learning:

```python
from lora_utils import build_lora_model_from_pretrained
model = build_lora_model_from_pretrained(base_model, r=4, lora_alpha=16)
```

## Inference

TRACE predicts the translatome purely from transcriptome — only mRNA sequence and cellular context are needed.

```python
from model.translation_base_model import TranslationBaseModel
from model.mask_heads import TranslationProfileHead

# Load model
model = TranslationBaseModel.from_config("config.yaml")
model.add_head("count", TranslationProfileHead.create_from_model(model))
model.load_pretrained_weights("checkpoint.pt")

# Predict — decode transcriptome into translatome
result = model.predict(
    seq_batch=seq_array,        # (seq_len, 4) or (bs, seq_len, 4)
    count_batch=None,           # Not needed at deployment (auto-filled with zeros)
    species="human",
    cell_type="heart",          # or expr_vector=torch.Tensor
    head_names=["count"]
)
```

## Evaluation

Evaluation scripts are in `src/eval/`:

| Script | Description |
|--------|-------------|
| `psite_pos_wise_corr_depth.py` | Position-wise correlation vs. sequencing depth |
| `periodicity_corr.py` | 3-nt periodicity correlation |
| `metagene_profile.py` | Metagene analysis around TIS/TTS |
| `te_evaluator.py` | Translation efficiency prediction evaluation |
| `rna_length_effect.py` | Length-dependent prediction analysis |
| `rna_mfe_effect.py` | MFE (folding energy) effect analysis |
| `start_codon_Kozak_motif.py` | Kozak motif effect |
| `uaug_effect.py` / `uorf_effect_batch.py` | uAUG/uORF effect |
| `orf_coding_performance.py` | ORF coding potential benchmark |

## Citation

If you use this code, please cite:

```bibtex
@article{trace2025,
  title={TRACE: Translation Resolution Across Cell Environments},
  author={Xiao, Chunfu},
  year={2025}
}
```

## License

This project is licensed for academic research use. Contact the author for commercial licensing.
