import os
import json
import yaml
import torch
import torch.nn as nn
import numpy as np
import inspect
from typing import Optional, List, Union, Tuple, Dict, Any
from config.model_config_expr import ModelConfig
from model.model_modules import LinearEmbedding, AdaEncoder, AdaZeroEncoderLayer

__author__ = "Chunfu Xiao"
__version__="1.6.0"
__email__ = "chunfushawn@gmail.com"


class HeadAdapter(nn.Module):
    """
    Wrap an existing head-like module and present canonical signature:
      forward(src_reps, src_mask=None, trg_inputs=None, **kwargs)
    `call_style` controls how to call wrapped module:
      - "src_only": call module(src_reps, src_mask, **kwargs)
      - "decoder_like": call module(src_reps, trg_inputs, src_mask, **kwargs)
      - "custom": use user-supplied callable adapter_fn(src_reps, src_mask, trg_inputs, **kwargs)
    """
    def __init__(self, module: nn.Module, requires_trg_inputs: bool = False, name: Optional[str] = None, adapter_fn=None):
        super().__init__()
        self.module = module
        self.requires_trg_inputs = bool(requires_trg_inputs)
        self.adapter_fn = adapter_fn
        self.name = name or getattr(module, "name", "BaseHead")

    def forward(self, src_reps, src_mask=None, trg_inputs=None, **kwargs):
        if self.adapter_fn is not None:
            return self.adapter_fn(src_reps, src_mask, trg_inputs, **kwargs)

        if self.requires_trg_inputs:
            # assume wrapped module expects (src_reps, src_mask, trg_inputs, ...)
            return self.module(src_reps, src_mask, trg_inputs, **kwargs)
        else:
            # assume wrapped module expects (src_reps, src_mask, ...)
            return self.module(src_reps, src_mask, **kwargs)

class TranslationBaseModel(nn.Module):
    """
    TranslationBaseModel: encoder-based model with pluggable heads and robust input handling.

    Improvements over the original:
      - Uses continuous Transcriptome Profile (d_expr) instead of discrete cell IDs.
      - Integrates a discrete `species` label to capture evolutionary translation baselines.
      - Includes a Bottleneck Projector to compress massive d_expr to d_cell_env.
      - Can automatically lookup `expr_vector` from a registered dictionary via `cell_type` string.
      - Safe fallback to a mean expression vector for unknown cell types.
    """
    def __init__(
        self,
        d_seq: int,
        d_count: int,
        d_model: int,
        d_expr: int = 40000,
        d_cell_env: int = 64,
        all_species: List[str] = ["human", "macaque", "mouse"],
        d_species: int = 16,
        n_heads: int = 8,
        number_of_layers: int = 12,
        d_ff: int = 2048,
        adaptive_dim: int = 32,
        gamma_scale: float = 0.5,
        p_drop: float = 0.1,
        model_name: str = "base_model",
    ):
        super().__init__()
        # store raw config portion on the instance (handy later)
        self._constructor_args = dict(
            d_seq=d_seq,
            d_count=d_count,
            d_model=d_model,
            d_expr=d_expr,
            d_cell_env=d_cell_env,
            all_species=all_species, 
            d_species=d_species,
            n_heads=n_heads,
            number_of_layers=number_of_layers,
            d_ff=d_ff,
            adaptive_dim=adaptive_dim,
            gamma_scale=gamma_scale,
            p_drop=p_drop,
            model_name=model_name
        )
        
        self.model_name = model_name
        self.d_seq = d_seq
        self.d_count = d_count
        self.d_model = d_model
        self.d_expr = d_expr
        self.d_cell_env = d_cell_env
        self.d_species = d_species

        self.n_heads = n_heads
        self.adaptive_dim = adaptive_dim
        self.gamma_scale = gamma_scale

        # ==========================================
        # Dynamic Species Dictionary Mapping
        # ==========================================
        self.all_species = all_species if all_species else []
        # Total classes = defined species + 1 (reserved index 0 for 'unknown')
        self.num_species = len(self.all_species) + 1
        # Create case-insensitive mapping: e.g., {"human": 1, "macaque": 2, "mouse": 3}
        self.species_mapping = {sp.lower(): idx + 1 for idx, sp in enumerate(self.all_species)}

        # embeds source data into high-dimensional potent embedding vectors
        self.src_emb = LinearEmbedding(self.d_seq, self.d_count, self.d_model)

        # ==========================================
        # Species Embedding Layer
        # Maps discrete species ID (e.g., 0 for Human, 1 for Mouse) to a dense vector
        # Index 0 is reserved for 'Unknown/Generic' species
        # ==========================================
        self.species_embedding = nn.Embedding(self.num_species, self.d_species, padding_idx=0)

        # ==========================================
        # Gene expression + Species projector
        # Now accepts d_expr + d_species as input to the bottleneck
        # ==========================================
        self.expr_projector = nn.Sequential(
            nn.Dropout(min(p_drop * 2, 0.9)),
            nn.Linear(self.d_expr + self.d_species, self.d_cell_env, bias=False),
            nn.LayerNorm(self.d_cell_env),
            nn.GELU(),
            nn.Linear(self.d_cell_env, self.adaptive_dim)
        )

        # Pass num_classes to Encoder/Layer for AdaLayerNorm embedding initialization
        encoder_layer = AdaZeroEncoderLayer(d_model, d_ff, n_heads, p_drop, self.adaptive_dim, self.gamma_scale)
        self.encoder = AdaEncoder(encoder_layer, number_of_layers)

        # pluggable heads
        self.heads = nn.ModuleDict() # key -> head module
        
        # initialize parameters (Xavier for weight tensors by default)
        self._init_parameters()

        self.register_buffer("mean_expr_vector", torch.zeros(self.d_expr)) 
        self.cell_expr_dict = {} 

    # -----------------------------
    # Expression Dictionary Manager
    # -----------------------------
    def load_expression_dict(self, expr_dict: Dict[str, Union[torch.Tensor, np.ndarray, list]]):
        """
        Load a dictionary mapping cell type names to their expression vectors.
        It automatically calculates the mean vector as a fallback.
        """
        self.cell_expr_dict = {}
        all_vectors = []
        for cell_name, vec in expr_dict.items():
            tensor_vec = self._ensure_tensor(vec, dtype=torch.float32)
            if tensor_vec.shape[-1] != self.d_expr:
                raise ValueError(f"Vector for {cell_name} has wrong dimension. Expected {self.d_expr}, got {tensor_vec.shape[-1]}")
            
            # 挤压多余的维度，确保是一维向量
            tensor_vec = tensor_vec.view(-1)
            self.cell_expr_dict[cell_name] = tensor_vec
            all_vectors.append(tensor_vec)
        
        # 计算并更新 fallback 使用的平均表达向量
        if all_vectors:
            stacked_vecs = torch.stack(all_vectors)
            mean_vec = stacked_vecs.mean(dim=0)
            # 使用 copy_ 保证 Buffer 正常更新，且与模型在同一设备
            self.mean_expr_vector.copy_(mean_vec)
            print(f"[Model] Successfully loaded {len(self.cell_expr_dict)} cell type expression profiles. Mean vector updated.")
        else:
            print("[Model] Warning: Provided expression dictionary is empty.")

    # -------------------------
    # initialization utilities
    # -------------------------
    def _init_parameters(self, default_initialization=False):
        """
        Initialize parameters. By default uses Xavier uniform for parameters that have dim > 1.
        If `default_initialization` is True, skip custom initialization (use PyTorch defaults).
        """
        if not default_initialization:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    if m.elementwise_affine: # Only init if learnable
                        if m.weight is not None:
                            nn.init.ones_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device('cpu')

    # -------------------------
    # Config helpers
    # -------------------------
    @classmethod
    def _load_config_from_file(cls, path: str) -> Dict[str, Any]:
        """
        Load configuration from a JSON or YAML file and return as a dict.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        lower = path.lower()
        if lower.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        elif lower.endswith((".yaml", ".yml")):
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        else:
            # attempt to parse as JSON first, then YAML
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            try:
                cfg = json.loads(text)
            except Exception:
                cfg = yaml.safe_load(text)
        if not isinstance(cfg, dict):
            raise ValueError("Config file must contain a JSON/YAML object (mapping).")
        return cfg

    @classmethod
    def _normalize_config(cls, config: Union[Dict[str, Any], ModelConfig, str]) -> ModelConfig:
        """
        Normalize user-provided configuration (dict, ModelConfig or filepath) into a ModelConfig instance.
        """
        if isinstance(config, ModelConfig):
            return config

        if isinstance(config, str):
            cfg_dict = cls._load_config_from_file(config)
        elif isinstance(config, dict):
            cfg_dict = config
        else:
            raise TypeError("config must be a dict, ModelConfig, or a path to a JSON/YAML file.")

        # Basic validation & required keys
        required = {"d_seq", "d_count", "d_model", "d_expr", "d_cell_env"}
        missing = required - cfg_dict.keys()
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")

        # coerce types & use defaults where not given
        return ModelConfig(
            d_seq=int(cfg_dict["d_seq"]),
            d_count=int(cfg_dict["d_count"]),
            d_model=int(cfg_dict["d_model"]),
            d_expr=int(cfg_dict["d_expr"]),
            d_cell_env=int(cfg_dict["d_cell_env"]),
            all_species=list(cfg_dict.get("all_species", ["human", "mouse", "macaque"])),
            d_species=int(cfg_dict.get("d_species", 16)),
            n_heads=int(cfg_dict.get("n_heads", 8)),
            number_of_layers=int(cfg_dict.get("number_of_layers", 6)),
            d_ff=int(cfg_dict.get("d_ff", 2048)),
            adaptive_dim=int(cfg_dict.get("adaptive_dim", 32)),
            gamma_scale=float(cfg_dict.get("gamma_scale", 0.5)),
            p_drop=float(cfg_dict.get("p_drop", 0.1)),
            expr_dict_path=cfg_dict.get("expr_dict_path", None), 
            model_name=cfg_dict.get("model_name"),
            seed=cfg_dict.get("seed"),
        )

    @classmethod
    def from_config(cls, config: Union[Dict[str, Any], ModelConfig, str]) -> "TranslationBaseModel":
        cfg = cls._normalize_config(config)

        if cfg.seed is not None:
            torch.manual_seed(int(cfg.seed))

        print(f"==> Create model using config from {config}: {cfg}")

        # 1. 实例化模型
        model = cls(
            d_seq=cfg.d_seq,
            d_count=cfg.d_count,
            d_model=cfg.d_model,
            d_expr=cfg.d_expr,
            d_cell_env=cfg.d_cell_env,
            all_species=cfg.all_species,
            d_species=cfg.d_species,
            n_heads=cfg.n_heads,
            number_of_layers=cfg.number_of_layers,
            d_ff=cfg.d_ff,
            adaptive_dim=cfg.adaptive_dim,
            gamma_scale=cfg.gamma_scale,
            p_drop=cfg.p_drop,
            model_name=cfg.model_name
        )

        if cfg.expr_dict_path is not None:
            if not os.path.isfile(cfg.expr_dict_path):
                print(f"[Warning] expr_dict_path '{cfg.expr_dict_path}' not found. Skipping auto-load.")
            else:
                print(f"[Model] Auto-loading expression dict from {cfg.expr_dict_path}...")
                
                # 兼容多种格式的字典文件 (JSON / Pickle / PyTorch .pt)
                if cfg.expr_dict_path.endswith('.json'):
                    import json
                    with open(cfg.expr_dict_path, 'r') as f:
                        expr_dict = json.load(f)
                elif cfg.expr_dict_path.endswith(('.pkl', '.pickle')):
                    import pickle
                    with open(cfg.expr_dict_path, 'rb') as f:
                        expr_dict = pickle.load(f)
                elif cfg.expr_dict_path.endswith('.pt'):
                    expr_dict = torch.load(cfg.expr_dict_path, map_location='cpu')
                else:
                    raise ValueError("Unsupported file format for expr_dict_path. Use .json, .pkl, or .pt")
                
                # 调用模型内置的方法挂载数据
                model.load_expression_dict(expr_dict)

        return model

    def save_config(self, path: str, as_yaml: bool = False) -> None:
        """
        Save the model's constructor args (basic config) to a JSON or YAML file so experiments are reproducible.
        """
        cfg = dict(self._constructor_args)
        # add optional metadata if present
        try:
            if hasattr(self, "model_name") and self.model_name:
                cfg["model_name"] = self.model_name
        except Exception:
            pass

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if as_yaml or path.lower().endswith((".yaml", ".yml")):
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
                
    # -------------------------
    # Input normalization helpers
    # -------------------------
    def _ensure_tensor(self, x: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Convert numpy/number/list/tensor to torch.Tensor (CPU). Does not move to device.
        """
        if isinstance(x, torch.Tensor):
            tensor = x
        elif isinstance(x, np.ndarray):
            tensor = torch.from_numpy(x)
        elif isinstance(x, (list, tuple)):
            tensor = torch.tensor(x)
        else:
            # scalar (int/float)
            tensor = torch.tensor(x)
        if dtype is not None:
            tensor = tensor.type(dtype)
        return tensor
    
    def _ensure_batch_dim_input(
        self,
        seq_batch: Any,
        count_batch: Any,
        src_mask: Optional[Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """
        Ensure inputs have a batch dimension. Accepts:
          - seq_batch shape (bs, seq_len, d_seq) or (seq_len, d_seq) for single sample.
          - count_batch shape (bs, seq_len, d_count), (seq_len, d_count) or None.
          - src_mask shape (bs, seq_len) or (seq_len,) or None.
          - If `device` provided, tensors are moved to that device before return

        Returns:
          - seq_batch_batched, count_batch_batched, src_mask_batched, was_squeezed
            where was_squeezed=True means original inputs were single-sample and we added batch dim.
        """
        was_squeezed = False
        
        # If input is numpy or list, convert first (safe to call _ensure_tensor)
        if not isinstance(seq_batch, torch.Tensor):
            seq_batch = self._ensure_tensor(seq_batch)

        # if count_batch is None，generate Tensor full of 0
        if count_batch is None:
            shape = list(seq_batch.shape)
            d_count = getattr(self, 'd_count', 1)
            shape[-1] = d_count
            count_batch = torch.zeros(shape, dtype=seq_batch.dtype)

        # if count_batch is not None
        if not isinstance(count_batch, torch.Tensor):
            count_batch = self._ensure_tensor(count_batch)

        # If both are 2D (seq_len, feature), add batch dim
        if seq_batch.dim() == 2:
            # single sample case: (seq_len, d_seq) -> (1, seq_len, d_seq)
            seq_batch = seq_batch.unsqueeze(0)
            was_squeezed = True
        elif seq_batch.dim() == 1:
            raise ValueError("seq_batch must be at least 2D: (seq_len, d_seq) or (bs, seq_len, d_seq)")

        if count_batch.dim() == 2:
            # single sample case: (seq_len, d_count) -> (1, seq_len, d_count)
            count_batch = count_batch.unsqueeze(0)
            was_squeezed = True
        elif count_batch.dim() == 1:
            raise ValueError("count_batch must be at least 2D: (seq_len, d_count) or (bs, seq_len, d_count)")

        # src_mask may be None, or shape (bs, seq_len) or (seq_len,) or (bs, seq_len, 1)
        if src_mask is None:
            # infer mask from seq_batch assuming padded value = -1 for your pipeline (consistent with collate)
            # seq_batch shape: (bs, seq_len, feature); valid if any dim != -1
            pad_masks = (seq_batch != -1).any(dim=-1)
            src_mask_batched = pad_masks
        else:
            if not isinstance(src_mask, torch.Tensor):
                src_mask = self._ensure_tensor(src_mask, dtype=torch.bool)
            # if shape (seq_len,), make it (1, seq_len)
            if src_mask.dim() == 1:
                src_mask_batched = src_mask.unsqueeze(0)
                was_squeezed = True
            else:
                src_mask_batched = src_mask

            # ensure boolean mask
            if src_mask_batched.dtype != torch.bool:
                src_mask_batched = src_mask_batched.bool()

        # final sanity checks: seq_len and mask lengths should match
        if seq_batch.shape[1] != src_mask_batched.shape[1]:
            raise ValueError(
                f"Sequence length mismatch: seq len {seq_batch.shape[1]} vs mask len {src_mask_batched.shape[1]}"
            )

        return seq_batch, count_batch, src_mask_batched, was_squeezed
    
    def _resolve_expr_vector(self, cell_type: Any, expr_vector: Any, batch_size: int) -> torch.Tensor:
        """
        Resolves the final (bs, d_expr) continuous tensor to be used by the model.
        Priority:
          1. Explicit `expr_vector` provided by user (e.g. during training with Dataloader).
          2. Lookup from `cell_type` strings using internal dictionary.
          3. Fallback to `self.mean_expr_vector` for unknowns or missing inputs.
        """
        # Case 1: User explicitly provided the raw expression vector
        if expr_vector is not None:
            tensor_expr = self._ensure_tensor(expr_vector, dtype=torch.float32)
            if tensor_expr.dim() == 1: # Single sample case
                if tensor_expr.shape[0] != self.d_expr:
                    raise ValueError(f"expr_vector has wrong dimension {tensor_expr.shape[0]}, expected {self.d_expr}")
                return tensor_expr.unsqueeze(0).expand(batch_size, -1).clone()
            elif tensor_expr.dim() == 2: # Batched case
                if tensor_expr.shape != (batch_size, self.d_expr):
                    raise ValueError(f"expr_vector has wrong shape {tuple(tensor_expr.shape)}, expected {(batch_size, self.d_expr)}")
                return tensor_expr
            else:
                raise ValueError("Unsupported expr_vector shape.")

        # Case 2: User provided cell_type identifier, we need to look it up
        # We need to construct a (bs, d_expr) tensor
        output_expr_batch = torch.empty((batch_size, self.d_expr), dtype=torch.float32, device=self.mean_expr_vector.device)
        
        # Helper to fetch a single vector
        def _get_vec(name):
            if isinstance(name, str) and name in self.cell_expr_dict:
                return self.cell_expr_dict[name].to(self.mean_expr_vector.device)
            # Fallback for unknown
            return self.mean_expr_vector

        if cell_type is None:
             output_expr_batch[:] = self.mean_expr_vector
        elif isinstance(cell_type, str):
             vec = _get_vec(cell_type)
             output_expr_batch[:] = vec
        elif isinstance(cell_type, (list, tuple, np.ndarray)):
             if len(cell_type) == 1:
                 vec = _get_vec(cell_type[0] if isinstance(cell_type, list) else cell_type.item())
                 output_expr_batch[:] = vec
             elif len(cell_type) == batch_size:
                 for i, ct in enumerate(cell_type):
                     output_expr_batch[i] = _get_vec(ct)
             else:
                 raise ValueError(f"Length of cell_type array ({len(cell_type)}) must match batch_size ({batch_size}) or 1.")
        else:
             # Unrecognized type, use mean
             output_expr_batch[:] = self.mean_expr_vector
             
        return output_expr_batch
    
    # ==========================================
    # String-Aware Species Normalization
    # Resolves strings/lists into batched Long indices using species_mapping
    # ==========================================
    def _normalize_species(self, species: Any, batch_size: int) -> torch.LongTensor:
        """
        Normalize species inputs (strings, ints, lists) into a long tensor of shape (batch_size,).
        If unknown or None -> returns 0 (Generic/Fallback).
        """
        # Helper to convert a single item (str or int) to the correct dictionary index
        def _single_to_index(val) -> int:
            if isinstance(val, str):
                return self.species_mapping.get(val.lower(), 0)
            try:
                ival = int(val)
                if 0 <= ival < self.num_species:
                    return ival
            except Exception:
                pass
            return 0

        if species is None:
            return torch.zeros((batch_size,), dtype=torch.long)
            
        if isinstance(species, str):
            idx = _single_to_index(species)
            return torch.full((batch_size,), idx, dtype=torch.long)
            
        if isinstance(species, (list, tuple, np.ndarray)):
            if len(species) == 1:
                idx = _single_to_index(species[0] if isinstance(species, list) else species.item())
                return torch.full((batch_size,), idx, dtype=torch.long)
            if len(species) == batch_size:
                mapped = [_single_to_index(x) for x in species]
                return torch.tensor(mapped, dtype=torch.long)
            raise ValueError(f"Length of species array ({len(species)}) must match batch_size ({batch_size}) or 1.")
            
        if isinstance(species, torch.Tensor):
            if species.numel() == 1:
                idx = _single_to_index(species.item())
                return torch.full((batch_size,), idx, dtype=torch.long)
            if species.dim() == 1 and species.numel() == batch_size:
                return species.to(dtype=torch.long).clamp(0, self.num_species - 1)

        # Fallback for unexpected scalars
        idx = _single_to_index(species)
        return torch.full((batch_size,), idx, dtype=torch.long)
    
    # -------------------------
    # Forward / predict
    # -------------------------
    def forward(
        self,
        seq_batch: torch.Tensor,
        count_batch: torch.Tensor,
        cell_type: Optional[Any] = None, 
        expr_vector: torch.Tensor = None,
        species: Optional[Any] = None,
        src_mask: Optional[torch.Tensor] = None,
        head_names: Optional[List[str]] = None, 
        head_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs
    ):
        """
        Strict forward.
        You must provide EITHER `expr_vector` (shape: bs, d_expr) OR `cell_type` (for dict lookup).
        Optionally provide `species` (shape: bs) to inject evolutionary baselines.
        """

        # --- basic type checks ---
        if not isinstance(seq_batch, torch.Tensor):
            raise TypeError("forward() expects seq_batch as torch.Tensor (dim==3). Use predict() for flexible inputs.")
        if not isinstance(count_batch, torch.Tensor):
            raise TypeError("forward() expects count_batch as torch.Tensor (dim==3). Use predict() for flexible inputs.")
        if not isinstance(expr_vector, torch.Tensor):
            raise TypeError("forward() expects expr_vector as torch.Tensor. Use predict() for flexible inputs.")
        if seq_batch.dim() != 3:
            raise ValueError(f"seq_batch must have dim==3 (bs, seq_len, d_seq). Got shape {tuple(seq_batch.shape)}")
        if count_batch.dim() != 3:
            raise ValueError(f"count_batch must have dim==3 (bs, seq_len, d_count). Got shape {tuple(count_batch.shape)}")
        
        bs = seq_batch.shape[0]
        seq_len = seq_batch.shape[1]

        # 1. Resolve Dynamic Cellular Environment (Transcriptome)
        final_expr_vector = self._resolve_expr_vector(cell_type, expr_vector, bs)
        final_expr_vector = final_expr_vector.to(seq_batch.device)

        # 2. Resolve Static Evolutionary Baseline (Species)
        species_idx_batch = self._normalize_species(species, bs).to(seq_batch.device)
        species_embs = self.species_embedding(species_idx_batch) # -> (bs, d_species)

        # 3. Concatenate Transcriptome + Species before the bottleneck projection
        # This allows the projector to interpret expression levels in the context of the specific species
        combined_env_features = torch.cat([final_expr_vector, species_embs], dim=-1) # -> (bs, d_expr + d_species)

        # src_mask validation if provided
        if src_mask is not None:
            if not isinstance(src_mask, torch.Tensor):
                raise TypeError("src_mask must be torch.Tensor (bool).")
            if src_mask.dim() != 2:
                raise ValueError("src_mask must have shape (bs, seq_len).")
            if src_mask.shape[0] != bs or src_mask.shape[1] != seq_len:
                raise ValueError(f"src_mask shape {tuple(src_mask.shape)} does not match seq_batch shape {tuple(seq_batch.shape[:2])}.")
            # ensure boolean dtype
            if src_mask.dtype != torch.bool:
                src_mask = src_mask.bool()
        
        # compute embeddings (expects shapes (bs, seq_len, d_seq) & (bs, seq_len, d_count))
        src_embs = self.src_emb(seq_batch, count_batch)  # -> (bs, seq_len, d_model)

        # 4. Create the final Compact Style from the concatenated features
        compact_style = self.expr_projector(combined_env_features) # -> (bs, adaptive_dim)

        # encode (pass compact_style for AdaLayerNorm)
        src_reps = self.encoder(src_embs, src_mask, compact_style)

        # if user requested no heads, return raw representations (optionally squeeze later)
        if not head_names:
            return src_reps

        # run requested heads
        outputs = {}
        head_inputs = head_inputs or {}  # map: head_name -> dict of kwargs for that head

        for name in head_names:
            if name not in self.heads:
                raise KeyError(f"Head {name} not found. Available: {list(self.heads.keys())}")
            head = self.heads[name]
            # per-head explicit inputs (highest priority)
            per_head_kwargs = dict(head_inputs.get(name, {}))

            # if head needs trg_inputs, look in per_head_kwargs or fallback to top-level kwargs e.g., kwargs.get('trg_inputs')
            if getattr(head, "requires_trg_inputs", False):
                if "trg_inputs" not in per_head_kwargs:
                    # fallback to common key
                    if "trg_inputs" in kwargs:
                        per_head_kwargs["trg_inputs"] = kwargs["trg_inputs"]
                    elif "coding_embeddings" in kwargs:
                        per_head_kwargs["trg_inputs"] = kwargs["coding_embeddings"]
                    else:
                        raise ValueError(f"Head '{name}' requires trg_inputs but none provided. Pass in head_inputs['{name}']['trg_inputs']=... or kwargs['trg_inputs']")

            # finally call the head with canonical signature
            outputs[name] = head(src_reps, src_mask=src_mask, **per_head_kwargs)

        return outputs
    
    # -------------------------
    # predict: flexible, preprocess & call forward
    # -------------------------
    def predict(
        self,
        seq_batch: Union[torch.Tensor, np.ndarray, list, tuple],
        count_batch: Union[torch.Tensor, np.ndarray, list, tuple] = None,
        cell_type: Any = None,
        expr_vector: Any = None,
        species: Any = None,
        src_mask: Optional[Union[torch.Tensor, np.ndarray, list]] = None,
        head_names: Optional[List[str]] = None,
        head_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
        move_inputs_to_device: bool = True,
        return_numpy: bool = False,
    ):
        """
        Flexible inference helper:
          - accepts numpy/list/scalar and single-sample shapes
          - prepares batched torch.Tensor inputs (CPU)
          - calls strict forward(...) which will move tensors to model device
          - squeezes single-sample batch dim when returning
        """
        # 1) set model to eval and disable grad
        self.eval()
        model_device = self.device

        # 2). normalize inputs and preserve whether user passed a single sample
        seq_batch, count_batch, src_mask, was_squeezed = self._ensure_batch_dim_input(seq_batch, count_batch, src_mask)
        bs = seq_batch.shape[0]

        # process expr_vector and cell_type
        final_expr = self._resolve_expr_vector(cell_type, expr_vector, bs)

        # resolve species
        species_idx = self._normalize_species(species, bs)

        # 3) move tensor inputs to model device 
        if move_inputs_to_device:
            # move only if input is a torch.Tensor (avoid converting np/list here)
            if isinstance(seq_batch, torch.Tensor):
                seq_batch = seq_batch.to(model_device)
            if isinstance(count_batch, torch.Tensor):
                count_batch = count_batch.to(model_device)
            if isinstance(src_mask, torch.Tensor):
                src_mask = src_mask.to(model_device)
            species_idx = species_idx.to(model_device)

        # 4) run forward under no_grad
        with torch.no_grad():
            outputs = self.forward(
                seq_batch=seq_batch, 
                count_batch=count_batch, 
                expr_vector=final_expr, 
                species=species_idx,
                src_mask=src_mask, 
                head_names=head_names, 
                head_inputs=head_inputs
            )

        # 5) If forward produced batched outputs but input was single-sample,
        #    forward() should already support returning batched output consistently.
        #    We'll only optionally squeeze the leading batch dimension if present AND if
        #    the caller passed single-sample-shaped inputs (we detect that by checking dims).
        def _squeeze_and_convert(obj):
            if isinstance(obj, torch.Tensor):
                # if batch dim == 1, squeeze it for convenience
                if was_squeezed and obj.shape[0] == 1:
                    obj = obj.squeeze(0)
                return obj.cpu().numpy() if return_numpy else obj
            if isinstance(obj, dict):
                return {k: _squeeze_and_convert(v) for k, v in obj.items()}
            return obj

        return _squeeze_and_convert(outputs)

    
    # -------------------------
    # head management helpers
    # -------------------------
    def add_head(self, name: str, 
                 head_module: nn.Module, 
                 overwrite: bool = False, 
                 move_to_model_device: bool = True, 
                 requires_trg_inputs: Optional[bool] = None) -> None:
        """
        Register a new head module into self.heads and optionally move it to the same device as the base model.

        Parameters
        ----------
        name : str
            Name used to register the head in self.heads (key of ModuleDict).
        head_module : nn.Module
            The head module to add.
        overwrite : bool
            If True replace an existing head with the same name.
        move_to_model_device : bool
            If True, move 'head_module' to the same device as this model's parameters (first param's device).
            This makes simple usage like `base_model.add_head(...); base_model.cuda(rank)` unnecessary.
        """
        # check duplicate
        if (name in self.heads) and (not overwrite):
            raise KeyError(f"Head {name} exists. use overwrite=True to replace.")

        # Optionally move head to the same device as the model
        if move_to_model_device:
            # try to determine device of the model by inspecting its parameters
            try:
                model_device = self.device
            except StopIteration:
                # model has no parameters? fallback to CPU
                model_device = torch.device("cpu")
            # move the head module (this moves parameters and buffers)
            head_module.to(model_device)

        # autodetect requires_trg_inputs if not provided
        if requires_trg_inputs is None:
            # check attribute on module
            if hasattr(head_module, "requires_trg_inputs"):
                requires_trg_inputs = bool(getattr(head_module, "requires_trg_inputs"))
            else:
                # inspect signature as fallback
                try:
                    sig = inspect.signature(head_module.forward)
                    requires_trg_inputs = "trg_inputs" in sig.parameters
                except (ValueError, TypeError):
                    requires_trg_inputs = False

        # if head_module not following HeadAdapter interface, wrap it
        if not isinstance(head_module, HeadAdapter):
            head_module = HeadAdapter(head_module, requires_trg_inputs=requires_trg_inputs, name=getattr(head_module, "name", name))


        # register the head into ModuleDict (this makes it a tracked submodule)
        self.heads[name] = head_module
        self.model_name = f'{self.model_name}-{head_module.name}'


    def remove_head(self, name: str) -> None:
        """remove head from ModuleDict (replease parameters) """
        if name in self.heads:
            self.model_name = self.model_name.replace("-" + self.heads[name].name, '')
            del self.heads[name]
        else:
            raise KeyError(f"Head {name} does not exist.")

    def list_heads(self)-> List[str]:
        return list(self.heads.keys())

    def save_head(self, name: str, path: str) -> None:
        if name not in self.heads:
            raise KeyError(name)
        torch.save(self.heads[name].state_dict(), path)

    def load_head(self, name: str, path: str, map_location: Optional[str] = None, overwrite: bool = False) -> None:
        # if head dont exist, add a same head，then load
        if name not in self.heads and not overwrite:
            raise KeyError(f"Head {name} missing: create it via add_head before load.")
        map_loc = map_location or ("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(path, map_location=map_loc)
        if name not in self.heads:
            raise KeyError("Head not present. Please add a matching head module first.")
        self.heads[name].load_state_dict(state)

    def load_pretrained_weights(self, ckpt_path: Optional[str], map_location: Optional[str] = None, strict: bool = False):
        """
        Load a pretrained checkpoint into this model instance.

        Recommendation:
        - Call with strict=False to allow missing keys (LoRA params will be missing).
        - Prefer to call this BEFORE wrapping model with DistributedDataParallel.

        Args:
        ckpt_path: path to checkpoint file (can be full checkpoint dict {'model': sd, ...} or raw state dict)
        map_location: torch.load map_location (default 'cuda' if available)
        strict: pass-through to load_state_dict
        Returns:
        load_result (NamedTuple returned by load_state_dict) or None if no ckpt_path.
        """
        if ckpt_path is None:
            return None
        map_loc = map_location or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=map_loc)
        # accept different checkpoint layouts:
        if isinstance(ckpt, dict) and "model" in ckpt:
            sd = ckpt["model"]
        elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            # maybe it's directly a state_dict
            sd = ckpt
        else:
            raise ValueError(f"Unsupported checkpoint format for {ckpt_path}")

        # If this model is possibly wrapped in DDP externally, try to load into unwrapped module
        try:
            # if self is wrapper-like (has module attribute), prefer the real module
            target = self.module if hasattr(self, "module") else self
        except Exception:
            target = self

        load_res = target.load_state_dict(sd, strict=strict)
        # report helpful info
        missing = load_res.missing_keys if hasattr(load_res, "missing_keys") else None
        unexpected = load_res.unexpected_keys if hasattr(load_res, "unexpected_keys") else None
        print(f"[model] load_pretrained_weights: path={ckpt_path}, strict={strict}, all={all}, missing={missing}, unexpected={unexpected}")
        return load_res

    def load_lora_and_heads(self, ckpt_path: Optional[str], map_location: Optional[str] = None, strict: bool = False):
        """
        Load a LoRA+heads checkpoint (a partial checkpoint containing only adapter/head tensors).
        Recommendation:
        - Inject LoRA adapters into the model BEFORE calling this if you are going to use LoRA.
        - Call with strict=False to allow missing keys (base params will be missing).

        This will update matching keys and leave other params untouched.
        """
        if ckpt_path is None:
            return None
        map_loc = map_location or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=map_loc)

        # accept different checkpoint layouts:
        if isinstance(ckpt, dict) and "model" in ckpt:
            sd = ckpt["model"]
        elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            # maybe it's directly a state_dict
            sd = ckpt
        else:
            raise ValueError(f"Unsupported checkpoint format for {ckpt_path}")

        # If this model is possibly wrapped in DDP externally, try to load into unwrapped module
        try:
            # if self is wrapper-like (has module attribute), prefer the real module
            target = self.module if hasattr(self, "module") else self
        except Exception:
            target = self

        res = target.load_state_dict(sd, strict=strict)
        missing = res.missing_keys if hasattr(res, "missing_keys") else None
        unexpected = res.unexpected_keys if hasattr(res, "unexpected_keys") else None
        print(f"[model] load_lora_and_heads: path={ckpt_path}, strict={strict}, missing={missing}, unexpected={unexpected}")
        return res
        