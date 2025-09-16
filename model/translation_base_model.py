import os
import json
import yaml
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Union, Tuple, Dict, Any
from config.model_config import ModelConfig
from model.model_modules import LinearEmbedding, AddPromptEmbedding, Encoder, EncoderLayer

__author__ = "Chunfu Xiao"
__version__="1.3.0"
__email__ = "chunfushawn@gmail.com"

        
class TranslationBaseModel(nn.Module):
    """
    TranslationBaseModel: encoder-based model with pluggable heads and robust input handling.

    Improvements over the original:
      - Accepts single-sample inputs without batch dimension (e.g. shape [seq_len, d_seq]).
      - Accepts cell_type as string/int/torch.Tensor/numpy/list.
      - Provides `predict(...)` helper for inference (eval mode + no_grad + device handling).
      - Keeps add_head/remove_head/list_heads/save_head/load_head APIs.
    """
    def __init__(
        self,
        d_seq: int,
        d_count: int,
        d_model: int,
        pmt_len: int,
        all_cell_types: List[str],
        n_heads: int = 8,
        number_of_layers: int = 6,
        d_ff: int = 2048,
        p_drop: float = 0.1,
        model_name: str = "base_model",
    ):
        super().__init__()
        # store raw config portion on the instance (handy later)
        self._constructor_args = dict(
            d_seq=d_seq,
            d_count=d_count,
            d_model=d_model,
            pmt_len=pmt_len,
            all_cell_types=all_cell_types,
            n_heads=n_heads,
            number_of_layers=number_of_layers,
            d_ff=d_ff,
            p_drop=p_drop,
            model_name=model_name
        )
        
        self.model_name = model_name
        self.d_seq = d_seq
        self.d_count = d_count
        self.d_model = d_model
        self.n_heads = n_heads

        # embeds source data into high-dimensional potent embedding vectors
        self.src_emb = LinearEmbedding(d_seq, d_count, d_model)
        self.pmt_len = pmt_len

        # cell type handling
        self.num_cells = len(all_cell_types)
        # mapping: token string -> index [0..num_cells-1]
        self.cell_type_mapping = dict(
            zip(
                all_cell_types, 
                range(self.num_cells)
                )
            )
        # add prompt embedding (expects integer index in [0..num_cells] where num_cells can mean "unknown/mask")
        self.add_prompt_emb = AddPromptEmbedding(self.pmt_len, self.num_cells, d_model)

        # encoder stack
        encoder_layer = EncoderLayer(d_model, d_ff, n_heads, p_drop)
        self.encoder = Encoder(encoder_layer, number_of_layers)

        # pluggable heads
        self.heads = nn.ModuleDict() # key -> head module
        
        # initialize parameters (Xavier for weight tensors by default)
        self._init_parameters()

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
                    if m.weight is not None:
                        nn.init.ones_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

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
        required = {"d_seq", "d_count", "d_model", "pmt_len", "all_cell_types"}
        missing = required - cfg_dict.keys()
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")

        # coerce types & use defaults where not given
        return ModelConfig(
            d_seq=int(cfg_dict["d_seq"]),
            d_count=int(cfg_dict["d_count"]),
            d_model=int(cfg_dict["d_model"]),
            pmt_len=int(cfg_dict["pmt_len"]),
            all_cell_types=list(cfg_dict["all_cell_types"]),
            n_heads=int(cfg_dict.get("n_heads", 8)),
            number_of_layers=int(cfg_dict.get("number_of_layers", 6)),
            d_ff=int(cfg_dict.get("d_ff", 2048)),
            p_drop=float(cfg_dict.get("p_drop", 0.1)),
            model_name=cfg_dict.get("model_name"),
            seed=cfg_dict.get("seed"),
        )

    @classmethod
    def from_config(cls, config: Union[Dict[str, Any], ModelConfig, str]) -> "TranslationBaseModel":
        """
        Construct a TranslationBaseModel from a config dict, ModelConfig instance, or path to JSON/YAML file.

        Example:
            model = TranslationBaseModel.from_config("configs/my_model.yaml")
            or
            cfg = {"d_seq":4, "d_count":10, "d_model":512, "pmt_len":3, "all_cell_types": ["A","B",...]}
            model = TranslationBaseModel.from_config(cfg)
        """
        cfg = cls._normalize_config(config)

        # optional: set random seed for reproducibility if provided
        if cfg.seed is not None:
            torch.manual_seed(int(cfg.seed))

        print(f"==> Create model using config from {config}: {cfg}")

        return cls(
            d_seq=cfg.d_seq,
            d_count=cfg.d_count,
            d_model=cfg.d_model,
            pmt_len=cfg.pmt_len,
            all_cell_types=cfg.all_cell_types,
            n_heads=cfg.n_heads,
            number_of_layers=cfg.number_of_layers,
            d_ff=cfg.d_ff,
            p_drop=cfg.p_drop,
            model_name=cfg.model_name
        )

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
          - count_batch shape (bs, seq_len, d_count) or (seq_len, d_count).
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
    
    def _normalize_cell_type(self, cell_type: Any, batch_size: int) -> torch.LongTensor:
        """
        Normalize cell_type inputs into a long tensor of shape (batch_size,).
        Accepts:
          - single str -> convert to index or num_cells for unknown
          - single int -> validate or map to num_cells if out-of-range
          - list / np.ndarray / torch.Tensor -> length must be batch_size or 1 (will broadcast)
        Unknown entries map to self.num_cells (reserved index).
        """
        # helper to convert single element to index
        def _single_to_index(val) -> int:
            if isinstance(val, str):
                return self.cell_type_mapping.get(val, self.num_cells)
            try:
                ival = int(val)
            except Exception:
                # fallback: unknown -> mask index
                return self.num_cells
            if 0 <= ival < self.num_cells:
                return ival
            return self.num_cells

        # If torch tensor
        if isinstance(cell_type, torch.Tensor):
            # if scalar tensor
            if cell_type.numel() == 1:
                idx = _single_to_index(cell_type.item())
                arr = torch.full((batch_size,), idx, dtype=torch.long)
                return arr
            # 1-D tensor: ensure length matches batch_size or broadcast if 1
            if cell_type.dim() == 1:
                if cell_type.numel() == batch_size:
                    return cell_type.to(dtype=torch.long)
                if cell_type.numel() == 1:
                    idx = _single_to_index(cell_type.item())
                    return torch.full((batch_size,), idx, dtype=torch.long)
                raise ValueError(f"cell_type tensor length {cell_type.numel()} != batch_size {batch_size}")
            raise ValueError("Unsupported cell_type tensor shape.")

        # numpy array
        if isinstance(cell_type, np.ndarray):
            arr = cell_type.flatten()
            if arr.size == 1:
                idx = _single_to_index(arr.item())
                return torch.full((batch_size,), idx, dtype=torch.long)
            if arr.size == batch_size:
                # convert elements individually (allow strings)
                mapped = [_single_to_index(x) for x in arr.tolist()]
                return torch.tensor(mapped, dtype=torch.long)
            raise ValueError(f"cell_type array length {arr.size} != batch_size {batch_size}")

        # python list/tuple
        if isinstance(cell_type, (list, tuple)):
            if len(cell_type) == 1:
                idx = _single_to_index(cell_type[0])
                return torch.full((batch_size,), idx, dtype=torch.long)
            if len(cell_type) == batch_size:
                mapped = [_single_to_index(x) for x in cell_type]
                return torch.tensor(mapped, dtype=torch.long)
            raise ValueError(f"cell_type list length {len(cell_type)} != batch_size {batch_size}")

        # scalar int/str
        idx = _single_to_index(cell_type)
        return torch.full((batch_size,), idx, dtype=torch.long)
    
    # -------------------------
    # Forward / predict
    # -------------------------
    def forward(
        self,
        seq_batch: torch.Tensor,
        count_batch: torch.Tensor,
        cell_type_idx: torch.LongTensor,
        src_mask: Optional[torch.Tensor] = None,
        head_names: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Strict forward that requires:
          - seq_batch: torch.Tensor, shape (bs, seq_len, d_seq), dim == 3
          - count_batch: torch.Tensor, shape (bs, seq_len, d_count), dim == 3
          - cell_type_idx: torch.LongTensor, shape (bs,), dtype long
          - src_mask: Optional[torch.BoolTensor], shape (bs, seq_len) or None

        This function will:
          - validate dtypes/shapes,
          - call embedding/prompt/encoder,
          - return encoder outputs or head outputs.

        Note: predict(...) is the convenience wrapper that converts numpy/lists/scalars
        and single-sample shapes into tensors and then calls forward(...).
        """

        # --- basic type checks ---
        if not isinstance(seq_batch, torch.Tensor):
            raise TypeError("forward() expects seq_batch as torch.Tensor (dim==3). Use predict() for flexible inputs.")
        if not isinstance(count_batch, torch.Tensor):
            raise TypeError("forward() expects count_batch as torch.Tensor (dim==3). Use predict() for flexible inputs.")
        if not isinstance(cell_type_idx, torch.Tensor):
            raise TypeError("forward() expects cell_type_idx as torch.LongTensor. Use predict() for flexible inputs.")
        if seq_batch.dim() != 3:
            raise ValueError(f"seq_batch must have dim==3 (bs, seq_len, d_seq). Got shape {tuple(seq_batch.shape)}")
        if count_batch.dim() != 3:
            raise ValueError(f"count_batch must have dim==3 (bs, seq_len, d_count). Got shape {tuple(count_batch.shape)}")
        
        bs = seq_batch.shape[0]
        seq_len = seq_batch.shape[1]

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

        # cell_type_idx validation
        if cell_type_idx.dim() != 1 or cell_type_idx.shape[0] != bs:
            raise ValueError("cell_type_idx must be a 1D long tensor of shape (bs,)")
        
        # compute embeddings (expects shapes (bs, seq_len, d_seq) & (bs, seq_len, d_count))
        src_embs = self.src_emb(seq_batch, count_batch)  # -> (bs, seq_len, d_model)

        # add cell prompt embedding; AddPromptEmbedding expects integer index (or tensor) and mask
        src_embs, src_mask = self.add_prompt_emb(src_embs, src_mask, cell_type_idx)

        # encode
        src_reps = self.encoder(src_embs, src_mask)

        # if user requested no heads, return raw representations (optionally squeeze later)
        if not head_names:
            return src_reps

        # run requested heads
        outputs = {}
        for name in head_names:
            if name not in self.heads:
                raise KeyError(f"Head {name} not found. Available: {list(self.heads.keys())}")
            head = self.heads[name]
            outputs[name] = head(src_reps, **kwargs) if callable(head.forward) else head(src_reps)

        return outputs
    
    # -------------------------
    # predict: flexible, preprocess & call forward
    # -------------------------
    def predict(
        self,
        seq_batch: Union[torch.Tensor, np.ndarray, list, tuple],
        count_batch: Union[torch.Tensor, np.ndarray, list, tuple],
        cell_type: Any = None,
        src_mask: Optional[Union[torch.Tensor, np.ndarray, list]] = None,
        head_names: Optional[List[str]] = None,
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
        # Decide device from model parameters (first param) or default device
        model_device = next(self.parameters()).device

        # 2). normalize inputs and preserve whether user passed a single sample
        seq_batch, count_batch, src_mask, was_squeezed = self._ensure_batch_dim_input(seq_batch, count_batch, src_mask)
        bs = seq_batch.shape[0]
        # normalize cell_type to tensor of shape (bs,)
        cell_type_idx = self._normalize_cell_type(cell_type, bs)

        # 3) move tensor inputs to model device 
        if move_inputs_to_device:
            # move only if input is a torch.Tensor (avoid converting np/list here)
            if isinstance(seq_batch, torch.Tensor):
                seq_batch = seq_batch.to(model_device)
            if isinstance(count_batch, torch.Tensor):
                count_batch = count_batch.to(model_device)
            if isinstance(src_mask, torch.Tensor):
                src_mask = src_mask.to(model_device)
            if isinstance(cell_type_idx, torch.Tensor):
                cell_type_idx = cell_type_idx.to(model_device)

        # 4) run forward under no_grad
        with torch.no_grad():
            outputs = self.forward(seq_batch, count_batch, cell_type_idx, src_mask, head_names=head_names)

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
    
    def add_head(self, name: str, head_module: nn.Module, overwrite: bool = False, move_to_model_device: bool = True) -> None:
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
                model_device = next(self.parameters()).device
            except StopIteration:
                # model has no parameters? fallback to CPU
                model_device = torch.device("cpu")
            # move the head module (this moves parameters and buffers)
            head_module.to(model_device)

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
        