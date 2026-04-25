"""
Unified model loading for the 9 supported model architectures.

Handles architecture-specific quirks:
  - OLMo-7B      : requires `hf_olmo` package; custom layer/norm paths
  - Ministral-3-* : Mistral3ForConditionalGeneration (multimodal); we load the
                    full model then expose the text LM through a thin wrapper
  - Qwen3-14B    : requires transformers>=4.51
  - All others    : standard AutoModelForCausalLM

Every public loader returns objects that expose a *uniform* interface:
    model.model.layers    – ModuleList of transformer blocks
    model.model.norm      – final RMSNorm / LayerNorm
    model.lm_head         – nn.Linear mapping hidden → vocab
    model.config          – with .num_hidden_layers, .rms_norm_eps, etc.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    logging as hf_logging,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
MODEL_PATHS_JSON = os.path.join(REPO_ROOT, "model_paths.json")


def _read_model_paths() -> Dict[str, str]:
    with open(MODEL_PATHS_JSON) as f:
        return json.load(f)


def resolve_model_path(short_name: str) -> str:
    paths = _read_model_paths()
    if short_name in paths:
        return paths[short_name]
    raise ValueError(f"Cannot resolve model path for {short_name!r}")


def _detect_arch(path: str) -> str:
    """Return a tag describing how to load this checkpoint."""
    cfg_path = os.path.join(path, "config.json")
    if not os.path.isfile(cfg_path):
        return "standard"
    with open(cfg_path) as f:
        cfg = json.load(f)
    mt = cfg.get("model_type", "")
    if mt == "hf_olmo":
        return "olmo"
    if mt == "mistral3":
        return "mistral3"
    return "standard"


# ── OLMo wrapper ──────────────────────────────────────────────

class _OLMoCompat(nn.Module):
    """Thin wrapper that re-exports OLMo internals at the paths LM_hf expects."""

    def __init__(self, olmo_model):
        super().__init__()
        self._olmo = olmo_model

    # Provide model.model.layers  →  olmo.model.transformer.blocks
    @property
    def layers(self):
        return self._olmo.model.transformer.blocks

    # Provide model.model.norm  →  olmo.model.transformer.ln_f
    @property
    def norm(self):
        return self._olmo.model.transformer.ln_f

    @property
    def embed_tokens(self):
        return self._olmo.model.transformer.wte

    def __call__(self, *a, **kw):
        return self._olmo.model(*a, **kw)

    def forward(self, *a, **kw):
        return self._olmo.model(*a, **kw)


class _OLMoCausalLMCompat(nn.Module):
    """Makes an OLMo checkpoint look like a standard CausalLM."""

    def __init__(self, olmo_model):
        super().__init__()
        self.model = _OLMoCompat(olmo_model)
        inner = olmo_model.model
        if hasattr(inner.transformer, "ff_out"):
            self.lm_head = inner.transformer.ff_out
        else:
            self.lm_head = None
        cfg = olmo_model.config
        if not hasattr(cfg, "num_hidden_layers"):
            cfg.num_hidden_layers = getattr(cfg, "n_layers", 32)
        if not hasattr(cfg, "hidden_size"):
            cfg.hidden_size = getattr(cfg, "d_model", 4096)
        if not hasattr(cfg, "num_attention_heads"):
            cfg.num_attention_heads = getattr(cfg, "n_heads", 32)
        if not hasattr(cfg, "layer_norm_eps"):
            cfg.layer_norm_eps = 1e-5
        self.config = cfg
        self._olmo = olmo_model

    def forward(self, *a, **kw):
        return self._olmo(*a, **kw)

    def generate(self, *a, **kw):
        return self._olmo.generate(*a, **kw)

    @property
    def device(self):
        return next(self._olmo.parameters()).device

    def to(self, *a, **kw):
        self._olmo.to(*a, **kw)
        return self

    def eval(self):
        self._olmo.eval()
        return self

    @property
    def generation_config(self):
        return self._olmo.generation_config

    @generation_config.setter
    def generation_config(self, value):
        self._olmo.generation_config = value

    def resize_token_embeddings(self, n):
        return self._olmo.resize_token_embeddings(n)

    def get_output_embeddings(self):
        if self.lm_head is not None:
            return self.lm_head
        return self._olmo.model.transformer.wte


def _load_olmo(path: str, dtype=torch.float16, **kw) -> nn.Module:
    try:
        import hf_olmo  # noqa: F401 – registers the auto-map classes
    except ImportError:
        raise ImportError(
            "OLMo requires the ai2-olmo package. Run: pip install ai2-olmo"
        )
    raw = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True, **kw,
    )
    return _OLMoCausalLMCompat(raw)


# ── Ministral-3 wrapper ──────────────────────────────────────

class _Ministral3TextCompat(nn.Module):
    """Thin wrapper exposing Mistral3's text LM at standard CausalLM paths."""

    def __init__(self, inner_model):
        super().__init__()
        self._full = inner_model
        self.lm_head = inner_model.lm_head
        self.model = inner_model.model.language_model  # MistralModel with .layers, .norm
        self.config = inner_model.config.text_config

    def forward(self, *a, **kw):
        return self._full(*a, **kw)

    def generate(self, *a, **kw):
        return self._full.generate(*a, **kw)

    @property
    def device(self):
        return next(self._full.parameters()).device

    def to(self, *a, **kw):
        self._full.to(*a, **kw)
        return self

    def eval(self):
        self._full.eval()
        return self

    @property
    def generation_config(self):
        return self._full.generation_config

    @generation_config.setter
    def generation_config(self, value):
        self._full.generation_config = value

    def resize_token_embeddings(self, n):
        return self._full.resize_token_embeddings(n)

    def get_output_embeddings(self):
        return self.lm_head


def _dequantize_fp8(model: nn.Module, checkpoint_path: str, dtype) -> None:
    """Convert FP8-quantized weights to *dtype* using checkpoint scale factors.

    Ministral-Instruct stores linear weights in float8_e4m3fn with per-tensor
    ``weight_scale_inv`` tensors.  Dequantization: ``w = w_fp8 * scale_inv``.
    """
    import glob
    from safetensors.torch import load_file

    scale_inv: Dict[str, torch.Tensor] = {}
    for sf in sorted(glob.glob(os.path.join(checkpoint_path, "model-*.safetensors"))):
        for k, v in load_file(sf, device="cpu").items():
            if k.endswith(".weight_scale_inv"):
                scale_inv[k] = v

    # Build a reverse lookup: strip the _scale_inv suffix to get the weight
    # name as it appears in the checkpoint, then map from that to the scale.
    ckpt_scale = {}
    for sk, sv in scale_inv.items():
        wk = sk.removesuffix("_scale_inv")
        ckpt_scale[wk] = sv

    for name, param in model.named_parameters():
        if param.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            continue
        # Try the parameter name directly, then several remappings that
        # from_pretrained / Mistral3 may introduce.
        s = ckpt_scale.get(name)
        if s is None and name.startswith("model."):
            s = ckpt_scale.get(name[len("model."):])
        # Mistral3ForConditionalGeneration maps checkpoint
        # "language_model.model.layers..." to param "model.language_model.layers..."
        if s is None and "language_model" in name:
            alt = name.replace("model.language_model.", "language_model.model.")
            s = ckpt_scale.get(alt)
        if s is not None:
            param.data = param.data.to(dtype) * s.to(dtype)
        else:
            param.data = param.data.to(dtype)


def _load_ministral3(path: str, dtype=torch.float16, **kw) -> nn.Module:
    cfg_path = os.path.join(path, "config.json")
    with open(cfg_path) as f:
        raw_cfg = json.load(f)

    raw_cfg["text_config"]["model_type"] = "mistral"

    # Some Ministral checkpoints ship with an FP8 quantization_config whose
    # activation_scheme="static" is not supported by our transformers build.
    # Drop the quantization config so the weights are loaded in the requested
    # dtype (float16/bfloat16) instead.
    raw_cfg.pop("quantization_config", None)

    from transformers import Mistral3Config, Mistral3ForConditionalGeneration

    config = Mistral3Config.from_dict(raw_cfg)

    model = Mistral3ForConditionalGeneration.from_pretrained(
        path, config=config, torch_dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True, **kw,
    )
    _dequantize_fp8(model, path, dtype)
    return _Ministral3TextCompat(model)


# ── Public API ───────────────────────────────────────────────

def load_model(
    short_name: str,
    device: str = "cuda",
    dtype=torch.float16,
    cache_dir: Optional[str] = None,
) -> Tuple[AutoTokenizer, nn.Module]:
    """Load any of the 9 supported models and return (tokenizer, model).

    The returned model always exposes:
        model.model.layers, model.model.norm, model.lm_head, model.config
    """
    hf_logging.set_verbosity_error()
    warnings.filterwarnings("ignore")

    path = resolve_model_path(short_name)
    arch = _detect_arch(path)
    kw: Dict[str, Any] = {}
    if cache_dir:
        kw["cache_dir"] = cache_dir

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, **kw)

    if arch == "olmo":
        model = _load_olmo(path, dtype=dtype, **kw)
    elif arch == "mistral3":
        model = _load_ministral3(path, dtype=dtype, **kw)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True, **kw,
        )

    model.to(device).eval()
    return tok, model


def load_model_for_nnsight(
    short_name: str,
    device: str = "cuda",
    temperature: float = 0.7,
    seed: int = 0,
    dtype=torch.float16,
) -> Tuple[Any, Any, str]:
    """Load model for the nnsight-based simulation pipeline (LM_hf.py).

    Returns (base_model, tokenizer, arch_tag) ready for wrapping with LanguageModel().
    Handles pad-token, generation config, etc.
    """
    hf_logging.set_verbosity_error()
    warnings.filterwarnings("ignore")

    path = resolve_model_path(short_name)
    arch = _detect_arch(path)

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    if arch == "olmo":
        try:
            import hf_olmo  # noqa: F401
        except ImportError:
            raise ImportError("OLMo requires ai2-olmo. Run: pip install ai2-olmo")
        base_model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True,
        )
        # hf_olmo uses tuple-based KV cache, not DynamicCache. Transformers
        # >=4.40 creates a DynamicCache by default which OLMo can't consume.
        cls = type(base_model)
        if not hasattr(cls, "_orig_supports_dynamic_cache"):
            cls._orig_supports_dynamic_cache = cls._supports_default_dynamic_cache
            cls._supports_default_dynamic_cache = classmethod(lambda c: False)
    elif arch == "mistral3":
        cfg_path = os.path.join(path, "config.json")
        with open(cfg_path) as f:
            raw_cfg = json.load(f)
        raw_cfg["text_config"]["model_type"] = "mistral"
        raw_cfg.pop("quantization_config", None)
        from transformers import Mistral3Config, Mistral3ForConditionalGeneration
        config = Mistral3Config.from_dict(raw_cfg)
        base_model = Mistral3ForConditionalGeneration.from_pretrained(
            path, config=config, torch_dtype=dtype, low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        _dequantize_fp8(base_model, path, dtype)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True,
        )

    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
        base_model.resize_token_embeddings(len(tok))

    try:
        base_model.generation_config = GenerationConfig.from_pretrained(path)
    except Exception:
        pass

    if temperature == 0:
        base_model.generation_config.do_sample = False
        base_model.generation_config.temperature = None
        base_model.generation_config.top_p = None
        base_model.generation_config.top_k = None
    else:
        base_model.generation_config.do_sample = True
        base_model.generation_config.temperature = temperature

    base_model.generation_config.seed = seed
    eos = base_model.generation_config.eos_token_id
    if isinstance(eos, list):
        base_model.generation_config.pad_token_id = eos[0]
    else:
        base_model.generation_config.pad_token_id = eos

    base_model.to(device).eval()
    return base_model, tok, arch
