"""
Shared prompt formatting for Sotopia simulation (LM_hf) and logit-lens analysis.

Uses tokenizer.chat_template when available; otherwise fallbacks match LM_hf heuristics.
"""

from __future__ import annotations

from typing import Any


def format_sotopia_prompt(model_short_name: str, raw_prompt: str, tokenizer: Any) -> str:
    """Format a raw task prompt consistently with Sotopia rollout."""
    _skip_auto_template = {"allenai_OLMo-3-7B-Instruct"}
    ct = getattr(tokenizer, "chat_template", None)
    if ct and model_short_name not in _skip_auto_template:
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": raw_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    if model_short_name in (
        "Mistral-7B-Instruct-v0.1",
        "Mistral-7B-Instruct-v0.2",
        "Mistral-7B-Instruct-v0.3",
    ):
        return f"[INST]{raw_prompt}[/INST]"

    if model_short_name.startswith("mistralai_Ministral"):
        return f"[INST]{raw_prompt}[/INST]"

    if "Llama" in model_short_name and "Instruct" in model_short_name:
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{raw_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    if "Llama" in model_short_name and "Chat" in model_short_name:
        return f"[INST]{raw_prompt}[/INST]"

    if model_short_name == "gpt2":
        return raw_prompt

    if model_short_name in (
        "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct",
        "Qwen_Qwen3-14B", "Qwen3-8B",
        "allenai_OLMo-3-7B-Instruct",
    ):
        return (
            f"<|im_start|>user\n{raw_prompt}<|im_end|>\n<|im_start|>assistant\n"
        )

    if model_short_name == "DeepSeek-R1-Distill-Llama-8B":
        return (
            f"<｜begin▁of▁sentence｜><｜User｜>{raw_prompt}<｜Assistant｜>"
        )

    return f"[INST]{raw_prompt}[/INST]"
