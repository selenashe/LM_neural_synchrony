import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from transformers import AutoModelForCausalLM, AutoTokenizer
from nnsight import LanguageModel
import numpy as np
import re
import torch
from transformers.generation.utils import GenerationConfig
from core.utils import *
import random

from core.prompt_formatting import format_sotopia_prompt
from core.model_loader import load_model_for_nnsight


class BaseLM:
    def __init__(self, model_path, model_name, device="cuda", temperature=0., seed=0, **kwargs):
        self.device = device
        self.model_name = model_name
        base_model, tokenizer, arch = load_model_for_nnsight(
            model_name, device=device, temperature=temperature, seed=seed,
        )
        self.temperature = temperature
        self.seed = seed
        self.model = LanguageModel(base_model, tokenizer=tokenizer)
        self._arch = arch
        self._last_thinking_trace = ""
    
    def get_inst(self, base_prompt):
        return format_sotopia_prompt(self.model_name, base_prompt, self.model.tokenizer)

    def get_response_format(self, resp):
        if self.model_name == "Mistral-7B-Instruct-v0.1":
            return resp.split('[/INST]')[-1]
        elif self.model_name == "Mistral-7B-Instruct-v0.2":
            return resp.split('[/INST]')[-1]
        elif self.model_name == "Mistral-7B-Instruct-v0.3":
            return resp.split('[/INST]')[-1]
        elif self.model_name == "DeepSeek-R1-Distill-Llama-8B":
            return resp.split('<｜Assistant｜>')[-1]
        elif "Llama" in self.model_name and "Chat" in self.model_name:
            return resp.split('[/INST]')[-1]
        elif "Llama" in self.model_name:
            return resp.split('<|end_header_id|>')[-1].replace('<|eot_id|>', '')
        elif self.model_name == "gpt2":
            return resp
        elif self.model_name.startswith("mistralai_Ministral"):
            return resp.split('[/INST]')[-1]
        elif self.model_name in (
            "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct",
            "Qwen_Qwen3-14B", "Qwen3-8B",
            "allenai_OLMo-3-7B-Instruct",
        ):
            return resp.split('<|im_start|>assistant\n')[-1]
        return resp

    def get_result_from_output(self, value):
        result = self.model.tokenizer.batch_decode(value)[0]
        result = self.get_response_format(result)
        result = result.replace('</s>', '')
        result = result.replace('<|im_end|>', '')
        result = result.replace('<|eot_id|>', '')
        result = result.replace('<｜end▁of▁sentence｜>', '')
        result = result.replace('<｜begin▁of▁sentence｜>', '')
        result = result.replace('<|endoftext|>', '')
        result = result.replace('<eos>', '')
        think_match = re.search(r'<think>(.*?)</think>', result, flags=re.DOTALL)
        if think_match:
            self._last_thinking_trace = think_match.group(1).strip()
            result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
        elif '<think>' in result:
            # Model ran out of tokens mid-thinking (no closing </think>)
            self._last_thinking_trace = result.split('<think>', 1)[1].strip()
            result = result.split('<think>', 1)[0]
        else:
            self._last_thinking_trace = ""
        result = result.strip()
        return result


class LM_nnsight(BaseLM):
    def __init__(self, model_path, model_name, device="cuda", temperature=0., seed=0., affected_level=0, affected_type="None"):
        super().__init__(model_path, model_name, device, temperature, seed)

    def generate_response(self, prompt, max_new_tokens=2000):
        prompt = self.get_inst(prompt)
        with self.model.generate(max_new_tokens=max_new_tokens) as generator:  
            with generator.invoke(prompt) as invoker:
                output = self.model.generator.output.save()
        # print(output)
        return self.get_result_from_output(output.value)

    def _nnsight_layers(self):
        """Return the nnsight proxy for the list of transformer blocks."""
        if self._arch == "olmo":
            return self.model.model.transformer.blocks
        if self._arch == "mistral3":
            return self.model.model.language_model.layers
        return self.model.model.layers

    def _nnsight_config(self):
        """Return the config object visible through nnsight."""
        if self._arch == "mistral3":
            return self.model.model.language_model.config
        if self._arch == "olmo":
            return self.model.config
        return self.model.model.config

    def _num_attention_heads(self):
        cfg = self._nnsight_config()
        return getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_heads", 32)

    def _layer_hidden(self, layer_proxy):
        """Extract the hidden-state tensor from a decoder-layer output proxy.

        Older / OLMo-style layers return a tuple (hidden, cache); newer
        transformers (>=4.50) return a bare tensor.  We handle both.
        """
        if self._arch == "olmo":
            return layer_proxy.output[0]
        return layer_proxy.output

    def generate_response_and_get_state(self, prompt, max_new_tokens, layer, token=-1, save_states_fake_agents=None, verbose=False, save_attn=False):
        reps = []
        attn_reps = []
        n_heads = self._num_attention_heads()
        
        prompt = self.get_inst(prompt)
        prompt_tokens = self.model.tokenizer.encode(prompt)
        
        layers = self._nnsight_layers()
        loi = []
        if layer == "All":
            for l in layers:
                loi.append(l)
        else:
            loi.append(layers[layer])

        with self.model.generate(max_new_tokens=max_new_tokens) as tracer:
            with tracer.invoke(prompt) as invoker:
                for i, l in enumerate(loi):
                    reps.append([self._layer_hidden(l)[:, -1:, :].save()])
                    if save_attn:
                        attn_reps.append([l.self_attn.output[0][:, -1:, :].save()])
                    if token == -1:
                        continue
                    for _ in range(max_new_tokens):
                        l.next()
                        l.self_attn.next()
                        reps[i].append(self._layer_hidden(l).save())
                        if save_attn:
                            attn_reps[i].append(l.self_attn.output[0].save())

                original = self.model.generator.output.save()
        
        reps_numpy = []
        att_reps_numpy = []
        output_tokens_num = len(original.value[0].tolist()) - len(prompt_tokens)
        if save_attn:
            for rep, att_rep in zip(reps, attn_reps):
                hidden_states = torch.cat(rep[:output_tokens_num], dim=1).cpu().numpy()
                reps_numpy.append(hidden_states[0])
                
                atts = torch.cat(att_rep[:output_tokens_num], dim=1).cpu().numpy()
                reshaped_atts = atts.reshape(atts.shape[0], output_tokens_num, n_heads, -1)
                att_reps_numpy.append(reshaped_atts[0])
        else:
            for rep in reps:
                hidden_states = torch.cat(rep[:output_tokens_num], dim=1).cpu().numpy()
                reps_numpy.append(hidden_states[0])
            
        reps_numpy = np.array(reps_numpy)
        att_reps_numpy = np.array(att_reps_numpy)

        ret = self.get_result_from_output(original.value)

        return ret, ret, reps_numpy, att_reps_numpy

    def __call__(self, prompt, max_new_tokens=2000):
        ans = self.generate_response(prompt, max_new_tokens)
        return ans