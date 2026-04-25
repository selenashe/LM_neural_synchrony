"""
Sotopia dialogue environment backed by Gemini on Vertex AI (LiteLLM).

Mirrors turn-taking, private masked intros, and CSV-friendly dialog lines from
`sot_env.SotopiaEnv`, but replaces local HF generation with Vertex completions.

Vertex settings match `goal_alignment_labels.classify_combo`:
  model, vertex_project, vertex_location, timeout, num_retries (via LiteLLM).

Does not save hidden-state .npy tensors (API models have no internal states).
"""

from __future__ import annotations

import json
import os
import pickle
import random
import re
import time
from typing import Optional

import numpy as np
from openai import OpenAI

try:
    from langchain_core.output_parsers import PydanticOutputParser
except ImportError:  # pragma: no cover
    from langchain.output_parsers import PydanticOutputParser  # type: ignore

from litellm import completion

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from experiment_config import ENV_AGENT_COMBOS_PATH, ENVS_PATH
from sotopia_utils.utils import (
    AgentBackground,
    EnvResponse,
    format_bad_output,
    get_bio,
)
from core.utils import create_folder_if_not_there

current_file_path = os.path.abspath(__file__)
directory = os.path.dirname(current_file_path)

# Same defaults as goal_alignment_labels.py / analyze_prompt_goal_tracking.py
DEFAULT_VERTEX_MODEL = "vertex_ai/gemini-3.1-pro-preview"
DEFAULT_VERTEX_PROJECT = "hs-soil-gemini"
DEFAULT_VERTEX_LOCATION = "global"
DEFAULT_COMPLETION_TIMEOUT = 120
DEFAULT_NUM_RETRIES = 3

_ACTOR_SYSTEM = """You are simulating one speaking turn in a structured multi-agent social scenario (Sotopia).

The user message contains:
- The agent's private context (background, relationship, scenario) with opponent secrets/goals masked.
- The dialogue so far; the last line ends with that agent having just been prompted to speak (e.g. "Alice said:" with nothing after the colon).

Rules:
- Output ONLY the words that agent speaks for this turn (no "Turn N:" prefix, no name prefix like "Alice said:").
- Stay in character; follow the private goal; do not reveal information the prompt marks as unknown to you.
- Be concise. If your goal is achieved and the scenario allows leaving, you may say LEAVE as instructed in the prompt.
- Reply on a single line when possible."""


def _vertex_generate(
    model: str,
    user_prompt: str,
    *,
    temperature: float,
    max_tokens: Optional[int] = None,
    vertex_project: str,
    vertex_location: str,
    max_attempts: int = 3,
) -> str:
    last_err: Optional[BaseException] = None
    for attempt in range(max_attempts):
        try:
            kwargs: dict = dict(
                model=model,
                messages=[
                    {"role": "system", "content": _ACTOR_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                vertex_project=vertex_project,
                vertex_location=vertex_location,
                temperature=temperature,
                timeout=DEFAULT_COMPLETION_TIMEOUT,
                num_retries=DEFAULT_NUM_RETRIES,
            )
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            resp = completion(**kwargs)
            text = (resp.choices[0].message.content or "").strip()
            text = re.sub(r"^```\w*\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            return text
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"  Vertex completion error (attempt {attempt + 1}/{max_attempts}): {e!r}; sleeping {wait}s")
            time.sleep(wait)
    assert last_err is not None
    raise last_err


def _postprocess_spoken_line(raw: str) -> str:
    text = raw.strip().split("\n")[0]
    if ": " in text:
        text = "".join(text.split(": ")[1:])
    return text


class GeminiSotopiaEnv:
    """Same dialogue protocol as `SotopiaEnv`, with Gemini/Vertex generation."""

    def __init__(
        self,
        model_1_name: str,
        model_2_name: str,
        vertex_model_1: str = DEFAULT_VERTEX_MODEL,
        vertex_model_2: Optional[str] = None,
        vertex_project: str = DEFAULT_VERTEX_PROJECT,
        vertex_location: str = DEFAULT_VERTEX_LOCATION,
        token_position: int = -1,
        env_model: str = "gpt-4",
        max_turns: int = 20,
        temperature: float = 0.7,
        test_baseline: bool = False,
        test_gpt: bool = False,
        test_mode: str = "None",
        max_new_tokens: Optional[int] = 50,
        save_states: bool = False,
        save_states_only_dialog: bool = False,
        save_states_dialog_imperson: bool = False,
        save_states_without_goal: bool = False,
        probing_goal: bool = False,
        probing_self_goal: bool = False,
        saving_dir: str | None = None,
        saving_dir_base: str | None = None,
        env_agent_combos_path: str = ENV_AGENT_COMBOS_PATH,
        mask_history: bool = False,
        verbose: bool = False,
        seed: int = 0,
    ):
        if saving_dir is not None:
            create_folder_if_not_there(saving_dir)

        self.vertex_model_1 = vertex_model_1
        self.vertex_model_2 = vertex_model_2 or vertex_model_1
        self.vertex_project = vertex_project
        self.vertex_location = vertex_location

        if ("None" in model_1_name or "Replace" not in model_1_name) and model_1_name == model_2_name:
            self.vertex_model_2 = self.vertex_model_1
            print("Using the same Vertex model id for both agents")

        if "Self" in model_1_name and "Self" in model_2_name:
            self.mask_other_utterances = True
        else:
            self.mask_other_utterances = False

        self.actor_role = "agent1"
        self.env_model = env_model
        self.cur_turn = 0
        self.max_turns = max_turns
        self.temperature = temperature
        self.agent1_intro = ""
        self.agent2_intro = ""
        self.complete_intro = ""
        self.cur_dialog: list[str] = []
        self.intended_dialog: list[str] = []
        self.cur_rewards: list = []
        self.actions: list = []
        self.cur_states: list = []
        self.next_states: list = []
        self.terminations: list = []
        self.p1_scores = None
        self.p2_scores = None
        self.model_1_name = model_1_name
        self.model_2_name = model_2_name
        self.agent1_name = ""
        self.agent2_name = ""
        self.agent1_goal = ""
        self.agent2_goal = ""
        self.additional_instruction = (
            'If your goal as the agent has been achieved, you should leave the conversation by saying "LEAVE". '
            "If your goal has not been achieved, you should say something to interact with the other. "
            "Keep your words concise.\n"
        )
        self.test_mode = test_mode
        self.max_new_tokens = max_new_tokens
        self.competition_episodes: list = []
        self.cooperation_episodes: list = []
        self.episode = 0
        self.save_states = save_states
        if self.save_states:
            print("Note: GeminiSotopiaEnv ignores save_states=True (no hidden states to save).")
        self.save_states_only_dialog = save_states_only_dialog
        self.save_states_dialog_imperson = save_states_dialog_imperson
        self.save_states_without_goal = save_states_without_goal
        self.probing_goal = probing_goal
        self.probing_self_goal = probing_self_goal
        self.env_agent_combos_path = env_agent_combos_path
        self.load_scenarios()
        self.prompts: list[str] = []
        self.prompts_only_dialogs: list = []
        self.prompts_dialog_imperson: list = []
        self.prompts_without_goal: list = []
        self.prompts_probing_goal: list = []
        self.prompts_probing_self_goal: list = []
        self.token_position = token_position
        self.mask_history = mask_history
        self.saving_dir_base = saving_dir_base
        self.seed = seed
        self.source_name = None

    def load_scenarios(self):
        with open(self.env_agent_combos_path, "r", encoding="utf-8") as f:
            self.env_agent_combos = json.load(f)

        with open(ENVS_PATH, "r") as f:
            self.envs = json.load(f)

        cooperation_episodes = []
        competition_episodes = []
        for k, v in self.envs.items():
            if v["source"] == "mutual_friends":
                cooperation_episodes.append(k)
            if v["source"] == "craigslist_bargains":
                competition_episodes.append(k)

        cooperation_episodes_number = []
        competition_episodes_number = []
        for i, comb in enumerate(self.env_agent_combos):
            if comb["env_id"] in cooperation_episodes:
                cooperation_episodes_number.append(i)
            elif comb["env_id"] in competition_episodes:
                competition_episodes_number.append(i)

        self.cooperation_episodes = cooperation_episodes_number
        self.competition_episodes = competition_episodes_number

        with open(os.path.join(REPO_ROOT, "sotopia_utils/sotopia_data/agents.json"), "r") as f:
            self.agent_profiles = json.load(f)

    def get_current_prompt(
        self,
        only_dialog_imperson=False,
        without_goal=False,
        probing_goal=False,
        probing_self_goal=False,
    ):
        prompt = "\n".join(self.cur_dialog)
        prompt = self.agent1_intro + prompt if self.cur_turn % 2 == 0 else self.agent2_intro + prompt
        return prompt

    def _clean_tags(self, text):
        return re.sub(r"<.*?>", "", text)

    def _mask_intro(self, intro, role: int):
        opponent_name = self.agent2_name if role == 0 else self.agent1_name
        if f"<p viewer='agent_{1 - role}'>" in intro:
            intro = re.sub(rf"<p viewer='agent_{1 - role}'>.*?<\/p>", "", intro)

        intro = re.sub(r' <p viewer="environment">.*?<\/p>', "", intro)

        intro = re.sub(rf"{opponent_name}'s goal: .*?(\n|$)", f"{opponent_name}'s goal: Unknown\n", intro)
        intro = re.sub(
            rf"{opponent_name}'s goal: .*?(\n<extra_info>.*?\n</extra_info>|\n|$)",
            f"{opponent_name}'s goal: Unknown\n",
            intro,
            flags=re.DOTALL,
        )
        return intro

    def _judge_terminate(self) -> bool:
        if self.cur_turn >= self.max_turns:
            return True
        last = self.intended_dialog[-1].lower()
        head, tail = last[:10], last[-10:]
        if any(x in head or x in tail for x in ["left", "leave"]):
            return True
        return False

    def save_conversation_history(self, seed):
        self.cur_rewards = np.array(self.cur_rewards)
        self.cur_states = np.array(self.cur_states)
        self.actions = np.array(self.actions)
        self.next_states = np.array(self.next_states)
        self.terminations = np.array(self.terminations)

        tbd = dict(
            complete_intro=self.complete_intro,
            dialog=self.cur_dialog,
            rewards=self.cur_rewards,
            observations=self.cur_states,
            actions=self.actions,
            next_observations=self.next_states,
            terminals=self.terminations,
            agent1_scores=self.p1_scores,
            agent2_scores=self.p2_scores,
        )
        with open(
            os.path.join(self.saving_dir, f"env_id={self.env_id}_role={self.actor_role}_seed={seed}.pkl"),
            "wb",
        ) as f:
            pickle.dump(tbd, f)

    def reset(self, env_id=None, actor_role="agent1"):
        self.cur_dialog = []
        self.intended_dialog = []
        self.cur_rewards = []
        self.cur_states = []
        self.actions = []
        self.next_states = []
        self.terminations = []
        self.prompts = []

        self.actor_role = actor_role

        if env_id is None:
            env_id = random.randint(0, len(self.env_agent_combos) - 1)
        self.episode = env_id
        env_agent_combo_storage = self.env_agent_combos[env_id]

        self.env_id = env_id
        env_id_key = env_agent_combo_storage["env_id"]

        env_profile = self.envs[env_id_key]

        source_name = env_profile["source"]
        self.source_name = source_name
        self.saving_dir = f"{self.saving_dir_base}/{source_name}"

        agent_ids = env_agent_combo_storage["agent_ids"]
        agent_profiles = [self.agent_profiles[i] for i in agent_ids]

        self.agent1_name = agent_profiles[0]["first_name"] + " " + agent_profiles[0]["last_name"]
        self.agent2_name = agent_profiles[1]["first_name"] + " " + agent_profiles[1]["last_name"]

        self.agent1_goal = f"{env_profile['agent_goals'][0]}"
        self.agent2_goal = f"{env_profile['agent_goals'][1]}"

        if source_name == "false_belief":
            p1_bg = ""
            p2_bg = ""
        else:
            p1_bg = get_bio(env_profile["relationship"], agent_profiles[0], agent_id=0)
            p2_bg = get_bio(env_profile["relationship"], agent_profiles[1], agent_id=1)

        background = AgentBackground(
            scenario=env_profile["scenario"],
            p1_background=p1_bg,
            p2_background=p2_bg,
            p1_goal=f"{env_profile['agent_goals'][0]}",
            p2_goal=f"{env_profile['agent_goals'][1]}",
            p1_name=self.agent1_name,
            p2_name=self.agent2_name,
        )
        agent1 = self.agent1_name
        agent2 = self.agent2_name

        if source_name == "false_belief":
            a1_instr = "Respond only with dialogue. Do not narrate actions or describe thoughts. You should continue the conversation until the other person says LEAVE, do not say LEAVE first. Keep your responses concise.\n"
            a2_instr = "Respond only with dialogue. Do not narrate actions or describe thoughts. Once you have decided which location to check, you should state your final choice and leave the conversation by saying LEAVE. Keep your responses concise.\n"
            self.agent1_intro = background.to_task_prompt(agent1, role=0) + "\n" + a1_instr
            self.agent2_intro = background.to_task_prompt(agent2, role=1) + "\n" + a2_instr
        else:
            self.agent1_intro = self._mask_intro(background.to_natural_language(agent1), 0) + self.additional_instruction
            self.agent2_intro = self._mask_intro(background.to_natural_language(agent2), 1) + self.additional_instruction

        self.complete_intro = self._clean_tags(background.to_complete_intro())
        self.agent1_intro = self._clean_tags(self.agent1_intro)
        self.agent2_intro = self._clean_tags(self.agent2_intro)

        self.agent1_goal = self._clean_tags(self.agent1_goal)
        self.agent2_goal = self._clean_tags(self.agent2_goal)

        self.cur_turn = 0
        self.cur_dialog.append(f"Turn {self.cur_turn}: {self.agent1_name} said:")
        self.intended_dialog.append(f"Turn {self.cur_turn}: {self.agent1_name} said:")

        if self.actor_role == "agent2":
            self._act(actor=2, save_states=False, turn=self.cur_turn)
            self.cur_dialog.append(f"Turn {self.cur_turn}: {self.agent2_name} said:")
            self.intended_dialog.append(f"Turn {self.cur_turn}: {self.agent2_name} said:")

        return None

    def get_final_reward(self):
        template = """{history}{schema}"""
        history = self.complete_intro + "\n".join(self.cur_dialog)
        with open(os.path.join(REPO_ROOT, "sotopia_utils/output_schema.txt"), "r") as f:
            schema = f.read()

        prompt = template.format(history=history, schema=schema)

        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model=self.env_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        output_text = response.choices[0].message.content

        output_parser = PydanticOutputParser(pydantic_object=EnvResponse)
        try:
            parsed_result = output_parser.parse(output_text)
        except Exception:
            reformat_parsed_result = format_bad_output(
                output_text, format_instructions=output_parser.get_format_instructions()
            )
            parsed_result = output_parser.parse(reformat_parsed_result)

        d = (
            parsed_result.agent_1_evaluation.dict()
            if self.actor_role == "agent1"
            else parsed_result.agent_2_evaluation.dict()
        )
        overall_score = 0
        for dimension in d:
            overall_score += d[dimension][1]

        overall_score /= len(d)
        return overall_score, parsed_result.agent_1_evaluation.dict(), parsed_result.agent_2_evaluation.dict()

    def get_prompt_for_fake_dialog(self, agent=1, type=None):
        if agent == 1:
            intro = self.agent1_intro
        else:
            intro = self.agent2_intro
        res = "\n".join(intro.split("\n")[:-1]).replace(self.additional_instruction[:-1], "")
        if type == 0:
            res += "\nGenerate 16 sentences in the format of 1. 'Sentence_1', 2. 'Sentence_2' ..., 16. 'Sentence_16', simulating possible words the agent might say in this interaction. Keep in mind the agent's goal. Generation:"
        elif type == 1:
            res += "\nGenerate 16 sentences in the format of 1. 'Sentence_1', 2. 'Sentence_2' ..., 16. 'Sentence_16'. The sentences should be the least likely things the agent would say in this interaction. Generation:"
        return res

    def _act(self, actor: int, save_states: bool = False, turn: int = 0):
        del save_states, turn  # API path has no hidden states
        prompt = self.get_current_prompt()
        self.prompts.append(prompt)
        vm = self.vertex_model_1 if actor == 1 else self.vertex_model_2
        raw = _vertex_generate(
            vm,
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            vertex_project=self.vertex_project,
            vertex_location=self.vertex_location,
        )
        result = _postprocess_spoken_line(raw)
        intended_result = result

        self.cur_turn += 1
        self.cur_dialog[-1] += result
        self.intended_dialog[-1] += intended_result

    def step(self):
        self._act(actor=1, save_states=False, turn=self.cur_turn)

        done = self._judge_terminate()
        if done:
            return done

        self.cur_dialog.append(
            f"Turn {self.cur_turn}: {self.agent2_name} said:"
            if self.cur_turn % 2 != 0
            else f"Turn {self.cur_turn}: {self.agent1_name} said:"
        )
        self.intended_dialog.append(
            f"Turn {self.cur_turn}: {self.agent2_name} said:"
            if self.cur_turn % 2 != 0
            else f"Turn {self.cur_turn}: {self.agent1_name} said:"
        )

        self._act(actor=2, save_states=False, turn=self.cur_turn)

        done = self._judge_terminate()
        if done:
            return done

        self.cur_dialog.append(
            f"Turn {self.cur_turn}: {self.agent2_name} said:"
            if self.cur_turn % 2 != 0
            else f"Turn {self.cur_turn}: {self.agent1_name} said:"
        )
        self.intended_dialog.append(
            f"Turn {self.cur_turn}: {self.agent2_name} said:"
            if self.cur_turn % 2 != 0
            else f"Turn {self.cur_turn}: {self.agent1_name} said:"
        )

        self.cur_rewards.append(0)
        self.terminations.append(False)

        return done
