"""
For each CSV in prompt_records/, send the logged dialogue to Gemini (Vertex)
and assess whether each agent stays aligned with their private goals across turns.

API parameters match goal_alignment_labels.classify_combo (LiteLLM completion).
Note: build_labels_v2.py does not perform API calls; labeling uses goal_alignment_labels.py.

Usage:
    python analyze_prompt_goal_tracking.py [--prompt-dir prompt_records] [--output analysis_outputs/goal_tracking_by_file.json]
    python analyze_prompt_goal_tracking.py --only 6_temp0.7_seed0.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

from litellm import completion

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Same as goal_alignment_labels.classify_combo (build_labels_v2.py has no API layer)
DEFAULT_MODEL = "vertex_ai/gemini-3.1-pro-preview"
VERTEX_PROJECT = "hs-soil-gemini"
VERTEX_LOCATION = "global"

ANALYSIS_SYSTEM = """\
You are reviewing a multi-turn dialogue between two LLM agents in a social simulation (Sotopia). \
Each agent has a private goal stated in the prompt; goals may conflict or align.

You will receive the full sequence of prompts and/or extracted transcript. Focus on:
- Whether each agent pursues their stated goal consistently across turns.
- Whether either agent forgets or contradicts concrete facts they were given (e.g. point values for items, quantities, names, constraints).
- Anything unnatural (e.g. repeating the other agent's private valuations as if they were their own, hallucinating new rules, abandoning the goal without reason).

Respond with ONLY a JSON object (no markdown fencing) with these fields:
- "summary": string, 2-4 sentences.
- "agent_1_goal_tracking": one of "strong", "mixed", "poor", "unclear"
- "agent_2_goal_tracking": same
- "issues": array of objects, each with "turn" (int or null), "agent" (string), "detail" (string) — concrete problems.
- "unnatural_behaviors": array of short strings.
- "notes": optional string for extra observations."""

USER_TEMPLATE = """\
File name: {filename}

--- Extracted dialogue (best-effort from prompts) ---
{transcript}

--- Raw prompt rows (for reference; may repeat context) ---
{raw_blocks}
"""


def _clean_tags(text: str) -> str:
    return re.sub(r"<.*?>", "", text, flags=re.DOTALL)


def extract_transcript_from_prompts(prompt_rows: list[str]) -> str:
    """Take the longest prompt row and pull Turn N: ... lines into a readable transcript."""
    if not prompt_rows:
        return ""
    longest = max(prompt_rows, key=len)
    longest = _clean_tags(longest)
    # Turn lines: capture until next Turn or end
    pattern = re.compile(
        r"(Turn \d+:[^\n]*? said:)(.*?)(?=\nTurn \d+:|$)",
        re.DOTALL | re.IGNORECASE,
    )
    chunks = []
    for m in pattern.finditer(longest):
        prefix, speech = m.group(1), m.group(2).strip()
        chunks.append(prefix + " " + speech[:2000])  # cap very long cells
    if chunks:
        return "\n".join(chunks)
    # Fallback: return tail of longest prompt
    return longest[-12000:]


def read_prompt_csv(path: Path) -> list[str]:
    rows: list[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or not header[0]:
            return rows
        for row in reader:
            if row and row[0].strip():
                rows.append(row[0])
    return rows


def call_analyze(
    model: str,
    filename: str,
    transcript: str,
    raw_blocks: str,
    max_retries: int = 3,
) -> dict:
    user_msg = USER_TEMPLATE.format(
        filename=filename,
        transcript=transcript[:50000],
        raw_blocks=raw_blocks[:80000],
    )
    prompt = f"{ANALYSIS_SYSTEM}\n\n{user_msg}"
    raw = ""
    for attempt in range(max_retries):
        try:
            response = completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                vertex_project=VERTEX_PROJECT,
                vertex_location=VERTEX_LOCATION,
                temperature=0,
                max_tokens=1024,
                timeout=120,
                num_retries=3,
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"^```json\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)
        except (json.JSONDecodeError, KeyError) as e:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1} (parse): {e}")
                time.sleep(1)
            else:
                return {
                    "summary": None,
                    "agent_1_goal_tracking": None,
                    "agent_2_goal_tracking": None,
                    "issues": [],
                    "unnatural_behaviors": [],
                    "notes": f"PARSE_ERROR: {e}",
                    "raw_response": raw[:4000],
                }
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  API error, retry in {wait}s: {e}")
                time.sleep(wait)
            else:
                raise
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt-dir",
        default=os.path.join(SCRIPT_DIR, "prompt_records"),
        help="Directory containing *_temp*_seed*.csv files",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(SCRIPT_DIR, "analysis_outputs", "goal_tracking_by_file.json"),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--only", default=None, help="Single filename under prompt-dir to process")
    parser.add_argument("--resume", action="store_true", help="Skip keys already present in output JSON")
    args = parser.parse_args()

    prompt_dir = Path(args.prompt_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: dict = {}
    if args.resume and out_path.exists():
        with open(out_path) as f:
            results = json.load(f)

    if args.only:
        files = [prompt_dir / args.only]
        if not files[0].is_file():
            raise SystemExit(f"Not found: {files[0]}")
    else:
        files = sorted(prompt_dir.glob("*.csv"))

    for i, fp in enumerate(files):
        key = fp.name
        if key in results:
            print(f"[skip] {key}")
            continue
        print(f"[{i + 1}/{len(files)}] {key}")
        prompt_rows = read_prompt_csv(fp)
        transcript = extract_transcript_from_prompts(prompt_rows)
        raw_blocks = "\n\n---ROW---\n\n".join(f"ROW {j}:\n{r[:6000]}" for j, r in enumerate(prompt_rows[:40]))
        analysis = call_analyze(args.model, key, transcript, raw_blocks)
        results[key] = {
            "transcript_preview": transcript[:2000],
            "analysis": analysis,
        }
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  saved checkpoint ({len(results)} files)")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
