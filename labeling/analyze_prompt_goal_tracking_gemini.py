"""
Analyze goal tracking in Gemini-generated Sotopia outputs.

This is adapted from analyze_prompt_goal_tracking.py, but defaults to scanning:
    sotopia_results_gemini/**/prompt_records/*.csv

Usage:
    python analyze_prompt_goal_tracking_gemini.py
    python analyze_prompt_goal_tracking_gemini.py --only dialogs/.../prompt_records/0_temp0.7_seed0.csv
    python analyze_prompt_goal_tracking_gemini.py --limit 5
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

SCRIPT_DIR = Path(__file__).resolve().parent

# Same API settings used in analyze_prompt_goal_tracking.py
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

REPAIR_SYSTEM = """\
You are a strict JSON repair assistant.
Given malformed JSON, output ONLY a corrected JSON object.
Do not add markdown fences or extra commentary.
Preserve the original meaning and fields as much as possible."""


def _clean_tags(text: str) -> str:
    return re.sub(r"<.*?>", "", text, flags=re.DOTALL)


def _extract_first_json_object(text: str) -> str:
    """Extract the first balanced JSON object from text."""
    if not text:
        return text
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return text[start:]


def _repair_json_with_model(
    model: str,
    raw_text: str,
    max_retries: int = 2,
) -> dict | None:
    repair_user = (
        "Malformed JSON (fix it into valid JSON object only):\n\n"
        f"{raw_text[:12000]}"
    )
    for attempt in range(max_retries):
        try:
            response = completion(
                model=model,
                messages=[
                    {"role": "system", "content": REPAIR_SYSTEM},
                    {"role": "user", "content": repair_user},
                ],
                vertex_project=VERTEX_PROJECT,
                vertex_location=VERTEX_LOCATION,
                temperature=0,
                max_tokens=2048,
                timeout=120,
                num_retries=2,
            )
            repaired_raw = response.choices[0].message.content.strip()
            repaired = _extract_first_json_object(repaired_raw)
            return json.loads(repaired)
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None
    return None


def extract_transcript_from_prompts(prompt_rows: list[str]) -> str:
    """Take the longest prompt row and pull Turn N: ... lines into a readable transcript."""
    if not prompt_rows:
        return ""
    longest = max(prompt_rows, key=len)
    longest = _clean_tags(longest)
    pattern = re.compile(
        r"(Turn \d+:[^\n]*? said:)(.*?)(?=\nTurn \d+:|$)",
        re.DOTALL | re.IGNORECASE,
    )
    chunks = []
    for match in pattern.finditer(longest):
        prefix, speech = match.group(1), match.group(2).strip()
        chunks.append(prefix + " " + speech[:2000])
    if chunks:
        return "\n".join(chunks)
    return longest[-12000:]


def read_prompt_csv(path: Path) -> list[str]:
    rows: list[str] = []
    with path.open(newline="", encoding="utf-8") as f:
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
    raw = ""
    for attempt in range(max_retries):
        try:
            response = completion(
                model=model,
                messages=[
                    {"role": "system", "content": ANALYSIS_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                vertex_project=VERTEX_PROJECT,
                vertex_location=VERTEX_LOCATION,
                temperature=0,
                max_tokens=2048,
                timeout=120,
                num_retries=3,
            )
            raw = response.choices[0].message.content.strip()
            raw_json = _extract_first_json_object(raw)
            try:
                return json.loads(raw_json)
            except json.JSONDecodeError:
                repaired = _repair_json_with_model(model, raw)
                if repaired is not None:
                    repaired["_repaired_json"] = True
                    return repaired
                raise
        except (json.JSONDecodeError, KeyError) as error:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1} (parse): {error}")
                time.sleep(1)
            else:
                return {
                    "summary": None,
                    "agent_1_goal_tracking": None,
                    "agent_2_goal_tracking": None,
                    "issues": [],
                    "unnatural_behaviors": [],
                    "notes": f"PARSE_ERROR: {error}",
                    "raw_response": raw[:4000],
                }
        except Exception as error:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  API error, retry in {wait}s: {error}")
                time.sleep(wait)
            else:
                raise
    return {}


def list_prompt_files(results_root: Path) -> list[Path]:
    return sorted(results_root.glob("**/prompt_records/*.csv"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default=str(SCRIPT_DIR / "sotopia_results_gemini"),
        help="Root directory that contains Gemini output folders",
    )
    parser.add_argument(
        "--output",
        default=str(SCRIPT_DIR / "analysis_outputs" / "gemini_goal_tracking_by_file.json"),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--only",
        default=None,
        help="Relative path under results-root to process (e.g. dialogs/.../prompt_records/file.csv)",
    )
    parser.add_argument("--resume", action="store_true", help="Skip keys already present in output JSON")
    parser.add_argument(
        "--retry-parse-errors",
        action="store_true",
        help="When used with --resume, reprocess entries whose notes start with PARSE_ERROR",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N files")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    if not results_root.is_dir():
        raise SystemExit(f"Not found: {results_root}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: dict = {}
    if args.resume and out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            results = json.load(f)

    if args.only:
        only_path = (results_root / args.only).resolve()
        if not only_path.is_file():
            raise SystemExit(f"Not found: {only_path}")
        files = [only_path]
    else:
        files = list_prompt_files(results_root)

    if args.limit is not None:
        files = files[: max(args.limit, 0)]

    if not files:
        print(f"No prompt_records CSV files found under {results_root}")
        return

    for i, file_path in enumerate(files, start=1):
        rel_key = str(file_path.resolve().relative_to(results_root.resolve()))
        if rel_key in results:
            if args.retry_parse_errors:
                existing = results.get(rel_key, {})
                notes = (
                    existing.get("analysis", {}).get("notes")
                    if isinstance(existing, dict)
                    else None
                )
                if isinstance(notes, str) and notes.startswith("PARSE_ERROR"):
                    print(f"[retry parse-error] {rel_key}")
                else:
                    print(f"[skip] {rel_key}")
                    continue
            else:
                print(f"[skip] {rel_key}")
                continue

        print(f"[{i}/{len(files)}] {rel_key}")
        prompt_rows = read_prompt_csv(file_path)
        transcript = extract_transcript_from_prompts(prompt_rows)
        raw_blocks = "\n\n---ROW---\n\n".join(
            f"ROW {j}:\n{row[:6000]}" for j, row in enumerate(prompt_rows[:40])
        )
        analysis = call_analyze(args.model, rel_key, transcript, raw_blocks)
        results[rel_key] = {
            "transcript_preview": transcript[:2000],
            "analysis": analysis,
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"  saved checkpoint ({len(results)} files)")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
