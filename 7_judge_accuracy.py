#!/usr/bin/env python
"""
7_judge_accuracy.py
===================
One accuracy judge scores answers against the reference answers: 1.0 correct, 0.5 partly correct, 0.0 incorrect
(prompt in prompts.py; thinking disabled, temperature 0). Every non-empty answer is scored whatever its response
label; a refusal (ABSTAIN) and a budget-exhausted answer count as 0 in 8_metrics.py. Run it once per judge (the
paper uses Gemma 4 31B, Qwen3.8 27B and Laguna S 2.1, 8-bit) with its own --tag; Answer Accuracy is their mean.
--unanswerable scores the post-cutoff questions against the unanswerable references (the leakage audit).

    python 7_judge_accuracy.py --dir results --references references_answerable.csv --model gemma-4-31b-it --tag gemma
    python 7_judge_accuracy.py --dir results --references references_unanswerable.csv --unanswerable --tag gemma

Scores go to <dir>/accuracy_<tag>/ (or leakage_<tag>/) with one CSV per results file (question, score).
"""
import argparse
import csv
import glob
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional

from openai import OpenAI
from tqdm import tqdm

from config import LLM_URL, make_client
from prompts import ACCURACY_JUDGE_PROMPT

csv.field_size_limit(sys.maxsize)


PREFLIGHT_CASES = [
    ("How many hearts of health does a cow have?", "A cow has 10 health points, which is 5 hearts.",
     "Cows have five hearts (10 HP).", 1.0),
    ("How many hearts of health does a cow have?", "A cow has 10 health points, which is 5 hearts.",
     "A cow has 20 hearts of health.", 0.0),
]


def _blank(value) -> bool:
    """True if a score cell is unset. A score of 0.0 is falsy, so `value or ""` treated
    it as unset: zeros were dropped from the mean and would have been re-scored."""
    return value is None or str(value).strip() == ""


def query_judge_score(client: OpenAI, question: str, ground_truth: str, candidate: str,
                      judge_model: str) -> Optional[float]:
    """Returns 0.0, 0.5 or 1.0, or None if the judge gave no parsable score after 3 attempts."""
    prompt = ACCURACY_JUDGE_PROMPT.format(question=question, ground_truth=ground_truth, candidate=candidate)
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=64,
                # The judge must not reason: sent explicitly because defaults differ by model.
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = (response.choices[0].message.content or "").strip()
            match = re.search(r"\b(1\.0|0\.5|0\.0|1|0)\b", content)
            if match:
                return float(match.group(1))
        except Exception:
            pass
        time.sleep(2.0 ** attempt)
    return None


def preflight(client: OpenAI, judge_model: str) -> None:
    for question, truth, candidate, expected in PREFLIGHT_CASES:
        got = query_judge_score(client, question, truth, candidate, judge_model)
        if got != expected:
            raise RuntimeError(f"Judge preflight failed: expected {expected}, got {got} for '{candidate}'")


def load_references(path: str, source: str) -> Dict[str, str]:
    """Reference answers for one half (rows the oracle answered)."""
    refs = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("source") == source and str(row.get("answered", "0")).strip() in ("1", "1.0"):
                if (row.get("question") or "").strip() and (row.get("answer") or "").strip():
                    refs[row["question"].strip()] = row["answer"].strip()
    print(f"Loaded {len(refs)} reference answers ({source}) from {path}")
    return refs


def main():
    ap = argparse.ArgumentParser(description="Score answers against reference answers with one LLM judge.")
    ap.add_argument("--dir", default="results")
    ap.add_argument("--references", required=True)
    ap.add_argument("--url", default=LLM_URL)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="short judge name, used for the output folder")
    ap.add_argument("--unanswerable", action="store_true", help="leakage audit: score post-cutoff questions")
    ap.add_argument("--match", default=None, help="only results files whose name matches this regex")
    ap.add_argument("-w", "--workers", type=int, default=64)
    args = ap.parse_args()

    source = "latest" if args.unanswerable else "Pre-cutoff"
    refs = load_references(args.references, source)
    out_dir = os.path.join(args.dir, f"{'leakage' if args.unanswerable else 'accuracy'}_{args.tag}")
    os.makedirs(out_dir, exist_ok=True)
    client = make_client(args.url, timeout=120)
    preflight(client, args.model)

    todo = []
    for path in sorted(glob.glob(os.path.join(args.dir, "results_*.csv"))):
        if args.match and not re.search(args.match, os.path.basename(path)):
            continue
        side = os.path.join(out_dir, os.path.basename(path))
        done = set()
        if os.path.exists(side):
            with open(side, encoding="utf-8") as f:
                done = {r["question"] for r in csv.DictReader(f) if not _blank(r.get("score"))}
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                q = r["question"].strip()
                if r.get("source") == source and q in refs and q not in done:
                    todo.append((side, q, (r.get("answer") or "").strip()))
    print(f"{len(todo)} answers to score with {args.model}", flush=True)

    def score(q, answer):
        if not answer:
            return 0.0                       # no final answer
        if answer == refs[q]:
            return 1.0
        return query_judge_score(client, q, refs[q], answer, args.model)

    lock = threading.Lock()
    failed = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(score, q, a): (side, q) for side, q, a in todo}
        for fut in tqdm(as_completed(futs), total=len(futs)):
            side, q = futs[fut]
            s = fut.result()
            if s is None:
                failed += 1                   # left unscored; a rerun retries it
                continue
            with lock:
                new = not os.path.exists(side)
                with open(side, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow(["question", "score"])
                    w.writerow([q, s])
    if failed:
        print(f"{failed} answers could not be scored; rerun to retry them")


if __name__ == "__main__":
    main()
