#!/usr/bin/env python
"""
6_judge_responses.py
====================
The response judge. Every response with a final answer (budget-exhausted ones are skipped) is labelled, from the
question and the response with any reasoning trace removed (cut to 12,000 characters), as exactly one of

  ABSTAIN   declines: says it does not know / the context lacks the answer, without denying the item or answering
  DENY      states that the asked-about item does not exist
  ANSWER    answers the question, treating the item as real

The paper uses Gemma 4 26B-A4B (8-bit, thinking disabled) at temperature 0. Labels go to <dir>/labels/<results
file> (question, source, label, raw); rerunning labels only what is missing.

    python 6_judge_responses.py --dir results --url http://localhost:8000/v1 --model gemma-4-26b-a4b-it
"""
import argparse
import csv
import glob
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from config import LLM_URL, make_client
from prompts import RESPONSE_JUDGE_PROMPT

csv.field_size_limit(sys.maxsize)
LABELS = ("ANSWER", "DENY", "ABSTAIN")


MAX_ANSWER_CHARS = 12000


def strip_thinking(text):
    t = str(text or "")
    if "</think>" in t:
        t = t.split("</think>")[-1]
    return re.sub(r"<think>.*?(</think>|$)", "", t, flags=re.S).strip()


def classify(client, model, question, answer):
    body = strip_thinking(answer)[:MAX_ANSWER_CHARS]
    if not body:
        return "EMPTY", ""
    raw = ""
    for _ in range(3):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0.0, max_tokens=8,
                messages=[{"role": "user", "content": RESPONSE_JUDGE_PROMPT.format(question=question, answer=body)}],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            raw = (r.choices[0].message.content or "").strip()
            m = re.search(r"\b(DENY|ABSTAIN|ANSWER)\b", raw.upper())
            if m:
                return m.group(1), raw
        except Exception as e:
            raw = f"error: {e}"[:200]
    return "unparsed", raw


def main():
    ap = argparse.ArgumentParser(description="Label every response ABSTAIN, DENY or ANSWER.")
    ap.add_argument("--dir", default="results")
    ap.add_argument("--url", default=LLM_URL)
    ap.add_argument("--model", default="gemma-4-26b-a4b-it")
    ap.add_argument("-w", "--workers", type=int, default=64)
    args = ap.parse_args()

    out_dir = os.path.join(args.dir, "labels")
    os.makedirs(out_dir, exist_ok=True)
    todo = []
    for path in sorted(glob.glob(os.path.join(args.dir, "results_*.csv"))):
        d = pd.read_csv(path, engine="python", dtype=str, keep_default_na=False)
        d = d[d.refusal_eval != "budget_exhausted"]
        side = os.path.join(out_dir, os.path.basename(path))
        done = set(pd.read_csv(side, dtype=str, engine="python").question) if os.path.exists(side) else set()
        todo += [(side, q, s, a) for q, s, a in zip(d.question, d.source, d.answer) if q not in done]
    print(f"{len(todo)} responses to label with {args.model}", flush=True)
    client = make_client(args.url)
    lock = threading.Lock()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(classify, client, args.model, q, a): (side, q, s) for side, q, s, a in todo}
        for fut in tqdm(as_completed(futs), total=len(futs)):
            side, q, s = futs[fut]
            label, raw = fut.result()
            with lock:
                new = not os.path.exists(side)
                with open(side, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow(["question", "source", "label", "raw"])
                    w.writerow([q, s, label, raw])


if __name__ == "__main__":
    main()
