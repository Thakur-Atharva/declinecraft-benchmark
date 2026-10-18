#!/usr/bin/env python
"""
5_answer.py
===========
Runs one model (one thinking setting) on the benchmark under one retrieval condition and writes its answers.

  no retrieval      --no-rag                       NO_RAG_PROMPT_TEMPLATE
  single-shot RAG   -r 0  (default)                ANSWER_PROMPT_TEMPLATE with the top k = 5 chunks
  iterative RAG     -r 2                           the model may request up to two more retrieval rounds

Append "-nothink" to the model name to disable thinking (sent both as chat_template_kwargs.enable_thinking = false
and as a "/no_think" suffix; through OpenRouter, as reasoning.enabled = false). Answers are generated at the given
temperature with a 16,000-token budget, reasoning included; an answer that uses the whole budget without a final
answer is recorded with refusal_eval = budget_exhausted. Rerunning the same command resumes.

Examples:
    python 5_answer.py -i questions.csv -m gemma-4-31b-it-nothink --url http://localhost:8000/v1 --no-rag
    python 5_answer.py -i questions.csv -m gemma-4-31b-it-nothink --url http://localhost:8000/v1 -r 0
    python 5_answer.py -i questions.csv -m google/gemini-3-flash-preview --url https://openrouter.ai/api/v1 -r 2
    python 5_answer.py ... --retrieval-mode full          # retrieval ablation (dense, bm25, hybrid_rerank, ...)
    python 5_answer.py ... -k 10                          # retrieval depth

Output: <out-dir>/results_<model>_t<T>_<norag|rag0|rag2>[_<mode>][_k<k>].csv
"""
import argparse
import csv
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

from config import GENERATION_MAX_TOKENS, LLM_URL, RETRIEVAL_TOP_K, get_node_key
from prompts import ABSTAIN_R1_TEMPLATE, ANSWER_PROMPT_TEMPLATE, NO_RAG_PROMPT_TEMPLATE
from retrieval import RETRIEVAL_MODES, SelfRAGController, build_retriever, get_embedding_function, openrouter_reasoning

csv.field_size_limit(sys.maxsize)
FIELDS = ["question", "source", "answer", "thinking", "rag_attempts_needed", "refusal_eval"]
STOP = ["<|im_end|>", "<|im_start|>", "<|eot_id|>", "<end_of_turn>", "</s>"]


class RAGSystem:
    """Retrieval, optional reflection, and answer generation for one model."""

    def __init__(self, model: str, url: str, temperature: float = 0.0, max_rag_attempts: int = 0,
                 no_rag: bool = False, retrieval_mode: str = "hybrid", prompt_variant: str = "default"):
        self.no_think = model.endswith("-nothink")
        self.model = model[:-len("-nothink")] if self.no_think else model
        self.url, self.temperature, self.no_rag, self.prompt_variant = url, temperature, no_rag, prompt_variant
        # Long thinking answers under heavy batching can take minutes.
        self.client = OpenAI(base_url=url, api_key=get_node_key(url) or "EMPTY",
                             timeout=float(os.environ.get("GENERATION_TIMEOUT", 1800)), max_retries=0)
        self._tls = threading.local()
        self.controller = None
        if not no_rag:
            retriever = build_retriever(retrieval_mode, embedding_function=get_embedding_function())
            # Reflection uses the answer's temperature and token budget.
            self.controller = SelfRAGController(retriever=retriever, client=self.client, model_name=self.model,
                                                max_attempts=max_rag_attempts, no_think=self.no_think,
                                                temperature=temperature, max_tokens=GENERATION_MAX_TOKENS)

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> Optional[str]:
        self._tls.reasoning, self._tls.budget_exhausted = "", False
        if self.no_rag:
            prompt = NO_RAG_PROMPT_TEMPLATE.format(question=query)
            context_text = ""
        else:
            if not context_chunks:
                return "The provided context does not contain the answer."
            context_text = "\n---\n".join([
                f"Title: {c['metadata'].get('title', 'Unknown')}\n"
                f"Source URL: {c['metadata'].get('source_url', '')}\n"
                f"\n{c['text']}"
                for c in context_chunks
            ])
            prompt = ANSWER_PROMPT_TEMPLATE.format(context=context_text, question=query)
        if self.prompt_variant == "abstain_r1":     # Abstain-R1's own template (fine-tuned-model comparison)
            prompt = ("" if self.no_rag else f"Context:\n{context_text}\n\n") + ABSTAIN_R1_TEMPLATE.format(question=query)
        if self.no_think:
            prompt = f"{prompt} /no_think"

        extra_body = {}
        if "gemini-3" in self.model.lower():
            extra_body["reasoning"] = {"effort": "low"}
        extra_body.update(openrouter_reasoning(self.url, self.model, self.no_think))
        if "openrouter" not in (self.url or "").lower():
            # Thinking is set explicitly in both directions: defaults differ by model.
            extra_body["chat_template_kwargs"] = {"enable_thinking": not self.no_think}

        for attempt in range(4):
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature, max_tokens=GENERATION_MAX_TOKENS, stop=STOP,
                    extra_body=extra_body or None)
                choice = response.choices[0]
                msg = choice.message
                self._tls.reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
                if msg.content and msg.content.strip():
                    return msg.content.strip()
                if choice.finish_reason == "length":       # the budget ran out before a final answer
                    self._tls.budget_exhausted = True
                    return None
            except Exception as e:
                print(f"[generation error] {type(e).__name__}: {str(e)[:200]}", flush=True)
            time.sleep(2.0 ** attempt)
        return None

    def ask(self, query: str, k: int) -> Tuple[Optional[str], int]:
        if self.no_rag:
            return self.generate_answer(query, []), 0
        context, attempts = self.controller.execute_retrieval(query, k=k)
        return self.generate_answer(query, context), attempts


def run(system: RAGSystem, input_csv: str, output_csv: str, k: int, workers: int) -> None:
    done = set()
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        with open(output_csv, encoding="utf-8") as f:
            done = {r["question"] for r in csv.DictReader(f)
                    if (r.get("answer") or "").strip() or r.get("refusal_eval") == "budget_exhausted"}
    with open(input_csv, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["question"] not in done]
    print(f"{len(done)} answered, {len(rows)} to go -> {output_csv}", flush=True)
    if not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    lock = threading.Lock()

    def task(row):
        ans, attempts = system.ask(row["question"], k)
        tls = system._tls
        if ans is None and not getattr(tls, "budget_exhausted", False):
            return                                   # transient failure: a rerun retries it
        thinking = getattr(tls, "reasoning", "") or ""
        if ans and "</think>" in ans:                 # inline reasoning tags from some servers
            head, ans = ans.split("</think>", 1)
            thinking = head.split("<think>", 1)[-1].strip()
        ans = re.sub(r"<think>.*", "", ans or "", flags=re.DOTALL).strip()
        out = {"question": row["question"], "source": row["source"], "answer": ans.replace("\n", " "),
               "thinking": thinking.replace("\n", " "), "rag_attempts_needed": attempts,
               "refusal_eval": "" if ans else "budget_exhausted"}
        with lock, open(output_csv, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(out)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in tqdm(as_completed([ex.submit(task, r) for r in rows]), total=len(rows)):
            fut.result()
    if system.controller is not None and system.controller.reflection_calls:
        print(system.controller.reflection_summary())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", default="questions.csv")
    ap.add_argument("-m", "--model", required=True, help='served model name; add "-nothink" to disable thinking')
    ap.add_argument("--url", default=LLM_URL, help="OpenAI-compatible endpoint serving the model")
    ap.add_argument("-t", "--temperature", type=float, default=0.0)
    ap.add_argument("-r", "--rag-attempts", type=int, default=0, help="0 = single-shot RAG, 2 = iterative RAG")
    ap.add_argument("--no-rag", action="store_true")
    ap.add_argument("--retrieval-mode", choices=RETRIEVAL_MODES, default="hybrid")
    ap.add_argument("-k", type=int, default=RETRIEVAL_TOP_K)
    ap.add_argument("--prompt", choices=["default", "abstain_r1"], default="default")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("-w", "--workers", type=int, default=32, help="parallel requests (vLLM batches them)")
    args = ap.parse_args()

    cond = "norag" if args.no_rag else f"rag{args.rag_attempts}"
    suffix = ("" if args.retrieval_mode == "hybrid" or args.no_rag else f"_{args.retrieval_mode}")
    suffix += "" if args.k == RETRIEVAL_TOP_K or args.no_rag else f"_k{args.k}"
    name = args.model.replace("/", "-") + ("" if args.prompt == "default" else f"-{args.prompt}")
    output = args.output or os.path.join(args.out_dir, f"results_{name}_t{args.temperature}_{cond}{suffix}.csv")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    system = RAGSystem(args.model, args.url, args.temperature, args.rag_attempts, args.no_rag,
                       args.retrieval_mode, args.prompt)
    run(system, args.input, output, args.k, args.workers)


if __name__ == "__main__":
    main()
