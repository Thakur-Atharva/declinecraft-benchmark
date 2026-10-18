#!/usr/bin/env python
"""
3_generate.py
=============
Scenario-first question generation. Both halves follow one procedure; only the corpus of the target chunk differs.

  1. Scenario   three random C_pre chunks -> the generator writes a player scenario (< 30 words).
  2. Target     the scenario is embedded and the ten most similar chunks are retrieved (dense) from C_pre
                (answerable) or C_post (unanswerable); technical pages and, for answerable questions, the
                scenario's own chunks are dropped; the target is drawn at random from the five highest left.
  3. Fact       the generator lists up to six facts the target states and a comparison text does not (answerable:
                the scenario's three chunks; unanswerable: the three C_pre chunks nearest the target); one is drawn.
  4. Question   the generator writes one question whose answer is that fact; it is appended to the scenario.
                Drafts with novelty cue words ("new", "recently", a month or year, ...) or not one question are dropped.
  5. Verify     a separate verifier, reading C_pre through hybrid retrieval + reranking (10 chunks per query, at
                most 12): answerable = the scenario does not state the fact AND C_pre answers the question;
                unanswerable = C_pre does not state the fact AND C_pre does not answer the question.
  6. Assemble   at most 2 questions per target chunk and 10 per page, up to 700 answerable and 300 unanswerable.

The stages run separately so each model is loaded once on a server that serves one model at a time:

    python 3_generate.py --stage generate --generator-url URL --generator-model qwen3.5-122b-a10b
    python 3_generate.py --stage judge    --verifier-url URL  --verifier-model gemma-4-31b-it
    python 3_generate.py --stage assemble --out questions.csv
"""
import argparse
import json
import os
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import pandas as pd
from openai import OpenAI

from config import CHROMA_DB_PATH, LLM_URL, TEST_COLLECTION_NAME, TRAIN_COLLECTION_NAME, get_node_key
from prompts import (ANSWER_DRIFT_PROMPT, CUE_FILTER_RE, FACT_EXTRACTION_PROMPT, FACT_QUESTION_PROMPT_PLAIN,
                     FACT_QUESTION_PROMPT_STYLED, FACT_VERIFY_PROMPT, SCENARIO_SYSTEM_PROMPT, TOPIC_LINE)
from retrieval import build_retriever, get_embedding_function


NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


TECHNICAL_PAGE_RE = re.compile(
    r"^(Debug |Display of |Block (conversion|definition|offset|variant)$|Data component format"
    r"|DataFixerUpper$|Minecraft Server Management Protocol$|Non-model-dependent block elements$"
    r"|Isotropic$|Culling$|Inventory item shading$|Hand animation$|Fog definition$|Waypoint style$"
    r"|Projected Random Patchy Square$|File extensions$|Controls/|Far Lands|Anomalous world seeds$"
    r"|Starter Seeds$|Random Tick$|Green particle$|Black entities$|Vibrant Visuals/Configurations$"
    r"|Mob type$|Spread$|Named update$|End sky$)"
)


@dataclass
class Endpoint:
    name: str
    client: OpenAI
    model: str


def make_endpoint(url: str, model: str, timeout: float = 900.0) -> Endpoint:
    """An OpenAI-compatible endpoint serving one model."""
    return Endpoint(model, OpenAI(base_url=url, api_key=get_node_key(url) or "EMPTY", timeout=timeout,
                                  max_retries=2), model)


def complete(ep: Endpoint, prompt: str, max_tokens: int, temperature: float = 0.0) -> str:
    """One chat completion with reasoning disabled; returns visible text."""
    r = ep.client.chat.completions.create(
        model=ep.model, temperature=temperature, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}], extra_body=NO_THINK)
    text = r.choices[0].message.content or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def ask_yes_no(ep: Endpoint, prompt: str) -> Optional[bool]:
    """True for YES, False for NO, None if the answer is neither (or the call failed)."""
    try:
        a = complete(ep, prompt, max_tokens=8).upper()
    except Exception:
        return None
    if a.startswith("YES"):
        return True
    if a.startswith("NO"):
        return False
    return None


def panel_yes(judges: List[Endpoint], prompt: str) -> Tuple[bool, Dict[str, Optional[bool]]]:
    """
    Asks every judge. The panel says YES if any judge says YES or gives no
    verdict: both outcomes here reject a fact or a question, and an unreadable
    verdict must not let an item through.
    """
    verdicts = {j.name: ask_yes_no(j, prompt) for j in judges}
    return any(v is not False for v in verdicts.values()), verdicts


def panel_all_yes(judges: List[Endpoint], prompt: str) -> Tuple[bool, Dict[str, Optional[bool]]]:
    """
    YES only if every judge says YES. Used where YES is the accepting answer
    (answerable mode: the fact must really be in C_pre), so that an unreadable
    verdict rejects the item, as in panel_yes.
    """
    verdicts = {j.name: ask_yes_no(j, prompt) for j in judges}
    return all(v is True for v in verdicts.values()), verdicts


class CPreContext:
    """C_pre context for judging: hybrid retrieval with cross-encoder reranking."""

    def __init__(self, embedding_function, max_chunks: int = 12):
        self.retriever = build_retriever("hybrid_rerank", embedding_function=embedding_function)
        self.max_chunks = max_chunks
        # The evaluation retrieves with plain hybrid and no reranker, so answerable questions
        # are checked against exactly what the model will be shown, not a deeper search.
        self.plain = build_retriever("hybrid", embedding_function=embedding_function)

    def top_ids(self, query: str, k: int = 5) -> List[str]:
        return [d.get("id") for d in self.plain.retrieve(query, k=k)]

    def get(self, queries: List[str], k_per_query: int = 10) -> Tuple[List[str], List[str]]:
        seen, docs, titles = set(), [], []
        for q in queries:
            for d in self.retriever.retrieve(q, k=k_per_query):
                if d["text"] in seen:
                    continue
                seen.add(d["text"])
                docs.append(d["text"])
                titles.append((d.get("metadata") or {}).get("title", ""))
        return docs[: self.max_chunks], titles[: self.max_chunks]


def question_sentence(text: str) -> str:
    """The asked sentence of a scenario+question, without a 'Suddenly, I wondered:' lead-in."""
    parts = [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p]
    s = next((p for p in reversed(parts) if p.endswith("?")), parts[-1] if parts else text)
    return s.split(":", 1)[1].strip() if "wondered" in s.lower() and ":" in s else s


def _parse_facts(text: str) -> Optional[List[str]]:
    try:
        facts = json.loads(text[text.find("{"): text.rfind("}") + 1])["facts"]
        return [f.strip() for f in facts if isinstance(f, str) and f.strip()]
    except Exception:
        return None


class Corpus:
    """The two collections, and the sampling and similarity lookups the pipeline needs."""

    def __init__(self, embedding_function):
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.pre = client.get_collection(TRAIN_COLLECTION_NAME, embedding_function=embedding_function)
        self.post = client.get_collection(TEST_COLLECTION_NAME, embedding_function=embedding_function)
        self.pre_ids: List[str] = self.pre.get(include=[])["ids"]

    def sample_pre(self, rng: random.Random, n: int = 3) -> List[Tuple[str, str, Dict[str, Any]]]:
        got = self.pre.get(ids=rng.sample(self.pre_ids, n), include=["documents", "metadatas"])
        return list(zip(got["ids"], got["documents"], got["metadatas"]))

    def similar(self, text: str, corpus: str, k: int = 10) -> List[Tuple[str, str, Dict[str, Any]]]:
        col = self.post if corpus == "post" else self.pre
        r = col.query(query_texts=[text], n_results=k, include=["documents", "metadatas"])
        return list(zip(r["ids"][0], r["documents"][0], r["metadatas"][0]))


def _generate_candidate(mode: str, generator: Endpoint, corpus: Corpus, rng: random.Random,
                        max_words: int, cue_filter: bool, style_rules: bool,
                        top_targets: int = 5) -> Dict[str, Any]:
    """Generator-only phase: scenario, target chunk, premise, question."""
    rec: Dict[str, Any] = {"mode": mode}
    try:
        scen_chunks = corpus.sample_pre(rng, 3)
        scen_text = "\n\n---\n\n".join(d for _, d, _ in scen_chunks)
        scenario = complete(generator, f"Context:\n{scen_text}\n\n{SCENARIO_SYSTEM_PROMPT}\n\nScenario:",
                            max_tokens=300, temperature=0.3)
        rec.update(scenario=scenario, scenario_chunks=[i for i, _, _ in scen_chunks],
                   scenario_titles=[m.get("title", "") for _, _, m in scen_chunks])
        if not scenario:
            return dict(rec, outcome="empty_scenario")

        # The target chunk is whatever the scenario itself leads to, in the half's corpus.
        targets = [t for t in corpus.similar(scenario, "post" if mode == "unanswerable" else "pre", k=10)
                   if not TECHNICAL_PAGE_RE.search(t[2].get("title", ""))]
        if mode == "answerable":
            scen_ids = {i for i, _, _ in scen_chunks}
            targets = [t for t in targets if t[0] not in scen_ids]
        if not targets:
            return dict(rec, outcome="no_target_chunk")
        target = targets[rng.randrange(min(top_targets, len(targets)))]
        rec.update(chunk_id=target[0], page=target[2].get("title", ""))

        # Facts the player does not already have: for an answerable question, what the
        # scenario does not say; for an unanswerable one, what C_pre does not say.
        old = scen_text if mode == "answerable" else "\n---\n".join(
            d for _, d, _ in corpus.similar(target[1], "pre", k=3))
        facts = _parse_facts(complete(generator, FACT_EXTRACTION_PROMPT.format(old=old, new=target[1]),
                                      max_tokens=1500))
        if not facts:
            return dict(rec, outcome="no_facts")
        fact = rng.choice(facts)
        rec["fact"] = fact

        template = FACT_QUESTION_PROMPT_STYLED if style_rules else FACT_QUESTION_PROMPT_PLAIN
        q = complete(generator, template.format(scenario=scenario, fact=fact, chunk=target[1],
                                                avoid="", topic_line=TOPIC_LINE[mode]),
                     max_tokens=200, temperature=0.5)
        question = f"{scenario.strip()} {q.strip()}".strip()
        rec["question"] = question
        if not q or question.count("?") != 1 or len(question.split()) > max_words:
            return dict(rec, outcome="bad_format")
        cue = CUE_FILTER_RE.search(q)
        if cue and cue_filter:
            return dict(rec, outcome="banned_phrase", banned_phrase=cue.group(0))
        if cue:
            rec["cue_in_question"] = cue.group(0)
        return dict(rec, outcome="candidate")
    except Exception as exc:
        return dict(rec, outcome="error", error=f"{type(exc).__name__}: {str(exc)[:200]}")


def _judge_candidate(rec: Dict[str, Any], judges: List[Endpoint], context: CPreContext,
                     mode: str) -> Dict[str, Any]:
    """Judge-only phase: is the premise where it should be, and does C_pre answer the question?"""
    try:
        if mode == "answerable":
            stated, verdicts = panel_yes(judges, FACT_VERIFY_PROMPT.format(
                fact=rec["fact"], context=rec["scenario"]))
            rec["scenario_verdicts"] = verdicts
            if stated:
                return dict(rec, outcome="scenario_states_fact")
        else:
            docs, titles = context.get([rec["fact"]])
            in_pre, verdicts = panel_yes(judges, FACT_VERIFY_PROMPT.format(
                fact=rec["fact"], context="\n---\n".join(docs)))
            rec.update(novelty_verdicts=verdicts, novelty_evidence_titles=titles)
            if in_pre:
                return dict(rec, outcome="fact_known_to_c_pre")

        docs, titles = context.get([question_sentence(rec["question"]), rec["question"]])
        drift = ANSWER_DRIFT_PROMPT.format(question=rec["question"], answer=rec["fact"],
                                           context="\n---\n".join(docs))
        if mode == "answerable":
            answerable, verdicts = panel_all_yes(judges, drift)
            rec.update(drift_verdicts=verdicts, drift_evidence_titles=titles)
            if not answerable:
                return dict(rec, outcome="not_answerable_from_c_pre")
        else:
            in_pre, verdicts = panel_yes(judges, drift)
            rec.update(drift_verdicts=verdicts, drift_evidence_titles=titles)
            if in_pre:
                return dict(rec, outcome="drift_answerable_from_c_pre")
        return dict(rec, outcome="judged_ok")
    except Exception as exc:
        return dict(rec, outcome="error", error=f"{type(exc).__name__}: {str(exc)[:200]}")


def generate_candidates(mode: str, n: int, generator: Endpoint, corpus: Corpus, out_path: str,
                        workers: int = 32, seed: int = 42, max_words: int = 55,
                        cue_filter: bool = True, style_rules: bool = False,
                        start: int = 0) -> Dict[str, Any]:
    """Draft n questions with the generator only, appending each to out_path as JSONL."""
    t0 = time.time()
    outcomes: Counter = Counter()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "a") as f, ThreadPoolExecutor(workers) as ex:
        for i, rec in enumerate(ex.map(
                lambda t: _generate_candidate(mode, generator, corpus,
                                              random.Random(seed * 1_000_003 + t), max_words,
                                              cue_filter, style_rules), range(start, start + n)), 1):
            outcomes[rec["outcome"]] += 1
            f.write(json.dumps(rec) + "\n")
            if i % 100 == 0:
                f.flush()
                print(f"[{mode}] drafted {i}/{n} ({time.time() - t0:.0f}s) {dict(outcomes)}", flush=True)
    stats = {"stage": "generate", "mode": mode, "drafted": n, "outcomes": dict(outcomes),
             "candidates": outcomes["candidate"], "seconds": round(time.time() - t0)}
    print(f"[{mode}] generate: {json.dumps(stats)}", flush=True)
    return stats


def judge_candidates(mode: str, in_path: str, out_path: str, judges: List[Endpoint],
                     context: CPreContext, workers: int = 32) -> Dict[str, Any]:
    """Judge every candidate in in_path with the judges only; writes verdicts to out_path.
    Drafts already rejected locally pass through, and a rerun re-judges nothing."""
    t0 = time.time()
    recs = [json.loads(l) for l in open(in_path)]
    done = {}
    if os.path.exists(out_path):
        for line in open(out_path):
            r = json.loads(line)
            done[r.get("question") or id(r)] = r
    todo = [r for r in recs if r["outcome"] == "candidate" and (r.get("question") not in done)]
    passthrough = [r for r in recs if r["outcome"] != "candidate" and (r.get("question") not in done)]
    print(f"[{mode}] judging {len(todo)} candidates ({len(done)} already judged)", flush=True)

    outcomes: Counter = Counter(r["outcome"] for r in done.values())
    with open(out_path, "a") as f, ThreadPoolExecutor(workers) as ex:
        for rec in passthrough:
            outcomes[rec["outcome"]] += 1
            f.write(json.dumps(rec) + "\n")
        for i, rec in enumerate(ex.map(lambda r: _judge_candidate(r, judges, context, mode), todo), 1):
            outcomes[rec["outcome"]] += 1
            f.write(json.dumps(rec) + "\n")
            if i % 100 == 0:
                f.flush()
                print(f"[{mode}] judged {i}/{len(todo)} ({time.time() - t0:.0f}s) {dict(outcomes)}", flush=True)
    stats = {"stage": "judge", "mode": mode, "judged": len(todo), "outcomes": dict(outcomes),
             "passed": outcomes["judged_ok"], "seconds": round(time.time() - t0)}
    print(f"[{mode}] judge: {json.dumps(stats)}", flush=True)
    return stats


def assemble(mode: str, judged_path: str, target: int, per_page_cap: int = 10,
             per_chunk_cap: int = 2) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Pick the final set from the judged drafts: premise caps only (no model calls).

    The chunk cap is the real constraint: no more than per_chunk_cap questions may rest on
    the same premise chunk. A page cap alone cannot do this job, because C_post pages are
    wildly uneven (712 chunks over 170 pages, median 2 per page, 64 for "Abandoned Camp"),
    so it both starves rich pages and still allowed 7 questions from one chunk. The page cap
    stays as a backstop: with chunks alone, one 64-chunk page took 54 of 300 questions."""
    recs = [json.loads(l) for l in open(judged_path)]
    passed = [r for r in recs if r["outcome"] == "judged_ok"]
    outcomes: Counter = Counter(r["outcome"] for r in recs if r["outcome"] != "judged_ok")
    accepted: List[Dict[str, Any]] = []
    per_page: Counter = Counter()
    per_chunk: Counter = Counter()

    for rec in passed:
        if len(accepted) >= target:
            outcomes["over_target"] += 1
            continue
        if per_chunk[rec["chunk_id"]] >= per_chunk_cap:
            outcomes["chunk_cap"] += 1
            continue
        if per_page[rec["page"]] >= per_page_cap:
            outcomes["page_cap"] += 1
            continue
        per_page[rec["page"]] += 1
        per_chunk[rec["chunk_id"]] += 1
        accepted.append(dict(rec, outcome="accepted"))
    stats = {"stage": "assemble", "mode": mode, "candidates_passed": len(passed),
             "accepted": len(accepted), "target": target, "outcomes": dict(outcomes),
             "pages_used": len(per_page), "max_questions_per_page": max(per_page.values(), default=0),
             "chunks_used": len(per_chunk), "max_questions_per_chunk": max(per_chunk.values(), default=0)}
    return accepted, stats


# Drafts per half: about 69% of answerable drafts and 26% of unanswerable ones survive verification and the caps.
HALVES = {"answerable": {"target": 700, "drafts": 1300}, "unanswerable": {"target": 300, "drafts": 1300}}


def paths(out_dir, half, round_no):
    return (os.path.join(out_dir, f"{half}_drafts_r{round_no}.jsonl"),
            os.path.join(out_dir, f"{half}_judged_r{round_no}.jsonl"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["generate", "judge", "assemble", "all"], default="all")
    ap.add_argument("--out-dir", default="build")
    ap.add_argument("--out", default="questions.csv")
    ap.add_argument("--generator-url", default=LLM_URL)
    ap.add_argument("--generator-model", default="qwen3.5-122b-a10b")
    ap.add_argument("--verifier-url", default=LLM_URL)
    ap.add_argument("--verifier-model", default="gemma-4-31b-it")
    ap.add_argument("--answerable-target", type=int, default=HALVES["answerable"]["target"])
    ap.add_argument("--unanswerable-target", type=int, default=HALVES["unanswerable"]["target"])
    ap.add_argument("--answerable-drafts", type=int, default=HALVES["answerable"]["drafts"])
    ap.add_argument("--unanswerable-drafts", type=int, default=HALVES["unanswerable"]["drafts"])
    ap.add_argument("--per-chunk-cap", type=int, default=2)
    ap.add_argument("--per-page-cap", type=int, default=10)
    ap.add_argument("--round", type=int, default=1, help="top-up round: fresh seeds, same stages")
    ap.add_argument("-w", "--workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    targets = {"answerable": args.answerable_target, "unanswerable": args.unanswerable_target}
    drafts = {"answerable": args.answerable_drafts, "unanswerable": args.unanswerable_drafts}
    ef = get_embedding_function()

    if args.stage in ("generate", "all"):
        corpus, generator = Corpus(ef), make_endpoint(args.generator_url, args.generator_model)
        for half in ("unanswerable", "answerable"):
            out, _ = paths(args.out_dir, half, args.round)
            if os.path.exists(out):
                print(f"[{half}] drafts exist, skipping: {out}", flush=True)
                continue
            generate_candidates(half, drafts[half], generator, corpus, out, workers=args.workers,
                                seed=args.seed + 1000 * args.round, start=args.round * 100_000)

    if args.stage in ("judge", "all"):
        judges, context = [make_endpoint(args.verifier_url, args.verifier_model)], CPreContext(ef)
        for half in ("unanswerable", "answerable"):
            src, dst = paths(args.out_dir, half, args.round)
            if os.path.exists(src):
                judge_candidates(half, src, dst, judges, context, workers=args.workers)

    if args.stage in ("assemble", "all"):
        frames = []
        for half in ("answerable", "unanswerable"):
            judged = sorted(f for f in os.listdir(args.out_dir) if f.startswith(f"{half}_judged_r"))
            if not judged:
                continue
            merged = os.path.join(args.out_dir, f"{half}_judged_all.jsonl")
            with open(merged, "w") as out:
                for name in judged:
                    out.write(open(os.path.join(args.out_dir, name)).read())
            accepted, st = assemble(half, merged, targets[half], per_page_cap=args.per_page_cap,
                                    per_chunk_cap=args.per_chunk_cap)
            print(f"[{half}] assemble: {json.dumps(st)}", flush=True)
            frames.append(pd.DataFrame([{
                "question": r["question"],
                "source": "latest" if half == "unanswerable" else "Pre-cutoff",
                "scenario_chunks": str(r["scenario_chunks"]), "scenario_titles": str(r["scenario_titles"]),
                "question_chunks": str([r["chunk_id"]]), "question_titles": str([r["page"]]),
                "target_fact": r["fact"],
            } for r in accepted]))
            if st["accepted"] < targets[half]:
                print(f"WARNING: {half} is {targets[half] - st['accepted']} short; run again with "
                      f"--round {args.round + 1} and more drafts", flush=True)
        if frames:
            out = pd.concat(frames, ignore_index=True)
            out.to_csv(args.out, index=False)
            print(f"wrote {len(out)} questions -> {args.out}  (source: Pre-cutoff = answerable, latest = unanswerable)")


if __name__ == "__main__":
    main()
