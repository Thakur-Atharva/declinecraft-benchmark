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
  6. Factcheck  (unanswerable) the verifier checks that the target chunk supports every claim in the fact, explaining
                before a final verdict: a fact that adds a comparison or a change claim ("unlike ...", "a change
                from ...") or misreads its passage is old information dressed as new.
  7. Screen     (unanswerable) a judge from another model family reads the FULL C_pre pages behind the top
                retrieved chunks (up to 24,000 characters), not 12 chunks: facts in infoboxes and tables, and facts
                on pages split out of older ones, slip past step 5.
  8. Assemble   in draft order, at most 2 questions per target chunk and 10 per page, up to 700 answerable and 300
                unanswerable (and a reserve beyond them). A candidate is skipped if (a) it is on the optional list
                of questions excluded by hand, (b) its target is a C_post sub-page ("Parent/Child": reference
                tables and documentation split out of older pages), (c) the factcheck failed, (d) the screen found
                its fact stated in C_pre, or (e) its fact repeats an accepted one: pairs with embedding cosine
                >= 0.85 or word Jaccard >= 0.85 go to the verifier, which answers SAME or DIFFERENT.

The stages run separately so each model is loaded once on a server that serves one model at a time:

    python 3_generate.py --stage generate  --generator-url URL --generator-model qwen3.5-122b-a10b
    python 3_generate.py --stage judge     --verifier-url URL  --verifier-model gemma-4-31b-it
    python 3_generate.py --stage factcheck --verifier-url URL  --verifier-model gemma-4-31b-it
    python 3_generate.py --stage screen    --screen-url URL    --screen-model qwen3.8-27b
    python 3_generate.py --stage assemble  --verifier-url URL  --verifier-model gemma-4-31b-it --out questions.csv

Every stage resumes: verdicts are cached per question in --out-dir (fact_checks.jsonl, unanswerable_screen.jsonl,
dedup_pairs.jsonl), so a rerun asks only what is missing.
"""
import argparse
import json
import os
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import numpy as np
import pandas as pd
from openai import OpenAI

from config import (CHROMA_DB_PATH, LLM_URL, TEST_COLLECTION_NAME, TRAIN_COLLECTION_NAME, TRAIN_JSON_FILE,
                    get_node_key)
from prompts import (ANSWER_DRIFT_PROMPT, CPRE_EXPLAINED_PROMPT, CUE_FILTER_RE, FACT_EXTRACTION_PROMPT,
                     FACT_QUESTION_PROMPT_PLAIN, FACT_QUESTION_PROMPT_STYLED, FACT_VERIFY_PROMPT, FAITHFUL_PROMPT,
                     SAME_FACT_PROMPT, SCENARIO_SYSTEM_PROMPT, TOPIC_LINE)
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


def ask_yes_no(ep: Endpoint, prompt: str, attempts: int = 1, max_tokens: int = 8) -> Optional[bool]:
    """True for YES, False for NO, None if no attempt gives either (or every call failed)."""
    for _ in range(attempts):
        try:
            a = complete(ep, prompt, max_tokens=max_tokens).upper()
        except Exception:
            continue
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




def merge_judged(out_dir: str, half: str) -> List[Dict[str, Any]]:
    """All rounds' judged drafts of a half, in round order, also written to <half>_judged_all.jsonl."""
    rounds = sorted((int(m.group(1)), f) for f in os.listdir(out_dir)
                    if (m := re.fullmatch(rf"{half}_judged_r(\d+)\.jsonl", f)))
    recs = [json.loads(l) for _, f in rounds for l in open(os.path.join(out_dir, f))]
    if rounds:
        with open(os.path.join(out_dir, f"{half}_judged_all.jsonl"), "w") as out:
            out.writelines(json.dumps(r) + "\n" for r in recs)
    return recs


def _cached(path: str) -> Dict[str, Dict[str, Any]]:
    """A per-question JSONL cache, keyed by question."""
    if not os.path.exists(path):
        return {}
    return {d["question"]: d for d in map(json.loads, open(path))}


def chunk_texts(collection, ids: List[str]) -> Dict[str, Tuple[str, str]]:
    """{chunk id: (text, page title)}. Paired with the ids Chroma returns: get(ids=...) does not keep the
    requested order, so zipping with the requested ids would hand each id another chunk's text."""
    ids = sorted(set(ids))
    got = collection.get(ids=ids, include=["documents", "metadatas"]) if ids else {
        "ids": [], "documents": [], "metadatas": []}
    out = {i: (d, (m or {}).get("title", "")) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])}
    missing = set(ids) - set(out)
    if missing:
        raise SystemExit(f"{len(missing)} chunk ids are not in the index (e.g. {sorted(missing)[:3]}): "
                         f"was it rebuilt after the drafts were written?")
    return out


# --- Stage factcheck: does the target chunk support the fact? ----------------------------------------------


def ask_explained(ep: Endpoint, prompt: str, attempts: int = 5) -> Tuple[bool, str]:
    """A check answered with an explanation and a final 'VERDICT: YES|NO' line: (verdict, explanation)."""
    for _ in range(attempts):
        try:
            text = complete(ep, prompt, max_tokens=300)
        except Exception:
            continue
        v = re.findall(r"VERDICT:\s*\**\s*(YES|NO)", text.upper())
        if v:
            return v[-1] == "YES", text
    raise RuntimeError(f"no verdict after {attempts} attempts")


def factcheck_candidates(out_dir: str, verifier: Endpoint, context: CPreContext, workers: int = 32) -> None:
    """Two checks on every verified unanswerable candidate, explained before the verdict:
      faithful     the target chunk supports every claim in the fact (applied at assembly);
      cpre_states  the verifier's C_pre question again, on the same evidence (recorded only: it over-flags new
                   facts it reasons are implied by older ones, e.g. a new mob's spawning from its relatives').
    Writes <out_dir>/fact_checks.jsonl, cached per question."""
    recs = [r for r in merge_judged(out_dir, "unanswerable") if r["outcome"] == "judged_ok"]
    out = os.path.join(out_dir, "fact_checks.jsonl")
    todo = [r for r in recs if r["question"] not in _cached(out)]
    print(f"[factcheck] {len(recs)} verified unanswerable candidates, {len(todo)} to check", flush=True)
    if todo:
        post = chromadb.PersistentClient(path=CHROMA_DB_PATH).get_collection(TEST_COLLECTION_NAME)
        chunks = chunk_texts(post, [r["chunk_id"] for r in todo])
        lock = threading.Lock()

        def one(r):
            faithful, why_f = ask_explained(verifier, FAITHFUL_PROMPT.format(
                fact=r["fact"], passage=chunks[r["chunk_id"]][0]))
            docs, titles = context.get([r["fact"]])
            cpre, why_c = ask_explained(verifier, CPRE_EXPLAINED_PROMPT.format(
                fact=r["fact"], context="\n---\n".join(docs)))
            row = {"question": r["question"], "fact": r["fact"], "page": r["page"], "chunk_id": r["chunk_id"],
                   "faithful": faithful, "faithful_why": why_f, "cpre_states": cpre, "cpre_why": why_c,
                   "cpre_titles": titles}
            with lock, open(out, "a") as f:
                f.write(json.dumps(row) + "\n")

        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(one, todo))
    rows = _cached(out).values()
    print(f"[factcheck] {len(rows)} checked: not faithful {sum(not r['faithful'] for r in rows)}, "
          f"C_pre states it (recorded only) {sum(r['cpre_states'] for r in rows)}", flush=True)


# --- Stage screen: full C_pre pages, read by a judge from another model family -----------------------------


SCREEN_MAX_CONTEXT_CHARS = 24000


def _title_key(title: str) -> str:
    return re.sub(r"[\s_]+", " ", title).strip().lower()


def page_context(retriever, pages: Dict[str, Dict[str, Any]], queries: List[str],
                 max_chars: int = SCREEN_MAX_CONTEXT_CHARS) -> Tuple[str, List[str]]:
    """Full C_pre pages behind the top 10 chunks retrieved for each query, best first, up to max_chars."""
    titles = []
    for q in queries:
        for d in retriever.retrieve(q, k=10):
            t = (d.get("metadata") or {}).get("title", "")
            if t and t not in titles:
                titles.append(t)
    key = {_title_key(k): k for k in pages}
    blocks, used = [], 0
    for t in titles:
        k = key.get(_title_key(t))
        if not k:
            continue
        text = f"Title: {t}\n{pages[k]['content']}"
        if used + len(text) > max_chars:
            text = text[:max(0, max_chars - used)]
        if not text:
            break
        blocks.append(text)
        used += len(text)
        if used >= max_chars:
            break
    return "\n\n---\n\n".join(blocks), titles


def screen_candidates(out_dir: str, screener: Endpoint, retriever, pages: Dict[str, Dict[str, Any]],
                      workers: int = 16) -> None:
    """For every verified unanswerable candidate: does C_pre, read as whole pages, state the fact
    (cpre_states_fact, applied at assembly), and does it answer the question (cpre_answers, recorded only: old
    mechanics carried over to a new item are fine)? An unreadable verdict is None, which assembly treats as
    flagged. Writes <out_dir>/unanswerable_screen.jsonl, cached per question."""
    recs = [r for r in merge_judged(out_dir, "unanswerable") if r["outcome"] == "judged_ok"]
    out = os.path.join(out_dir, "unanswerable_screen.jsonl")
    todo = [r for r in recs if r["question"] not in _cached(out)]
    print(f"[screen] {len(recs)} verified unanswerable candidates, {len(todo)} to screen", flush=True)
    lock = threading.Lock()

    def one(r):
        asked = re.split(r"(?<=[.!?])\s+", r["question"].strip())[-1]
        ctx, titles = page_context(retriever, pages, [asked, r["fact"]])
        stated = ask_yes_no(screener, FACT_VERIFY_PROMPT.format(fact=r["fact"], context=ctx),
                            attempts=3, max_tokens=4)
        answers = ask_yes_no(screener, ANSWER_DRIFT_PROMPT.format(question=r["question"], answer=r["fact"],
                                                                  context=ctx), attempts=3, max_tokens=4)
        row = {"question": r["question"], "fact": r["fact"], "page": r["page"], "cpre_states_fact": stated,
               "cpre_answers": answers, "context_titles": titles[:6]}
        with lock, open(out, "a") as f:
            f.write(json.dumps(row) + "\n")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))
    rows = _cached(out).values()
    print(f"[screen] {len(rows)} screened: C_pre states the fact {sum(r['cpre_states_fact'] is not False for r in rows)}"
          f" (unreadable counted), C_pre answers the question (recorded only) "
          f"{sum(bool(r['cpre_answers']) for r in rows)}", flush=True)


# --- Stage assemble ----------------------------------------------------------------------------------------


SIMILAR_COSINE = 0.85      # fact pairs at or above either threshold go to the SAME/DIFFERENT judge
SIMILAR_JACCARD = 0.85


def is_subpage(title: str) -> bool:
    """A sub-page ("Sticky Piston/Table", "Ocean Ruins/Structure"). In C_post these are reference tables and
    documentation split out of older pages: the page is new, its content is not."""
    return "/" in title


def words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def jaccard(a: str, b: str) -> float:
    """Word-set Jaccard similarity of a to b (0 if a has no words)."""
    ta, tb = set(words(a)), set(words(b))
    return len(ta & tb) / len(ta | tb) if ta else 0.0


class SameFactJudge:
    """SAME / DIFFERENT verdicts on pairs of facts, cached in a JSONL file with each pair in sorted order.
    With no endpoint, a pair missing from the cache is an error (reproducing an assembly from its caches)."""

    def __init__(self, ep: Optional[Endpoint], cache_path: str):
        self.ep, self.path, self.lock = ep, cache_path, threading.Lock()
        self.cache = {}
        if os.path.exists(cache_path):
            for d in map(json.loads, open(cache_path)):
                self.cache[(d["a"], d["b"])] = d["verdict"]

    def __call__(self, a: str, b: str) -> str:
        k = (a, b) if a <= b else (b, a)
        with self.lock:
            if k in self.cache:
                return self.cache[k]
        if self.ep is None:
            raise SystemExit(f"no cached SAME/DIFFERENT verdict for a fact pair and no judge endpoint: {k}")
        for attempt in range(5):
            try:
                v = complete(self.ep, SAME_FACT_PROMPT.format(a=k[0], b=k[1]), max_tokens=4).upper()
                break
            except Exception:
                if attempt == 4:
                    raise
        verdict = "SAME" if v.startswith("SAME") else "DIFFERENT"
        with self.lock:
            self.cache[k] = verdict
            with open(self.path, "a") as f:
                f.write(json.dumps({"a": k[0], "b": k[1], "verdict": verdict}) + "\n")
        return verdict


def embed_facts(facts: List[str], embedding_function, cache_path: str) -> Dict[str, np.ndarray]:
    """Unit-length embeddings of the facts, cached in an .npz file next to the drafts."""
    cache = {}
    if os.path.exists(cache_path):
        z = np.load(cache_path)
        cache = dict(zip(z["facts"].tolist(), z["vectors"]))
    todo = [f for f in dict.fromkeys(facts) if f not in cache]
    for i in range(0, len(todo), 64):
        e = np.array(embedding_function(todo[i:i + 64]), dtype=np.float32)
        cache.update(zip(todo[i:i + 64], e / np.linalg.norm(e, axis=1, keepdims=True)))
    if todo:
        keys = list(cache)
        np.savez(cache_path, facts=np.array(keys), vectors=np.stack([cache[k] for k in keys]))
    return {f: cache[f] for f in facts}


def assemble(mode: str, recs: List[Dict[str, Any]], target: int, per_page_cap: int = 10, per_chunk_cap: int = 2,
             excluded: frozenset = frozenset(), checks: Optional[Dict[str, Dict[str, Any]]] = None,
             screen: Optional[Dict[str, Dict[str, Any]]] = None, judge: Optional[SameFactJudge] = None,
             emb: Optional[Dict[str, np.ndarray]] = None,
             workers: int = 16) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, str]]:
    """Pick the final set from the judged drafts, in draft order. Returns (accepted, stats, duplicate_of).

    The chunk cap is the real constraint: no more than per_chunk_cap questions may rest on
    the same premise chunk. A page cap alone cannot do this job, because C_post pages are
    wildly uneven (712 chunks over 170 pages, median 2 per page, 64 for "Abandoned Camp"),
    so it both starves rich pages and still allowed 7 questions from one chunk. The page cap
    stays as a backstop: with chunks alone, one 64-chunk page took 54 of 300 questions.

    Then, for a candidate that fits the caps (only accepted candidates use up cap slots): the hand-exclusion list;
    for unanswerable candidates, the sub-page rule, the factcheck (checks) and the full-page screen (screen), each
    of which must have a verdict for every candidate it sees; and duplicate facts, compared with every fact
    already accepted in the half (the first accepted copy stays)."""
    passed = [r for r in recs if r["outcome"] == "judged_ok"]
    outcomes: Counter = Counter(r["outcome"] for r in recs if r["outcome"] != "judged_ok")
    accepted: List[Dict[str, Any]] = []
    per_page: Counter = Counter()
    per_chunk: Counter = Counter()
    dup_of: Dict[str, str] = {}

    for rec in passed:
        q = rec["question"]
        if len(accepted) >= target:
            outcomes["over_target"] += 1
            continue
        if per_chunk[rec["chunk_id"]] >= per_chunk_cap:
            outcomes["chunk_cap"] += 1
            continue
        if per_page[rec["page"]] >= per_page_cap:
            outcomes["page_cap"] += 1
            continue
        if q in excluded:
            outcomes["excluded_by_hand"] += 1
            continue
        if mode == "unanswerable":
            if is_subpage(rec["page"]):
                outcomes["post_subpage"] += 1
                continue
            if checks is not None:
                if q not in checks:
                    raise SystemExit(f"no factcheck for a candidate; run --stage factcheck first: {q[:80]}")
                if not checks[q]["faithful"]:
                    outcomes["fact_not_supported_by_chunk"] += 1
                    continue
            if screen is not None:
                if q not in screen:
                    raise SystemExit(f"no screen verdict for a candidate; run --stage screen first: {q[:80]}")
                if screen[q]["cpre_states_fact"] is not False:      # YES, or no readable verdict
                    outcomes["screen_cpre_states_fact"] += 1
                    continue
        if judge is not None:
            f = rec["fact"]
            similar = [a for a in accepted if float(emb[f] @ emb[a["fact"]]) >= SIMILAR_COSINE
                       or jaccard(f, a["fact"]) >= SIMILAR_JACCARD]
            with ThreadPoolExecutor(workers) as ex:
                verdicts = list(ex.map(lambda a: judge(f, a["fact"]), similar))
            same = [a for a, v in zip(similar, verdicts) if v == "SAME"]
            if same:
                outcomes["duplicate_fact"] += 1
                dup_of[q] = same[0]["question"]
                continue
        per_page[rec["page"]] += 1
        per_chunk[rec["chunk_id"]] += 1
        accepted.append(dict(rec, outcome="accepted"))
    stats = {"stage": "assemble", "mode": mode, "candidates_passed": len(passed),
             "accepted": len(accepted), "target": target, "outcomes": dict(outcomes),
             "pages_used": len(per_page), "max_questions_per_page": max(per_page.values(), default=0),
             "chunks_used": len(per_chunk), "max_questions_per_chunk": max(per_chunk.values(), default=0)}
    return accepted, stats, dup_of


def load_exclusions(path: str) -> frozenset:
    """Questions excluded by hand: a JSON list of {"question": ...} objects, or a CSV with a question column.
    Matched on question text, so they hold across re-assemblies."""
    if not path:
        return frozenset()
    if path.endswith(".json"):
        return frozenset(e["question"] for e in json.load(open(path)))
    return frozenset(pd.read_csv(path)["question"])


def question_rows(half: str, accepted: List[Dict[str, Any]], collection) -> pd.DataFrame:
    """One row per question. target_chunk keeps the target chunk's text, so the link from a question to its source
    survives a re-index (which renumbers chunk ids); the page title is in question_titles."""
    chunks = chunk_texts(collection, [r["chunk_id"] for r in accepted])
    wrong = [r["chunk_id"] for r in accepted if chunks[r["chunk_id"]][1] != r["page"]]
    if wrong:
        raise SystemExit(f"{len(wrong)} target chunk ids now point at another page (e.g. {wrong[:3]}): "
                         f"the index was rebuilt after the drafts were written")
    return pd.DataFrame([{
        "question": r["question"],
        "source": "latest" if half == "unanswerable" else "Pre-cutoff",
        "scenario_chunks": str(r["scenario_chunks"]), "scenario_titles": str(r["scenario_titles"]),
        "question_chunks": str([r["chunk_id"]]), "question_titles": str([r["page"]]),
        "target_fact": r["fact"], "target_chunk": chunks[r["chunk_id"]][0],
    } for r in accepted], columns=["question", "source", "scenario_chunks", "scenario_titles", "question_chunks",
                                   "question_titles", "target_fact", "target_chunk"])


# Drafts per half: about 69% of answerable drafts and 26% of unanswerable ones survive verification and the caps.
HALVES = {"answerable": {"target": 700, "drafts": 1300}, "unanswerable": {"target": 300, "drafts": 1300}}


def paths(out_dir, half, round_no):
    return (os.path.join(out_dir, f"{half}_drafts_r{round_no}.jsonl"),
            os.path.join(out_dir, f"{half}_judged_r{round_no}.jsonl"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["generate", "judge", "factcheck", "screen", "assemble", "all"], default="all")
    ap.add_argument("--out-dir", default="build")
    ap.add_argument("--out", default="questions.csv",
                    help="the benchmark; the reserve set goes to <out>_plus_reserve.csv beside it")
    ap.add_argument("--generator-url", default=LLM_URL)
    ap.add_argument("--generator-model", default="qwen3.5-122b-a10b")
    ap.add_argument("--verifier-url", default=LLM_URL, help="verification, factcheck and duplicate judge")
    ap.add_argument("--verifier-model", default="gemma-4-31b-it")
    ap.add_argument("--screen-url", default=LLM_URL, help="full-page screen: a model from another family")
    ap.add_argument("--screen-model", default="qwen3.8-27b")
    ap.add_argument("--answerable-target", type=int, default=HALVES["answerable"]["target"])
    ap.add_argument("--unanswerable-target", type=int, default=HALVES["unanswerable"]["target"])
    ap.add_argument("--answerable-reserve", type=int, default=40, help="extra questions kept beyond the target")
    ap.add_argument("--unanswerable-reserve", type=int, default=100)
    ap.add_argument("--answerable-drafts", type=int, default=HALVES["answerable"]["drafts"])
    ap.add_argument("--unanswerable-drafts", type=int, default=HALVES["unanswerable"]["drafts"])
    ap.add_argument("--per-chunk-cap", type=int, default=2)
    ap.add_argument("--per-page-cap", type=int, default=10)
    ap.add_argument("--exclusions", default="", help="optional questions excluded by hand (.json or .csv)")
    ap.add_argument("--cached-only", action="store_true",
                    help="assemble from cached duplicate verdicts only (no model calls; a missing pair is an error)")
    ap.add_argument("--round", type=int, default=1, help="top-up round: fresh seeds, same stages")
    ap.add_argument("-w", "--workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    targets = {"answerable": args.answerable_target, "unanswerable": args.unanswerable_target}
    reserves = {"answerable": args.answerable_reserve, "unanswerable": args.unanswerable_reserve}
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

    if args.stage in ("factcheck", "all"):
        factcheck_candidates(args.out_dir, make_endpoint(args.verifier_url, args.verifier_model), CPreContext(ef),
                             workers=args.workers)

    if args.stage in ("screen", "all"):
        pages = json.load(open(TRAIN_JSON_FILE, encoding="utf-8"))
        screen_candidates(args.out_dir, make_endpoint(args.screen_url, args.screen_model),
                          build_retriever("hybrid", embedding_function=ef), pages, workers=args.workers)

    if args.stage in ("assemble", "all"):
        excluded = load_exclusions(args.exclusions)
        checks = _cached(os.path.join(args.out_dir, "fact_checks.jsonl"))
        screen = _cached(os.path.join(args.out_dir, "unanswerable_screen.jsonl"))
        judge = SameFactJudge(None if args.cached_only else make_endpoint(args.verifier_url, args.verifier_model),
                              os.path.join(args.out_dir, "dedup_pairs.jsonl"))
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        colls = {"answerable": client.get_collection(TRAIN_COLLECTION_NAME),
                 "unanswerable": client.get_collection(TEST_COLLECTION_NAME)}
        print(f"exclusions by hand: {len(excluded)}; factchecks: {len(checks)}; screen verdicts: {len(screen)}",
              flush=True)
        frames = {"bench": [], "reserve": []}
        report = {"stats": {}, "duplicate_of": {}}
        for half in ("answerable", "unanswerable"):
            recs = merge_judged(args.out_dir, half)
            if not recs:
                continue
            facts = sorted({r["fact"] for r in recs if r["outcome"] == "judged_ok"})
            emb = embed_facts(facts, ef, os.path.join(args.out_dir, "fact_embeddings.npz"))
            for name, target in (("bench", targets[half]), ("reserve", targets[half] + reserves[half])):
                accepted, st, dups = assemble(
                    half, recs, target, per_page_cap=args.per_page_cap, per_chunk_cap=args.per_chunk_cap,
                    excluded=excluded, checks=checks if half == "unanswerable" else None,
                    screen=screen if half == "unanswerable" else None, judge=judge, emb=emb, workers=16)
                print(f"[{half}] assemble {name}: {json.dumps(st)}", flush=True)
                report["stats"][f"{half}_{name}"] = st
                report["duplicate_of"].update(dups)
                frames[name].append(question_rows(half, accepted, colls[half]))
            if report["stats"][f"{half}_bench"]["accepted"] < targets[half]:
                print(f"WARNING: {half} is {targets[half] - report['stats'][f'{half}_bench']['accepted']} short; "
                      f"run again with --round {args.round + 1} and more drafts", flush=True)
        if frames["bench"]:
            reserve_out = os.path.splitext(args.out)[0] + "_plus_reserve.csv"
            for name, path in (("bench", args.out), ("reserve", reserve_out)):
                out = pd.concat(frames[name], ignore_index=True)
                out.to_csv(path, index=False)
                print(f"wrote {len(out)} questions -> {path}  (source: Pre-cutoff = answerable, latest = unanswerable)")
            json.dump(report, open(os.path.join(args.out_dir, "assemble_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
