#!/usr/bin/env python
"""
4_reference_answers.py
======================
Reference answers for Answer Accuracy, written by an oracle model (DeepSeek V4.1 Flash through OpenRouter in the
paper) that is given the full wiki pages each question was written from plus the scenario's chunks, and no
search tool. Answerable questions are answered as the game stood at the cutoff (from the pinned C_pre
revisions); unanswerable ones for the current game (from C_post), for the leakage audit.

    python 4_reference_answers.py -i questions.csv -o references_answerable.csv
    python 4_reference_answers.py -i questions.csv -o references_unanswerable.csv --unanswerable-only
"""
import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import chromadb
import pandas as pd
from tqdm import tqdm

from config import (CHROMA_DB_PATH, OPENROUTER_BASE_URL, TEST_COLLECTION_NAME, TEST_JSON_FILE,
                    TRAIN_COLLECTION_NAME, TRAIN_JSON_FILE, make_client)
from prompts import REFERENCE_PROMPT, REFERENCE_PROMPT_UNANSWERABLE


OUTPUT_COLUMNS = [
    "question", "source", "answer", "premise_supported", "evidence", "context_mode", "context_chars",
    "thinking", "answered", "rag_attempts_needed", "refusal_eval",
    "model", "provider", "cost", "search_cost", "prompt_tokens", "completion_tokens",
    "reasoning_tokens", "citations", "seconds", "generation_id", "error",
]


STRUCTURED_SUFFIX = (
    "\n\nReply with a JSON object and nothing else:\n"
    '{"answer": "<the reference answer>", '
    '"premise_supported": "yes" | "partly" | "no", '
    '"evidence": "<a short quote from the context that carries the answer, or the empty string>"}\n'
    "premise_supported records whether the context actually supports what the question takes for granted. "
    "Answer the question in `answer` regardless."
)


REFUSAL_PHRASES = [
    "does not contain the answer", "not present in the given context", "i don't know", "i do not know",
    "no information", "unable to answer", "context does not provide", "does not contain information",
    "doesn't contain information", "doesn't include information", "does not include information",
    "provided context does not", "provided context doesn't", "context doesn't contain",
    "context does not contain",
]


def _is_refusal_heuristic(answer_text: str) -> bool:
    text = re.sub(r"<think>.*?(</think>|$)", "", answer_text or "", flags=re.DOTALL).strip().lower()
    return any(p in text for p in REFUSAL_PHRASES)


def load_collections():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_collection(TRAIN_COLLECTION_NAME), client.get_collection(TEST_COLLECTION_NAME)


def load_pages():
    """
    Full page text by title, from the crawl JSONs. Each page carries the `oldid` it was
    crawled at, so a page-level context is as pinned as a chunk-level one: C_pre pages
    are the state at the cutoff and C_post pages the state after it. Nothing is fetched
    from the live wiki, which would carry edits made after the corpus was built.
    """
    out = {}
    for key, path in (("Pre-cutoff", TRAIN_JSON_FILE), ("latest", TEST_JSON_FILE)):
        out[key] = {url.rsplit("/w/", 1)[-1].replace("_", " "): page
                    for url, page in json.load(open(path, encoding="utf-8")).items()}
    return out


def _ids(row, column):
    value = row.get(column)
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        return json.loads(value.replace("'", '"'))
    except json.JSONDecodeError:
        return []


def _chunk_block(coll, ids):
    if not ids:
        return []
    res = coll.get(ids=ids, include=["documents", "metadatas"])
    return [f"Title: {m.get('title')}\nSource URL: {m.get('source_url')}\n\n {d}"
            for d, m in zip(res["documents"], res["metadatas"])]


def build_prompt(row, train_coll, test_coll, mode="premise", pages=None, structured=False) -> str:
    """
    The oracle's evidence. Three widths, because how much context a reference answer
    needs is an open question the modes let us measure:

      premise  the chunks the question was written from
      chunks   those plus the chunks the scenario was written from
      pages    the premise pages in full, plus the scenario chunks -- so the oracle can
               see a qualifier in another section, or that the premise is unsupported
    """
    coll = test_coll if row["source"] == "latest" else train_coll
    blocks = _chunk_block(coll, _ids(row, "question_chunks"))

    if mode in ("chunks", "pages"):
        blocks += _chunk_block(train_coll, _ids(row, "scenario_chunks"))
    if mode == "pages":
        # The pages the premise chunks came from, read off the chunks' own metadata and
        # de-duplicated: three chunks from two pages give those two pages once each.
        # Taken from the collection the question belongs to, because the two collections
        # reuse the same chunk ids (all 712 C_post ids also exist in C_pre).
        titles, seen = [], set()
        premise = _ids(row, "question_chunks")
        if premise:
            got = coll.get(ids=premise, include=["metadatas"])
            for m in got["metadatas"]:
                title = m.get("title")
                if title and title not in seen:
                    seen.add(title)
                    titles.append(title)
        full = []
        for title in titles:
            page = (pages or {}).get(row["source"], {}).get(title) or (pages or {}).get("Pre-cutoff", {}).get(title)
            if page:
                full.append(f"Title: {title}\nSource URL: {page.get('url')}\n"
                            f"Revision: {page.get('oldid')} ({page.get('date')})\n\n{page['content']}")
        # The full pages replace the premise chunks they contain; the scenario chunks stay.
        blocks = full + _chunk_block(train_coll, _ids(row, "scenario_chunks")) if full else blocks

    context = "\n---\n".join(blocks) or "No context retrieved."
    system = REFERENCE_PROMPT_UNANSWERABLE if row["source"] == "latest" else REFERENCE_PROMPT
    suffix = STRUCTURED_SUFFIX if structured else ""
    return f"{system}{suffix}\n\nContext:\n{context}\n\nQuestion: {row['question']}"


def _parse_structured(content: str) -> dict:
    """
    The oracle is asked for a JSON object. Models wrap it in prose or a code fence often
    enough that a strict parse would throw away good answers, so the object is located
    rather than assumed, and a reply that is plainly just an answer is kept as one.
    """
    text = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict) and obj.get("answer"):
                return {"answer": str(obj["answer"]).strip(),
                        "premise_supported": str(obj.get("premise_supported", "")).strip().lower(),
                        "evidence": str(obj.get("evidence", ""))[:600]}
        except json.JSONDecodeError:
            pass
    return {"answer": content.strip(), "premise_supported": "", "evidence": ""}


def answer_row(client, row, args, colls) -> dict:
    """One reference answer: a single chat call with the page context, retried on errors or empty replies."""
    out = {"question": row["question"], "source": row["source"], "thinking": "", "rag_attempts_needed": 0,
           "refusal_eval": "heuristic", "model": args.model, "answer": "", "answered": 0, "cost": 0.0}
    body = {"usage": {"include": True}}
    if args.provider:
        body["provider"] = {"only": args.provider.split(",")}
    prompt = build_prompt(row, *colls, mode=args.context, pages=args.pages)
    out["context_mode"], out["context_chars"] = args.context, len(prompt)
    for attempt in range(1, args.retries + 1):
        try:
            resp = client.chat.completions.create(model=args.model, messages=[{"role": "user", "content": prompt}],
                                                  extra_body=body).model_dump()
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {e}"[:300]
            time.sleep(5 * attempt)
            continue
        usage = resp.get("usage") or {}
        out["cost"] += usage.get("cost") or 0.0
        out.update(provider=resp.get("provider"), prompt_tokens=usage.get("prompt_tokens"),
                   completion_tokens=usage.get("completion_tokens"))
        content = ((resp["choices"][0].get("message") or {}).get("content") or "").strip()
        if content:
            parsed = _parse_structured(content)
            out.update({**parsed, "answered": 0 if _is_refusal_heuristic(parsed["answer"]) else 1, "error": ""})
            return out
        out["error"] = f"empty reply (finish_reason={resp['choices'][0].get('finish_reason')})"
    return out


def main():
    p = argparse.ArgumentParser(description="Write reference answers with an oracle model.")
    p.add_argument("-i", "--input", default="questions.csv")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--model", default="deepseek/deepseek-v4.1-flash")
    p.add_argument("--url", default=OPENROUTER_BASE_URL)
    p.add_argument("--provider", default="", help="comma-separated OpenRouter provider slugs")
    p.add_argument("--context", choices=["premise", "chunks", "pages"], default="pages")
    p.add_argument("--unanswerable-only", action="store_true", help="answer the post-cutoff questions instead")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--retries", type=int, default=3)
    args = p.parse_args()

    questions = pd.read_csv(args.input)
    questions = questions[questions["source"] == ("latest" if args.unanswerable_only else "Pre-cutoff")]
    order = questions["question"].tolist()
    done = {}
    if os.path.exists(args.output) and os.path.getsize(args.output) > 0:
        done = {r["question"]: r for r in pd.read_csv(args.output, keep_default_na=False).to_dict("records")}
    pending = questions[~questions["question"].isin([q for q, r in done.items() if str(r.get("answer", "")).strip()])]
    print(f"{len(questions)} questions | {len(questions) - len(pending)} already answered | {len(pending)} to run")
    if pending.empty:
        return
    client = make_client(args.url, timeout=300)
    colls = load_collections()
    args.pages = load_pages() if args.context == "pages" else None
    lock = threading.Lock()

    def save():
        rows = [done[q] for q in order if q in done]
        pd.DataFrame(rows).reindex(columns=OUTPUT_COLUMNS).to_csv(args.output, index=False)

    def task(row):
        result = answer_row(client, row, args, colls)
        with lock:
            done[row["question"]] = result
            save()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for f in tqdm(as_completed([pool.submit(task, r) for r in pending.to_dict("records")]), total=len(pending)):
            f.result()
    ok = sum(1 for q in order if q in done and str(done[q].get("answer", "")).strip())
    print(f"Saved {args.output}: {ok}/{len(questions)} answered")


if __name__ == "__main__":
    main()
