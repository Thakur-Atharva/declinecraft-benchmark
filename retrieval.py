"""
retrieval.py
============
Retrieval over the C_pre index and the iterative RAG controller.

  dense           cosine k-NN over Qwen3-Embedding-8B vectors (ChromaDB)
  bm25            Okapi BM25 (k1 = 1.5, b = 0.75) over lower-cased word tokens, stopwords removed
  hybrid          dense + BM25 (top 20 each) fused by reciprocal rank fusion, k = 60  (default)
  hybrid_rerank   hybrid, then the cross-encoder reranks the 20 highest fused chunks
  hybrid_rewrite  hybrid over the original query plus three LLM rewrites (all lists fused)
  full            hybrid_rewrite + reranking

Iterative RAG (SelfRAGController): after the first k chunks, the answering model judges whether the context
suffices; if not, its follow-up questions retrieve more chunks (sharing the original k), up to two rounds.
"""
import json
import math
import os
import re
import threading
import time
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from abc import ABC, abstractmethod

import chromadb
import requests
from chromadb.errors import NotFoundError
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI

from config import (CHROMA_DB_PATH, EMBED_MODEL, EMBED_URL, OPENAI_API_KEY, QUERY_REWRITE_BASE_URL,
                    QUERY_REWRITE_CACHE, QUERY_REWRITE_MODEL, RERANKER_BASE_URL, RERANKER_MODEL, RRF_K,
                    TRAIN_COLLECTION_NAME, get_node_key)
from prompts import REFLECTION_PROMPT_TEMPLATE, REWRITE_PROMPT


def get_embedding_function(model_name: str = EMBED_MODEL, base_url: str = EMBED_URL):
    """The query/document embedder: an OpenAI-compatible /v1/embeddings endpoint."""
    ef = OpenAIEmbeddingFunction(model_name=model_name, api_key=get_node_key(base_url) or "EMPTY", api_base=base_url)
    ef(["test"])          # fail now, not in the middle of a run
    return ef


class BaseRetriever(ABC):
    """Abstract interface for all DeclineCraft retrievers."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves top-k context chunks for a single query.
        Returns a list of dicts: [{"text": str, "metadata": dict}]
        """
        pass

    @abstractmethod
    def retrieve_batch(self, queries: List[str], k: int = 5) -> List[List[Dict[str, Any]]]:
        """
        Batches multiple queries into a single retrieval call.
        Returns a list of chunk lists corresponding to each query.
        """
        pass


class DenseRetriever(BaseRetriever):
    """Dense retriever using ChromaDB collection queries."""

    def __init__(
        self,
        db_path: str = CHROMA_DB_PATH,
        collection_name: str = TRAIN_COLLECTION_NAME,
        embedding_function: Any = None,
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.collection = None

        try:
            client = chromadb.PersistentClient(path=db_path)
            self.collection = client.get_collection(
                name=collection_name,
                embedding_function=embedding_function,
            )
        except NotFoundError:
            print(f"[DenseRetriever] Error: Collection '{collection_name}' not found at '{db_path}'.")
        except Exception as e:
            print(f"[DenseRetriever] Fatal Error connecting to ChromaDB: {e}")
            raise e

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Performs dense similarity search with exponential backoff on retry."""
        if not self.collection:
            return []

        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=k,
                    include=['documents', 'metadatas', 'distances'],
                )
                # The chunk id is kept so hybrid fusion can recognise the same chunk
                # in the dense and BM25 lists. Without it the two lists never merged
                # and a chunk found by both took two of the k slots.
                retrieved = []
                if results and results.get("documents") and results["documents"][0]:
                    for cid, doc, meta in zip(results['ids'][0], results['documents'][0], results['metadatas'][0]):
                        retrieved.append({"id": cid, "text": doc, "metadata": meta})
                return retrieved
            except Exception as e:
                if attempt == max_attempts:
                    print(f"[DenseRetriever] Failed to retrieve after {attempt} attempts: {e}")
                    return []
                time.sleep(2 ** (attempt - 1))
        return []

    def retrieve_batch(self, queries: List[str], k: int = 5) -> List[List[Dict[str, Any]]]:
        """Batches multiple queries into a single ChromaDB search call."""
        if not self.collection or not queries:
            return [[] for _ in queries]

        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                results = self.collection.query(
                    query_texts=queries,
                    n_results=k,
                    include=['documents', 'metadatas', 'distances'],
                )
                batch_items = []
                if results and results.get("documents"):
                    for ids, docs, metas in zip(results['ids'], results['documents'], results['metadatas']):
                        items = [{"id": i, "text": d, "metadata": m} for i, d, m in zip(ids, docs, metas)]
                        batch_items.append(items)
                else:
                    batch_items = [[] for _ in queries]
                return batch_items
            except Exception as e:
                if attempt == max_attempts:
                    print(f"[DenseRetriever] Failed batch retrieval: {e}")
                    return [[] for _ in queries]
                time.sleep(2 ** (attempt - 1))
        return [[] for _ in queries]


# Okapi BM25, k1 = 1.5, b = 0.75.


STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


def tokenize(text: str) -> List[str]:
    """
    Tokenizes text into lowercase alphanumeric terms.
    Preserves dotted numbers (version strings such as 1.21.6), underscores, and
    alphanumeric tokens. Filters out common stopwords.

    Full stops are not part of word tokens: a class that allowed them made the
    last word of every sentence a different term ("golem." vs "golem").
    """
    tokens = re.findall(r'\d+(?:\.\d+)+|[a-z0-9_]+', text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


class BM25Index:
    """
    In-memory Okapi BM25 Index for lexical search over text chunks.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size: int = 0
        self.avgdl: float = 0.0
        self.doc_ids: List[str] = []
        self.doc_lengths: Dict[str, int] = {}
        self.doc_token_counts: Dict[str, Counter] = {}
        self.df: Dict[str, int] = defaultdict(int)
        self.idf: Dict[str, float] = {}
        self.documents: Dict[str, Dict[str, Any]] = {}

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Indexes a list of documents.
        Each document dict must have:
          - 'id': str
          - 'text': str
          - 'metadata': dict (optional)
        """
        for doc in documents:
            doc_id = doc['id']
            text = doc.get('text', '')
            tokens = tokenize(text)

            self.doc_ids.append(doc_id)
            self.doc_lengths[doc_id] = len(tokens)
            counts = Counter(tokens)
            self.doc_token_counts[doc_id] = counts
            self.documents[doc_id] = doc

            for term in counts.keys():
                self.df[term] += 1

        self.corpus_size = len(self.doc_ids)
        if self.corpus_size > 0:
            total_tokens = sum(self.doc_lengths.values())
            self.avgdl = total_tokens / self.corpus_size

        # Precompute IDF for all terms
        self._compute_idf()

    def _compute_idf(self) -> None:
        """Computes Robertson-Spärck Jones IDF for all terms in the vocabulary."""
        n = self.corpus_size
        for term, freq in self.df.items():
            # Robertson-Spärck Jones formula with smoothing
            self.idf[term] = math.log(((n - freq + 0.5) / (freq + 0.5)) + 1.0)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """
        Scores all documents against the query and returns the top_k results.
        Returns:
          List of (document_dict, bm25_score) tuples, sorted descending.
        """
        query_tokens = tokenize(query)
        if not query_tokens or self.corpus_size == 0:
            return []

        scores: Dict[str, float] = defaultdict(float)

        for token in query_tokens:
            if token not in self.idf:
                continue

            idf = self.idf[token]

            for doc_id in self.doc_ids:
                tf = self.doc_token_counts[doc_id].get(token, 0)
                if tf == 0:
                    continue

                dl = self.doc_lengths[doc_id]
                # Okapi BM25 numerator and denominator
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl)) if self.avgdl > 0 else 1.0
                score_term = idf * (tf * (self.k1 + 1.0)) / denom
                scores[doc_id] += score_term

        # Sort doc_ids by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for doc_id, score in ranked:
            doc_data = self.documents[doc_id].copy()
            doc_data['bm25_score'] = score
            results.append((doc_data, score))

        return results


class RerankerUnavailable(RuntimeError):
    """Raised when no cross-encoder endpoint could score a batch."""


class CrossEncoderReranker:
    """Reranks candidate chunks against a query using a served cross-encoder."""

    def __init__(
        self,
        model: str = RERANKER_MODEL,
        base_url: str = RERANKER_BASE_URL,
        api_key: str = OPENAI_API_KEY,
        timeout: float = 120.0,
        strict: bool = True,
    ):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.strict = strict
        # Endpoint shape is probed once, then reused for the run.
        self._endpoint: Optional[Tuple[str, str]] = None

    # --- endpoint plumbing -------------------------------------------------

    def _candidate_endpoints(self) -> List[Tuple[str, str]]:
        """(style, url) pairs to try, most likely first."""
        root = self.base_url[:-3].rstrip('/') if self.base_url.endswith('/v1') else self.base_url
        return [
            ("rerank", f"{self.base_url}/rerank"),
            ("rerank", f"{root}/rerank"),
            ("score", f"{self.base_url}/score"),
            ("score", f"{root}/score"),
        ]

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _payload(self, style: str, query: str, documents: List[str]) -> Dict[str, Any]:
        if style == "rerank":
            return {
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": len(documents),
            }
        return {"model": self.model, "text_1": query, "text_2": documents}

    @staticmethod
    def _parse(style: str, body: Dict[str, Any], n: int) -> Optional[List[float]]:
        """Extracts a score per input document, preserving input order."""
        scores = [None] * n
        rows = body.get("results") if style == "rerank" else body.get("data")
        if not isinstance(rows, list):
            return None
        for row in rows:
            if not isinstance(row, dict):
                return None
            idx = row.get("index")
            val = row.get("relevance_score", row.get("score"))
            if idx is None or val is None or not (0 <= idx < n):
                continue
            scores[idx] = float(val)
        if any(s is None for s in scores):
            return None
        return scores  # type: ignore[return-value]

    # --- public API --------------------------------------------------------

    def score(self, query: str, documents: List[str]) -> List[float]:
        """Returns one relevance score per document, in input order."""
        if not documents:
            return []

        attempts = [self._endpoint] if self._endpoint else self._candidate_endpoints()
        errors: List[str] = []

        for style, url in attempts:
            try:
                resp = requests.post(
                    url,
                    headers=self._headers(),
                    data=json.dumps(self._payload(style, query, documents)),
                    timeout=self.timeout,
                )
            except Exception as e:
                errors.append(f"{url}: {type(e).__name__}: {e}")
                continue

            if resp.status_code != 200:
                errors.append(f"{url}: HTTP {resp.status_code} {resp.text[:160]}")
                continue

            try:
                parsed = self._parse(style, resp.json(), len(documents))
            except Exception as e:
                errors.append(f"{url}: unparseable response: {e}")
                continue

            if parsed is None:
                errors.append(f"{url}: response missing scores for some documents")
                continue

            self._endpoint = (style, url)
            return parsed

        self._endpoint = None
        msg = (
            f"Cross-encoder reranking failed for model '{self.model}'. Tried:\n  "
            + "\n  ".join(errors)
        )
        if self.strict:
            raise RerankerUnavailable(msg)
        print(f"[CrossEncoderReranker] WARNING (non-strict): {msg}")
        return [0.0] * len(documents)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Scores candidates against the query and returns the top_k by score."""
        if not candidates:
            return []

        texts = [c.get("text", "") for c in candidates]
        scores = self.score(query, texts)

        if len(scores) != len(candidates) or not any(scores):
            # Non-strict fallback path only; strict mode raised above.
            return candidates[:top_k]

        ranked = sorted(zip(candidates, scores), key=lambda p: p[1], reverse=True)
        out = []
        for doc, s in ranked[:top_k]:
            item = dict(doc)
            item["rerank_score"] = s
            out.append(item)
        return out


PREFLIGHT_QUERY = "My copper golem stopped moving after a while. How do I get it working again?"


class RewriteUnavailable(RuntimeError):
    """Raised when the rewriting model returned no usable rewrites."""


class QueryRewriter:
    """Expands one query into several focused retrieval queries via an LLM."""

    def __init__(
        self,
        model: str = QUERY_REWRITE_MODEL,
        base_url: str = QUERY_REWRITE_BASE_URL,
        api_key: Optional[str] = None,
        n_rewrites: int = 3,
        include_original: bool = True,
        timeout: float = 120.0,
        max_retries: int = 3,
        cache_path: Optional[str] = QUERY_REWRITE_CACHE,
    ):
        if not model or not base_url:
            raise ValueError(
                "QueryRewriter needs an explicit model and server: set QUERY_REWRITE_MODEL and "
                "QUERY_REWRITE_BASE_URL, or pass --rewrite-model and --rewrite-url."
            )
        self.model = model
        self.base_url = base_url
        self.n_rewrites = n_rewrites
        self.include_original = include_original
        self.max_retries = max_retries
        self.cache_path = cache_path
        self.client = OpenAI(base_url=base_url, api_key=api_key or get_node_key(base_url) or "EMPTY",
                             timeout=timeout, max_retries=0)
        self._lock = threading.Lock()
        # query -> rewrites (without the original query)
        self._cache: Dict[str, List[str]] = self._load_cache()

    # --- disk cache ------------------------------------------------------------

    def _read_file(self) -> dict:
        if not self.cache_path or not os.path.exists(self.cache_path):
            return {}
        with open(self.cache_path, encoding="utf-8") as f:
            return json.load(f)

    def _load_cache(self) -> Dict[str, List[str]]:
        # Keyed by model, so rewrites from a different rewriter are never reused.
        return {q: list(v) for q, v in self._read_file().get(self.model, {}).items()}

    def _save_cache(self) -> None:
        """Writes the cache atomically. Called with self._lock held."""
        if not self.cache_path:
            return
        data = self._read_file()
        data[self.model] = self._cache
        os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)), exist_ok=True)
        tmp = f"{self.cache_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.cache_path)

    # --- requests --------------------------------------------------------------

    @staticmethod
    def _clean(line: str) -> str:
        line = re.sub(r"^\s*[-*•]\s*", "", line)
        line = re.sub(r"^\s*\d+[.):]\s*", "", line)
        return line.strip().strip('"').strip()

    def _request(self, query: str) -> List[str]:
        extra_body = None
        if "openrouter" not in self.base_url.lower():
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": REWRITE_PROMPT.format(n=self.n_rewrites, query=query),
            }],
            temperature=0.0,
            max_tokens=256,
            extra_body=extra_body,
        )
        choice = resp.choices[0]
        raw = choice.message.content or ""
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL)
        variants = []
        for line in raw.splitlines():
            cleaned = self._clean(line)
            if cleaned and len(cleaned.split()) <= 25:
                variants.append(cleaned)
        if not variants:
            raise ValueError(f"no rewrites in reply (finish_reason={choice.finish_reason}): {raw[:120]!r}")
        return variants[: self.n_rewrites]

    def preflight(self) -> None:
        """Checks once, before a run starts, that rewrites are available."""
        if self._cache:
            print(f"[QueryRewriter] Using {len(self._cache)} cached rewrites from '{self.model}' "
                  f"({self.cache_path}); uncached questions go to {self.base_url}")
            return
        variants = self.rewrite(PREFLIGHT_QUERY)
        print(f"[QueryRewriter] '{self.model}' at {self.base_url} OK "
              f"({len(variants) - int(self.include_original)} rewrites)")

    def rewrite(self, query: str) -> List[str]:
        """
        Returns the list of queries to retrieve with: the original query (unless
        include_original is False) followed by the rewrites. Raises
        RewriteUnavailable if every attempt fails; failures are not cached.
        """
        with self._lock:
            variants = self._cache.get(query)

        if variants is None:
            last_error = None
            for attempt in range(self.max_retries):
                try:
                    variants = self._request(query)
                    break
                except Exception as e:
                    last_error = e
                    time.sleep(2.0 ** attempt)
            else:
                raise RewriteUnavailable(
                    f"Query rewriting failed after {self.max_retries} attempts with '{self.model}' "
                    f"at {self.base_url}: {type(last_error).__name__}: {last_error}"
                )
            with self._lock:
                self._cache[query] = list(variants)
                self._save_cache()

        out: List[str] = []
        if self.include_original:
            out.append(query)
        for v in variants:
            if v not in out:
                out.append(v)
        return out


def _doc_key(doc: Dict[str, Any]) -> str:
    """Stable identity for a chunk across dense and sparse result lists.

    Both lists carry the ChromaDB chunk id ('doc_N'); the text fallback is only for
    retrievers that return no id, where it cannot match an id-keyed chunk.
    """
    return doc.get("id") or (doc.get("metadata") or {}).get("chunk_id") or doc.get("text", "")[:200]


class HybridRetriever(BaseRetriever):
    """
    Production Hybrid Retriever combining Dense vector search and Sparse BM25
    via Reciprocal Rank Fusion (RRF), with optional query rewriting and reranking.
    """

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        bm25_index: Optional[BM25Index] = None,
        rrf_k: int = 60,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        reranker: Optional[CrossEncoderReranker] = None,
        query_rewriter: Optional[QueryRewriter] = None,
        use_dense: bool = True,
        use_sparse: bool = True,
    ):
        self.dense = dense_retriever or DenseRetriever()
        self.bm25 = bm25_index
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.reranker = reranker
        self.query_rewriter = query_rewriter
        self.use_dense = use_dense
        self.use_sparse = use_sparse
        self._init_lock = threading.Lock()

    def _ensure_bm25_initialized(self) -> None:
        """Initializes the BM25 index from the ChromaDB collection if not yet loaded."""
        if self.bm25 is not None or not self.use_sparse:
            return

        with self._init_lock:
            if self.bm25 is not None:
                return

            # Built locally and published only when complete. Assigning the empty
            # index first let other threads see a BM25 index with no documents and
            # skip sparse search: in bm25 mode every question that arrived during
            # the build got no context, and in hybrid mode they got dense only.
            index = BM25Index()
            if self.dense and self.dense.collection:
                try:
                    all_docs = self.dense.collection.get(include=['documents', 'metadatas'])
                    doc_items = []
                    if all_docs and all_docs.get('ids'):
                        for doc_id, text, meta in zip(all_docs['ids'], all_docs['documents'], all_docs['metadatas']):
                            doc_items.append({
                                "id": doc_id,
                                "text": text,
                                "metadata": meta or {},
                            })
                    index.add_documents(doc_items)
                    print(f"[HybridRetriever] Initialized BM25 index over {index.corpus_size} chunks.")
                except Exception as e:
                    print(f"[HybridRetriever] Warning: Could not initialize BM25 from collection: {e}")
            if not self.use_dense and index.corpus_size == 0:
                raise RuntimeError("BM25-only retrieval has an empty index; refusing to answer without context.")
            self.bm25 = index

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Executes the configured retrieval pipeline:
        1. Optional query rewriting into several focused sub-queries
        2. Dense and/or BM25 retrieval for each query variant
        3. Reciprocal Rank Fusion across all ranked lists
        4. Optional cross-encoder reranking of the fused pool
        """
        self._ensure_bm25_initialized()

        candidate_pool = max(20, k * 3)
        queries = self.query_rewriter.rewrite(query) if self.query_rewriter else [query]

        ranked_lists: List[Tuple[List[Dict[str, Any]], float]] = []
        for q in queries:
            if self.use_dense:
                ranked_lists.append((self.dense.retrieve(q, k=candidate_pool), self.dense_weight))
            if self.use_sparse and self.bm25 and self.bm25.corpus_size > 0:
                sparse = [doc for doc, _ in self.bm25.search(q, top_k=candidate_pool)]
                ranked_lists.append((sparse, self.sparse_weight))

        # Reranking needs a deeper pool than the final k.
        fuse_depth = max(k * 4, 20) if self.reranker else k
        fused = self._reciprocal_rank_fusion(ranked_lists, top_k=fuse_depth)

        if self.reranker:
            return self.reranker.rerank(query, fused, top_k=k)

        return fused[:k]

    def _reciprocal_rank_fusion(
        self,
        ranked_lists: List[Tuple[List[Dict[str, Any]], float]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Merges any number of weighted ranked lists using Reciprocal Rank Fusion:
          RRF_Score(d) = sum_over_lists( weight_list / (rrf_k + rank_in_list(d)) )
        """
        rrf_scores: Dict[str, float] = {}
        doc_store: Dict[str, Dict[str, Any]] = {}

        for docs, weight in ranked_lists:
            for rank, doc in enumerate(docs, start=1):
                key = _doc_key(doc)
                if key not in doc_store:
                    doc_store[key] = doc
                rrf_scores[key] = rrf_scores.get(key, 0.0) + (weight / (self.rrf_k + rank))

        sorted_keys = sorted(rrf_scores, key=lambda kk: rrf_scores[kk], reverse=True)[:top_k]
        results = []
        for key in sorted_keys:
            item = dict(doc_store[key])
            item['rrf_score'] = rrf_scores[key]
            results.append(item)
        return results

    def retrieve_batch(self, queries: List[str], k: int = 5) -> List[List[Dict[str, Any]]]:
        """Performs hybrid retrieval for a batch of queries."""
        return [self.retrieve(q, k=k) for q in queries]


RETRIEVAL_MODES = ["dense", "bm25", "hybrid", "hybrid_rerank", "hybrid_rewrite", "full"]


def build_retriever(mode: str = "hybrid", embedding_function: Any = None, db_path: Optional[str] = None,
                    collection_name: Optional[str] = None, preflight: bool = True) -> BaseRetriever:
    """Builds the retriever for a pipeline name. The reranker and the rewriter are checked with one request each
    before any question is asked."""
    mode = (mode or "hybrid").lower().strip()
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"Unknown retrieval mode '{mode}'. Valid modes: {RETRIEVAL_MODES}")
    dense_kwargs = {"embedding_function": embedding_function}
    if db_path:
        dense_kwargs["db_path"] = db_path
    if collection_name:
        dense_kwargs["collection_name"] = collection_name
    dense = DenseRetriever(**dense_kwargs)
    if mode == "dense":
        return dense

    reranker = None
    if mode in ("hybrid_rerank", "full"):
        reranker = CrossEncoderReranker()
        if preflight:
            reranker.score("How do you tame a wolf?", ["Wolves can be tamed with bones."])
    rewriter = None
    if mode in ("hybrid_rewrite", "full"):
        rewriter = QueryRewriter()
        if preflight:
            rewriter.preflight()

    retriever = HybridRetriever(dense_retriever=dense, rrf_k=RRF_K, reranker=reranker, query_rewriter=rewriter,
                                use_dense=(mode != "bm25"), use_sparse=True)
    retriever._ensure_bm25_initialized()       # build BM25 before worker threads start
    return retriever


# --- Iterative RAG ----------------------------------------------------------------------------------


OPENROUTER_THINKING_OPT_IN = ("google/gemini-2.5-flash",)


OPENROUTER_PROVIDERS = {
    "openai/gpt-oss-120b": {"order": ["deepinfra", "akashml"], "quantizations": ["bf16"], "allow_fallbacks": False},
}


def openrouter_reasoning(base_url: Optional[str], model: str, no_think: bool) -> Dict[str, Any]:
    """Per-model OpenRouter options: a provider pin for open-weight models, and the unified reasoning switch. OpenRouter ignores the "/no_think" prompt suffix
    for most providers, so a -nothink build turns reasoning off explicitly; opt-in models turn it
    on explicitly. Other models keep the provider default. Empty for non-OpenRouter endpoints."""
    if "openrouter" not in (base_url or "").lower():
        return {}
    out = {"provider": OPENROUTER_PROVIDERS[model]} if model in OPENROUTER_PROVIDERS else {}
    if no_think:
        out["reasoning"] = {"enabled": False}
    elif model in OPENROUTER_THINKING_OPT_IN:
        out["reasoning"] = {"enabled": True}
    return out


_PRINT_FIRST_FAILURES = 5


def parse_assessment(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Extracts the reflection JSON from a model reply.
    Returns (assessment, None) on success, or (None, failure_reason).
    The JSON object is taken from the first '{' to the last '}', so prose or a
    code fence around it does not cause a parse failure.
    """
    raw = raw or ""
    if "</think>" in raw:
        raw = raw.split("</think>")[-1]
    raw = re.sub(r'<think>.*', '', raw, flags=re.DOTALL).strip()

    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None, "no_json"
    try:
        result = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None, "bad_json"
    if not isinstance(result, dict) or "assessment" not in result:
        return None, "bad_schema"

    verdict = str(result["assessment"]).strip().lower()
    if verdict.startswith("insufficient"):
        result["assessment"] = "Insufficient"
    elif verdict.startswith("sufficient"):
        result["assessment"] = "Sufficient"
    else:
        return None, "bad_schema"

    follow_ups = result.get("follow_up_questions") or []
    if not isinstance(follow_ups, list):
        follow_ups = [follow_ups]
    result["follow_up_questions"] = [str(q).strip() for q in follow_ups if str(q).strip()][:5]
    return result, None


class SelfRAGController:
    """Manages iterative retrieval, reflection, and context augmentation."""

    def __init__(
        self,
        retriever: BaseRetriever,
        client: OpenAI,
        model_name: str,
        max_attempts: int = 2,
        no_think: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 16000,
        request_retries: int = 3,
    ):
        self.retriever = retriever
        self.client = client
        self.model_name = model_name
        self.max_attempts = max_attempts
        self.no_think = no_think
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_retries = request_retries
        self._stats_lock = threading.Lock()
        self.reflection_calls = 0
        self.reflection_failures: Counter = Counter()

    def _record_failure(self, reason: str, detail: str) -> None:
        with self._stats_lock:
            self.reflection_failures[reason] += 1
            n = sum(self.reflection_failures.values())
        if n <= _PRINT_FIRST_FAILURES:
            print(f"[SelfRAG] Reflection failed ({reason}): {detail[:200]}")

    def reflection_summary(self) -> str:
        """One line for the end-of-run report."""
        with self._stats_lock:
            failed = sum(self.reflection_failures.values())
            reasons = ", ".join(f"{r} {c}" for r, c in self.reflection_failures.most_common())
            calls = self.reflection_calls
        line = f"[SelfRAG] Reflection calls: {calls}, failed: {failed}"
        if calls:
            line += f" ({100.0 * failed / calls:.1f}%)"
        return line + (f" -- {reasons}" if reasons else "")

    def assess_sufficiency(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Queries the LLM in JSON format to evaluate context sufficiency.
        A failed reflection returns {"assessment": None, "failure": reason}.
        """
        if not context_chunks:
            return {"assessment": "Insufficient", "missing_info": "No context found.", "follow_up_questions": []}

        context_text = "\n---\n".join([
            f"Title: {c['metadata'].get('title', 'Unknown')}\n"
            f"Source URL: {c['metadata'].get('source_url', '')}\n"
            f"\n{c['text']}"
            for c in context_chunks
        ])

        prompt = REFLECTION_PROMPT_TEMPLATE.format(context=context_text, query=query)
        if self.no_think:
            prompt = f"{prompt} /no_think"

        extra_body = {}
        if "gemini-3" in self.model_name.lower():
            extra_body["reasoning"] = {"effort": "low"}
        extra_body.update(openrouter_reasoning(str(getattr(self.client, "base_url", "")), self.model_name, self.no_think))
        if "openrouter" not in str(getattr(self.client, "base_url", "")).lower():
            # Matches evaluator.generate_answer: thinking is set explicitly in both
            # directions, because the "/no_think" suffix is not honoured by every
            # model and the default differs (Gemma 4 thinks only when asked).
            extra_body["chat_template_kwargs"] = {"enable_thinking": not self.no_think}

        with self._stats_lock:
            self.reflection_calls += 1

        # Only request errors are retried: a reply that does not parse would most
        # likely fail the same way again.
        response, last_error = None, None
        for attempt in range(self.request_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stop=["<|im_end|>", "<|im_start|>", "<|eot_id|>", "<end_of_turn>", "</s>"],
                    extra_body=extra_body if extra_body else None,
                )
                break
            except Exception as e:
                last_error = e
                time.sleep(2.0 ** attempt)

        if response is None:
            self._record_failure("request_error", f"{type(last_error).__name__}: {last_error}")
            return {"assessment": None, "failure": "request_error"}

        choice = response.choices[0]
        raw = choice.message.content or ""
        result, reason = parse_assessment(raw)
        if result is None:
            if not raw.strip() and choice.finish_reason == "length":
                reason = "budget_exhausted"
            self._record_failure(reason, raw.strip() or f"empty reply, finish_reason={choice.finish_reason}")
            return {"assessment": None, "failure": reason}
        return result

    def execute_retrieval(
        self,
        query: str,
        k: int = 5,
        callback: Optional[Callable[[int], None]] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Runs the iterative retrieval loop.
        Returns: (final_context_chunks, attempts_consumed), where an attempt is one
        reflection that asked for more context. A failed reflection ends the loop
        and is not counted.
        """
        attempts_needed = 0
        if callback:
            callback(attempts_needed)

        initial_context = self.retriever.retrieve(query, k=k)
        final_context = initial_context
        context_dict = {_doc_key(c): c for c in initial_context}

        if self.max_attempts == 0:
            return final_context, 0

        while attempts_needed < self.max_attempts:
            assessment = self.assess_sufficiency(query, final_context)
            if assessment.get("assessment") != "Insufficient":
                break  # Sufficient, or the reflection failed

            attempts_needed += 1
            if callback:
                callback(attempts_needed)

            follow_ups = assessment.get("follow_up_questions", [])
            if not follow_ups:
                break

            # The follow-ups share the original k, so each attempt adds at most about k new chunks.
            k_per_q = max(1, k // len(follow_ups))
            followup_batches = self.retriever.retrieve_batch(follow_ups, k=k_per_q)
            for batch in followup_batches:
                for chunk in batch:
                    context_dict.setdefault(_doc_key(chunk), chunk)

            final_context = list(context_dict.values())

        return final_context, attempts_needed


if __name__ == "__main__":
    # Quick check: python retrieval.py "How do I tame a horse?" [mode]
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "How do I tame a horse?"
    r = build_retriever(sys.argv[2] if len(sys.argv) > 2 else "hybrid", embedding_function=get_embedding_function())
    for d in r.retrieve(query, k=5):
        print(d.get("id"), "|", (d.get("metadata") or {}).get("title"), "|", d["text"][:100].replace("\n", " "))
