# DeclineCraft: code

Code to rebuild the DeclineCraft benchmark and to evaluate a model on it, accompanying the paper
[DeclineCraft: A RAG Abstention Benchmark for Jointly Measuring Combined Retrieval and Parametric Knowledge](https://arxiv.org/abs/2610.28340).
If you use the code or data, please [cite the paper](#citation). The scripts are numbered in pipeline order; each
one resumes if rerun.

| Step | Script | What it does |
|---|---|---|
| 1 | `1_scrape.py` | Builds the two corpora from minecraft.wiki: **C_pre** (each page's newest revision at or before 31 March 2025) and **C_post** (pages first created after it), then filters out pages that are not real base-game mechanics |
| 2 | `2_index.py` | Chunks both corpora (1,500 characters, 250 overlap, `[Page: <title>]` prefix) and embeds them into ChromaDB |
| 3 | `3_generate.py` | Scenario-first question generation, verification against C_pre, and assembly (700 answerable, 300 unanswerable) |
| 4 | `4_reference_answers.py` | Reference answers from an oracle model given the source wiki pages |
| 5 | `5_answer.py` | Runs a model on the benchmark: no retrieval, single-shot RAG or iterative RAG; thinking enabled or disabled |
| 6 | `6_judge_responses.py` | Response judge: labels every response ABSTAIN, DENY or ANSWER |
| 7 | `7_judge_accuracy.py` | One accuracy judge: scores answers 1 / 0.5 / 0 against the references (run once per judge) |
| 8 | `8_metrics.py` | Correct Abstention Rate, Answerable Attempt Rate, Answer Accuracy, leakage accuracy, and paired tests |

Shared modules: `config.py` (settings and endpoints), `prompts.py` (every prompt, verbatim), `retrieval.py`
(dense, BM25, hybrid, reranking, query rewriting, iterative RAG), `wiki.py` (MediaWiki API and wikitext cleaning).
`annotation/` holds the human-annotation tool (see its README).

## Data

The benchmark questions, reference answers, corpora snapshots and model responses will be available at:
**TODO: data link**

The corpora can also be rebuilt from minecraft.wiki with `1_scrape.py` and `2_index.py` (steps 1-2 below).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.11
pip install -r requirements.txt
```

Every model is called through an OpenAI-compatible endpoint, so any server works (vLLM, llama.cpp's
`llama-server`, OpenRouter). Set the endpoints in the environment or in a `.env` file next to the scripts:

| Variable | Used for | Paper setting |
|---|---|---|
| `DC_EMBED_URL`, `DC_EMBED_MODEL` | embeddings (indexing, dense retrieval, question generation) | Qwen3-Embedding-8B, Q8_0 GGUF, `llama-server` |
| `DC_RERANK_URL`, `DC_RERANK_MODEL` | cross-encoder reranker (verification; ablation) | nvidia/llama-nemotron-rerank-1b-v2, vLLM |
| `DC_REWRITE_URL`, `DC_REWRITE_MODEL` | query rewriting (ablation only) | Qwen3 30B-A3B, thinking disabled |
| `DC_LLM_URL` | default endpoint for generation and judges | vLLM |
| `DC_API_KEY` | key for self-hosted endpoints, if they need one | |
| `OPENROUTER_API_KEY` | closed models and the reference-answer oracle | |
| `DC_DATA_DIR` | where corpora and the index are written (default `data/`) | |

## Rebuilding the benchmark (steps 1-4)

```bash
python 1_scrape.py pre --seed-categories          # C_pre -> data/c_pre.json
python 1_scrape.py post                           # C_post -> data/c_post.json
python 1_scrape.py filter                         # drop joke, version, story, media and non-article pages
python 2_index.py                                 # data/chroma (C_pre and C_post collections)

python 3_generate.py --stage generate --generator-url $URL --generator-model qwen3.5-122b-a10b
python 3_generate.py --stage judge    --verifier-url  $URL --verifier-model  gemma-4-31b-it
python 3_generate.py --stage assemble --out questions.csv

python 4_reference_answers.py -i questions.csv -o references_answerable.csv
python 4_reference_answers.py -i questions.csv -o references_unanswerable.csv --unanswerable-only
```

- Generation runs with thinking disabled: temperature 0.3 for the scenario, 0.5 for the question, 0 otherwise.
  The verifier runs at temperature 0 with thinking disabled. The generate and judge stages are separate, so a
  server that serves one model at a time loads each model once.
- `questions.csv` has one row per question: `question` (scenario + question), `source` (`Pre-cutoff` =
  answerable, `latest` = unanswerable), the chunk ids and titles of the scenario and target, and `target_fact`.
- For the paper, every unanswerable question was also reviewed by hand against C_pre. Malformed ones (fake items
  or mechanics) were replaced by unused candidates that had passed the same checks. That review is not
  automated here.
- The reference prompt mentions web search, but no search tool is given: the oracle answers from the full pages
  the question was written from, plus the scenario's chunks. The paper uses DeepSeek V4.1 Flash via OpenRouter.
- C_pre is pinned by the cutoff date. C_post pages are stored at their latest revision when scraped, so a later
  crawl can differ.

## Evaluating a model (steps 5-8)

```bash
M=gemma-4-31b-it                                  # served model name; "-nothink" disables thinking
for cond in "--no-rag" "-r 0" "-r 2"; do
  python 5_answer.py -i questions.csv -m $M         --url $URL $cond
  python 5_answer.py -i questions.csv -m $M-nothink --url $URL $cond
done
python 6_judge_responses.py --dir results --url $URL --model gemma-4-26b-a4b-it
python 7_judge_accuracy.py --dir results --references references_answerable.csv --url $URL --model gemma-4-31b-it --tag gemma
python 7_judge_accuracy.py --dir results --references references_answerable.csv --url $URL --model qwen3.8-27b --tag qwen
python 7_judge_accuracy.py --dir results --references references_answerable.csv --url $URL --model laguna-s-2.1 --tag laguna
python 8_metrics.py --dir results
```

- **Conditions.** `--no-rag` is no retrieval. `-r 0` is single-shot RAG: the top 5 chunks from the hybrid
  retriever. `-r 2` is iterative RAG: the model may ask for up to two further retrieval rounds when it judges
  the chunks insufficient.
- **Generation settings.** Temperature 0 (`-t 0.7` for the second temperature) and a 16,000-token budget,
  reasoning included.
- **Closed models** go through OpenRouter: `--url https://openrouter.ai/api/v1`.
- **Retrieval ablation:** `--retrieval-mode {dense,bm25,hybrid_rerank,hybrid_rewrite,full}`.
  **Retrieval depth:** `-k 10`.
- **Leakage audit.** Score the no-RAG answers to unanswerable questions against the post-cutoff references:
  `7_judge_accuracy.py --unanswerable --references references_unanswerable.csv --tag <judge>`.
  `8_metrics.py` then reports `leakage_accuracy`.
- **Abstention-trained models** can also be run with Abstain-R1's own template: `--prompt abstain_r1`.
- **Response labels and scoring.** The response judge (Gemma 4 26B-A4B in the paper) runs at temperature 0
  with thinking disabled. Correct Abstention Rate counts only ABSTAIN (strict rule). The lenient rule, reported
  as `CAR_lenient`, also counts DENY. Budget-exhausted answers are left out of both rates and score 0 for
  accuracy.

## Reproducibility notes

- Two runs of the same configuration are not identical even at temperature 0. Batched GPU inference is not
  bit-reproducible, and closed APIs are not deterministic. The embedding server's batched mode also perturbs
  query vectors slightly, so the top-5 chunks can differ for a few percent of queries. Compare conditions using
  runs made in the same batch.
- Wiki text is from minecraft.wiki (CC BY-NC-SA 3.0); the corpora are rebuilt by `1_scrape.py`, not shipped.

## Citation

If you find DeclineCraft useful in your research, please cite:

```bibtex
@misc{thakur2026declinecraft,
      title={{DeclineCraft}: A {RAG} Abstention Benchmark for Jointly Measuring Combined Retrieval and Parametric Knowledge},
      author={Atharva Thakur and Abulhair Saparov and Bruno Ribeiro},
      year={2026},
      eprint={2610.28340},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2610.28340},
}
```

## License

The code is released under the MIT License (see `LICENSE`). Data derived from minecraft.wiki (corpora, questions
and reference answers) is licensed CC BY-NC-SA, following the wiki's license.
