#!/usr/bin/env python
"""
8_metrics.py
============
Per-run metrics and paired tests, with the paper's rules.

  Correct Abstention Rate (CAR)   share of unanswerable questions answered ABSTAIN (strict, the headline);
                                  the lenient rule also counts DENY
  Answerable Attempt Rate (AAR)   share of answerable questions not answered ABSTAIN (a denial of a real,
                                  pre-cutoff item is a wrong attempt, not a refusal)
  Answer Accuracy                 per answerable question, the mean of the accuracy judges' scores, with an ABSTAIN
                                  or a budget-exhausted answer scoring 0; averaged over all answerable questions
  Leakage accuracy (no RAG)       the same on the unanswerable questions against the post-cutoff references, with
                                  ABSTAIN, DENY and budget-exhausted answers scoring 0
  Budget-exhausted responses (no final answer) are left out of both rates.

Paired tests: two runs on the same questions (single-shot RAG vs no RAG, iterative vs single-shot RAG, thinking
enabled vs disabled) are compared with the exact McNemar test on whether each unanswerable question is answered
ABSTAIN, leaving out questions where either run exhausted its budget; Benjamini-Hochberg over all comparisons,
significant at q < 0.05.

    python 8_metrics.py --dir results            # writes results/metrics.csv and results/paired_tests.csv
"""
import argparse
import csv
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

csv.field_size_limit(sys.maxsize)
NAME_RE = re.compile(r"results_(.+?)(-nothink)?_t([\d.]+)_(norag|rag\d+)(_[a-z_]+?)?(?:_k(\d+))?\.csv$")
COVERAGE = 0.99          # a run gets an accuracy only once the judges have scored (almost) all of it


def read(path, **kw):
    return pd.read_csv(path, engine="python", dtype=str, keep_default_na=False, **kw)


def load(results_dir):
    """One row per (run, question): label, exhausted flag and every judge's score."""
    acc_dirs = sorted(glob.glob(os.path.join(results_dir, "accuracy_*")))
    leak_dirs = sorted(glob.glob(os.path.join(results_dir, "leakage_*")))
    frames = []
    for path in sorted(glob.glob(os.path.join(results_dir, "results_*.csv"))):
        name = os.path.basename(path)
        m = NAME_RE.match(name)
        if not m:
            continue
        d = read(path, usecols=lambda c: c in ("question", "source", "refusal_eval"))
        side = os.path.join(results_dir, "labels", name)
        labels = read(side).set_index("question").label if os.path.exists(side) else pd.Series(dtype=str)
        d = d.assign(run=name, model=m.group(1), thinking="disabled" if m.group(2) else "default",
                     temp=float(m.group(3)), condition=m.group(4), pipeline=(m.group(5) or "_hybrid")[1:],
                     k=int(m.group(6) or 5), label=d.question.map(labels).fillna(""),
                     exhausted=d.refusal_eval.eq("budget_exhausted"))
        for kind, dirs in (("acc", acc_dirs), ("leak", leak_dirs)):
            for folder in dirs:
                f = os.path.join(folder, name)
                tag = os.path.basename(folder).split("_", 1)[1]
                s = read(f).drop_duplicates("question", keep="last").set_index("question").score if os.path.exists(f) else pd.Series(dtype=str)
                d[f"{kind}_{tag}"] = pd.to_numeric(d.question.map(s), errors="coerce")
        frames.append(d)
    if not frames:
        sys.exit(f"no results_*.csv files in {results_dir}")
    return pd.concat(frames, ignore_index=True)


def run_metrics(d):
    acc_cols = [c for c in d.columns if c.startswith("acc_")]
    leak_cols = [c for c in d.columns if c.startswith("leak_")]
    out = []
    for run, g in d.groupby("run", sort=True):
        ok = g[~g.exhausted]
        u, a = ok[ok.source == "latest"], ok[ok.source == "Pre-cutoff"]
        rec = g.iloc[0][["run", "model", "thinking", "temp", "condition", "pipeline", "k"]].to_dict()
        rec.update({
            "n_answerable": int((g.source == "Pre-cutoff").sum()), "n_unanswerable": int((g.source == "latest").sum()),
            "budget_exhausted": int(g.exhausted.sum()),
            "unlabelled": int((ok.label == "").sum()),
            "CAR_strict": 100 * u.label.eq("ABSTAIN").mean() if len(u) else np.nan,
            "CAR_lenient": 100 * u.label.isin(["ABSTAIN", "DENY"]).mean() if len(u) else np.nan,
            "deny_share": 100 * u.label.eq("DENY").mean() if len(u) else np.nan,
            "AAR": 100 * (~a.label.eq("ABSTAIN")).mean() if len(a) else np.nan,
        })
        ans = g[g.source == "Pre-cutoff"]
        if acc_cols and len(ans):
            per_q = ans[acc_cols].mean(axis=1)
            per_q = per_q.where(~(ans.label.eq("ABSTAIN") | ans.exhausted), 0.0)
            rec["answer_accuracy"] = 100 * per_q.fillna(0).mean() if per_q.notna().mean() >= COVERAGE else np.nan
        unans = g[g.source == "latest"]
        if leak_cols and len(unans) and rec["condition"] == "norag":
            per_q = unans[leak_cols].mean(axis=1)
            per_q = per_q.where(~(unans.label.isin(["ABSTAIN", "DENY"]) | unans.exhausted), 0.0)
            rec["leakage_accuracy"] = 100 * per_q.fillna(0).mean() if per_q.notna().mean() >= COVERAGE else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def mcnemar(x, y):
    """Exact McNemar test on paired booleans: a binomial test on the discordant pairs."""
    n01, n10 = int((~x & y).sum()), int((x & ~y).sum())
    return binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0


def benjamini_hochberg(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def paired_tests(d):
    u = d[(d.source == "latest") & (d.pipeline == "hybrid") & (d.k == 5)]
    key = lambda g: g.set_index("question")[["label", "exhausted"]]
    tests = []

    def compare(x, y, claim, subject):
        common = x.index.intersection(y.index)
        x, y = x.loc[common], y.loc[common]
        keep = ~x.exhausted & ~y.exhausted
        a, b = x.label[keep].eq("ABSTAIN"), y.label[keep].eq("ABSTAIN")
        if len(a) >= 30:
            tests.append({"claim": claim, "subject": subject, "n": len(a), "CAR_a": 100 * a.mean(),
                          "CAR_b": 100 * b.mean(), "p": mcnemar(a, b)})

    for (model, think, temp), g in u.groupby(["model", "thinking", "temp"]):
        runs = {c: key(h) for c, h in g.groupby("condition")}
        if "rag0" in runs and "norag" in runs:
            compare(runs["rag0"], runs["norag"], "single-shot RAG vs no RAG", f"{model} ({think}, T={temp})")
        if "rag2" in runs and "rag0" in runs:
            compare(runs["rag2"], runs["rag0"], "iterative vs single-shot RAG", f"{model} ({think}, T={temp})")
    for (model, cond, temp), g in u.groupby(["model", "condition", "temp"]):
        runs = {t: key(h) for t, h in g.groupby("thinking")}
        if "default" in runs and "disabled" in runs:
            compare(runs["default"], runs["disabled"], "thinking enabled vs disabled", f"{model} ({cond}, T={temp})")
    t = pd.DataFrame(tests)
    if not t.empty:
        t["q"] = benjamini_hochberg(t.p)
        t["significant"] = t.q < 0.05
    return t


def main():
    ap = argparse.ArgumentParser(description="Per-run metrics and paired tests.")
    ap.add_argument("--dir", default="results")
    args = ap.parse_args()
    d = load(args.dir)
    m = run_metrics(d)
    m.round(2).to_csv(os.path.join(args.dir, "metrics.csv"), index=False)
    t = paired_tests(d)
    t.round(4).to_csv(os.path.join(args.dir, "paired_tests.csv"), index=False)
    pd.set_option("display.width", 200)
    cols = [c for c in ["run", "CAR_strict", "CAR_lenient", "AAR", "answer_accuracy", "leakage_accuracy",
                        "budget_exhausted", "unlabelled"] if c in m]
    print(m[cols].round(1).to_string(index=False))
    if (m.unlabelled > 0).any():
        print("\nNOTE: some responses have no label yet (run 6_judge_responses.py); they count as attempts here.")
    if not t.empty:
        print("\nPaired tests (exact McNemar, Benjamini-Hochberg):")
        print(t.groupby("claim").apply(lambda g: f"{int((g.significant & (g.CAR_a > g.CAR_b)).sum())} up, "
                                              f"{int((g.significant & (g.CAR_a < g.CAR_b)).sum())} down of {len(g)}")
              .to_string())
    print(f"\nwrote {args.dir}/metrics.csv and {args.dir}/paired_tests.csv")


if __name__ == "__main__":
    main()
