#!/usr/bin/env python3
"""Fetch the public InferenceX benchmark dataset.

Pulls every model in MODELS from the InferenceX public API and writes:

  - data/raw/<Model>.json      raw API response (one JSON array of rows)
  - data/flat/<Model>.csv      flattened CSV (metrics nested → top-level)

Both forms are kept so downstream consumers can pick whichever is convenient
(the JSON preserves the upstream schema verbatim; the CSV is easier to
slice with awk/pandas/spreadsheets).

The list of models was discovered by scraping the InferenceX dashboard JS
bundles and probing each candidate against the API. It is not authoritative
— add new identifiers to MODELS as InferenceX adds them.

Refresh policy: re-run this script to pull fresh measurements. The API
periodically backfills and corrects rows; treat the local files as a
snapshot of the upstream as of the date in summary.json. Do not edit the
files in place — let the next fetch overwrite them.

Usage:
    python benchmark/inferenceX/fetch.py
    python benchmark/inferenceX/fetch.py --model DeepSeek-R1-0528    # single model
    python benchmark/inferenceX/fetch.py --discover                   # re-scan JS for new models
"""
import argparse
import csv
import gzip
import json
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

API_BASE = "https://inferencex.semianalysis.com/api/v1/benchmarks"
DASHBOARD = "https://inferencex.semianalysis.com/"
USER_AGENT = "llm_perf-benchmark-fetcher"

# Models known to return non-empty results from the API. Update by running
# `--discover`. Order is alphabetical for stable diffs; row counts are a
# convenience snapshot, not a contract.
MODELS = [
    "DeepSeek-R1-0528",       # MoE + MLA, FP4/FP8
    "DeepSeek-V4-Pro",        # MoE + MLA
    "GLM-5",                  # MoE
    "Kimi-K2.5",              # MoE
    "Llama-3.3-70B-Instruct-FP8",  # Dense GQA
    "MiniMax-M2.5",           # MoE
    "Qwen-3.5-397B-A17B",     # MoE (17B active)
    "gpt-oss-120b",           # MoE
]

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_FLAT = ROOT / "data" / "flat"


def _http_get(url: str, *, timeout: int = 60) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        return raw


def fetch_model(model: str) -> list[dict]:
    """Fetch all benchmark rows for one model. Returns [] on 4xx."""
    url = f"{API_BASE}?model={urllib.request.quote(model)}"
    try:
        return json.loads(_http_get(url))
    except urllib.error.HTTPError as e:
        if 400 <= e.code < 500:
            return []
        raise


def flatten(row: dict) -> dict:
    """Flatten the `metrics` sub-object into top-level fields.

    The API's BenchmarkRow has metric quantiles (median/mean/p99/std) for
    TPOT, ITL, TTFT, E2E, and interactivity nested under `metrics`. We hoist
    them so each measurement is a flat dict, easier to write to CSV.
    """
    out = dict(row)
    metrics = out.pop("metrics", None) or {}
    out.update(metrics)
    return out


def write_raw(model: str, rows: list[dict]) -> Path:
    out = DATA_RAW / f"{model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, sort_keys=True))
    return out


def write_flat(model: str, rows: list[dict]) -> Path:
    out = DATA_FLAT / f"{model}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    flat = [flatten(r) for r in rows]
    if not flat:
        out.write_text(f"# {model}: 0 rows\n")
        return out
    keys = sorted({k for r in flat for k in r.keys()})
    with out.open("w", newline="") as f:
        f.write(f"# Source: {API_BASE}?model={model}\n")
        f.write(f"# License: Apache 2.0 — see ../../LICENSE\n")
        f.write(f"# Attribution required; see ../../NOTICE\n")
        f.write(f"# Rows: {len(flat)}\n")
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)
    return out


def discover_models() -> list[tuple[str, int]]:
    """Scrape the dashboard JS bundles for model identifiers and probe them.

    Returns identifiers that the API confirms have at least one row. This is
    how MODELS was originally populated; run periodically to catch additions.
    """
    html = _http_get(DASHBOARD).decode("utf-8", errors="replace")
    js_paths = sorted(set(re.findall(r"/_next/static/[^\"\s]+\.js", html)))
    blob = ""
    for path in js_paths:
        try:
            blob += _http_get(
                f"https://inferencex.semianalysis.com{path}", timeout=20
            ).decode("utf-8", errors="replace") + "\n"
        except Exception:
            continue

    candidates: set[str] = set()
    patterns = [
        r'"(DeepSeek-[\w.-]+)"',
        r'"(gpt-oss-[\w.-]+)"',
        r'"(Llama-[\w.-]+)"',
        r'"(Qwen[\w.-]+)"',
        r'"(Kimi-[\w.-]+)"',
        r'"(GLM-[\w.-]+)"',
        r'"(Mixtral[\w.-]+)"',
        r'"(Mistral[\w.-]+)"',
        r'"(Gemma[\w.-]+)"',
        r'"(Grok[\w.-]+)"',
        r'"(MiniMax-[\w.-]+)"',
        r'"(Phi-[\w.-]+)"',
    ]
    for p in patterns:
        for m in re.finditer(p, blob):
            candidates.add(m.group(1))

    working = []
    for model in sorted(candidates):
        rows = fetch_model(model)
        if rows:
            working.append((model, len(rows)))
    return working


def write_summary(per_model_counts: dict[str, int]) -> None:
    summary = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_base": API_BASE,
        "license": "Apache-2.0 (see LICENSE)",
        "attribution_required": True,
        "models": [
            {
                "model": m,
                "rows": per_model_counts.get(m, 0),
                "raw_path": f"data/raw/{m}.json",
                "flat_path": f"data/flat/{m}.csv",
            }
            for m in sorted(per_model_counts)
        ],
        "total_rows": sum(per_model_counts.values()),
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--model", help="Fetch only this model (skip MODELS list)")
    ap.add_argument(
        "--discover",
        action="store_true",
        help="Scrape JS bundles for new model identifiers and report (does not fetch)",
    )
    args = ap.parse_args()

    if args.discover:
        print("Scanning dashboard JS bundles for model identifiers...")
        found = discover_models()
        print(f"\n{len(found)} working identifiers:")
        for m, c in sorted(found, key=lambda x: -x[1]):
            print(f"  {c:>5}  {m}")
        new = sorted({m for m, _ in found} - set(MODELS))
        if new:
            print(f"\nNot in MODELS — add to fetch.py:")
            for m in new:
                print(f'  "{m}",')
        else:
            print("\nNo new models — MODELS is up to date.")
        return 0

    targets = [args.model] if args.model else MODELS
    counts: dict[str, int] = {}
    for model in targets:
        print(f"Fetching {model} ...", end=" ", flush=True)
        rows = fetch_model(model)
        if not rows:
            print("0 rows (skipping)")
            counts[model] = 0
            continue
        write_raw(model, rows)
        write_flat(model, rows)
        counts[model] = len(rows)
        # Quick (hardware, framework) histogram for the user
        hw = Counter((r.get("hardware"), r.get("framework")) for r in rows)
        top = ", ".join(f"{h}/{f}={n}" for (h, f), n in hw.most_common(3))
        print(f"{len(rows)} rows  [{top}]")

    if not args.model:
        write_summary(counts)
        n_with_data = len([c for c in counts.values() if c])
        print(
            f"\nWrote summary.json (total {sum(counts.values())} rows across {n_with_data} models)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
