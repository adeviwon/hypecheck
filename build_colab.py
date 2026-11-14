"""Generate HypeCheck.ipynb — the complete Veritas AI Hackathon submission notebook.

Reads the current pipeline.py / test_pipeline.py / docs/report.json so the notebook always
embeds the measured artifacts. Run with:  python3 build_colab.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = "adeviwon/hypecheck"
RAW = f"https://raw.githubusercontent.com/{REPO}/main/data"
COLAB = f"https://colab.research.google.com/github/{REPO}/blob/main/HypeCheck.ipynb"
SITE = "https://adeviwon.github.io/hypecheck/"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


REPORT = json.loads((ROOT / "docs/report.json").read_text())
M = {row["name"]: row for row in REPORT["metrics"]}
PIPELINE = (ROOT / "pipeline.py").read_text()
TESTS = (ROOT / "test_pipeline.py").read_text()

FILES = ("posts.parquet", "market-hourly.csv", "expanded-daily.csv", "independent-post-scores.csv",
         "DOGE-prices.csv", "SHIB-prices.csv", "PEPE-prices.csv", "BONK-prices.csv", "FLOKI-prices.csv")
SHA = {name: digest(ROOT / "data" / name) for name in FILES}
SIZE = {name: (ROOT / "data" / name).stat().st_size for name in FILES}

PRED = REPORT["predictions"]
with_posts = [row for row in PRED if row["post_count"] > 0]
disagree = [row for row in PRED if (row["market_prob"] >= 0.5) != (row["social_prob"] >= 0.5)]
disagree_posts = [row for row in disagree if row["post_count"] > 0]


def right(rows, key):
    return sum(1 for row in rows if (row[key] >= 0.5) == bool(row["actual_up"]))


def pct(x):
    return f"{100 * x:.2f}%"


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(out)


VALUES = {
    "@@METRIC_TABLE@@": table(
        [[name, pct(M[name]["accuracy"]), pct(M[name]["balanced_accuracy"]), f"{M[name]['brier']:.4f}", M[name]["n"]]
         for name in ["Always up", "Training majority", "Previous direction",
                      "Market only", "Market + attention", "Market + X"]],
        ["Method", "Direction accuracy", "Balanced accuracy", "Brier error (lower is better)", "Test coin-days"]),
    "@@EXT_TABLE@@": table(
        [[row["name"], pct(row["accuracy"]), f"{row['brier']:.4f}"] for row in REPORT["extended"]["metrics"]],
        ["Method (secondary arm, identical held-out rows)", "Direction accuracy", "Brier error"]),
    "@@COVERAGE_TABLE@@": table(
        [[row["coin"], row["eligible_posts"], row["days_with_posts"], row["price_days"]] for row in REPORT["coverage"]],
        ["Coin", "Eligible original posts", "Days with posts", "Price days"]),
    "@@DATA_TABLE@@": table(
        [[f"`{name}`", f"{SIZE[name] / 1e6:.2f} MB", f"`{SHA[name][:16]}...`"] for name in FILES],
        ["File", "Size", "SHA-256 (first 16 characters)"]),
    "@@SHA_JSON@@": json.dumps({name: SHA[name] for name in FILES if name != "posts.parquet"}, indent=4),
    "@@POSTS_SHA@@": SHA["posts.parquet"],
    "@@BRIER_MARKET@@": f"{M['Market only']['brier']:.4f}",
    "@@BRIER_SOCIAL@@": f"{M['Market + X']['brier']:.4f}",
    "@@GAP@@": f"{REPORT['uncertainty']['mean_social_minus_market_brier']:+.6f}",
    "@@LOW@@": f"{REPORT['uncertainty']['interval95'][0]:+.6f}",
    "@@HIGH@@": f"{REPORT['uncertainty']['interval95'][1]:+.6f}",
    "@@N_WITH@@": str(len(with_posts)),
    "@@N_TOTAL@@": str(len(PRED)),
    "@@N_DATES@@": str(PRED and REPORT["metrics"][0]["n_dates"]),
    "@@N_DISAGREE@@": str(len(disagree)),
    "@@N_DISAGREE_POSTS@@": str(len(disagree_posts)),
    "@@RIGHT_MARKET@@": str(right(disagree, "market_prob")),
    "@@RIGHT_SOCIAL@@": str(right(disagree, "social_prob")),
    "@@RIGHT_MARKET_POSTS@@": str(right(disagree_posts, "market_prob")),
    "@@RIGHT_SOCIAL_POSTS@@": str(right(disagree_posts, "social_prob")),
    "@@DATASET_REV@@": REPORT["meta"]["dataset_revision"],
    "@@SENT_MODEL@@": REPORT["meta"]["sentiment_model"],
    "@@SENT_REV@@": REPORT["meta"]["sentiment_revision"],
    "@@IND_REV@@": REPORT["meta"]["independent_scoring_revision"],
    "@@LIMITATIONS@@": "\n".join("- " + item for item in REPORT["meta"]["limitations"]),
    "@@REPO@@": REPO,
    "@@RAW@@": RAW,
    "@@SITE@@": SITE,
    "@@COLAB@@": COLAB,
}

cells = []


def fill(text):
    for token, value in VALUES.items():
        text = text.replace(token, value)
    return text


def md(text):
    src = fill(text.strip()).splitlines(keepends=True)
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in [l.rstrip("\n") for l in src]]})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": fill(text.strip()).splitlines(keepends=True)})


md("""
# HypeCheck — Signal or Noise?

**Track: AI + Finance** · Veritas AI International High School Hackathon 2026
**Team:** Nathaniel Wong (model) · Aaden Yick (data) · Alastair Chu (demo and write-up) · Malvern College Hong Kong

*This project is a prototype and is not financial advice.*

**Our question.** Does X sentiment add predictive information beyond market history about the
**next-day direction** of five memecoins — DOGE, SHIB, PEPE, BONK and FLOKI?

**What goes in, what comes out.** In: the posts published in the 24 hours before a fixed UTC cutoff,
plus that coin's own price history up to the same moment. Out: a probability that the next UTC day
closes up, placed beside three simple baselines and a market-only model, plus a historical replay
where you can see what actually happened next.

**The headline result is negative, and we report it as measured.** Adding sentiment did **not**
improve the forecasts. Market + X scored a *worse* held-out probability error (Brier
@@BRIER_SOCIAL@@) than market-only (@@BRIER_MARKET@@), and on the @@N_WITH@@ test coin-days that actually had posts it was
clearly worse (42.96% vs 51.41% direction accuracy). The date-block interval for the difference
crosses zero. Below, we show exactly where it breaks and why.

**How the guide would scope it.** The guide's example of a weekend-sized version of *"predict the stock market"* is
*"a test of whether news-headline sentiment helps predict next-day price moves for five stocks, reported honestly either way."*
Our project is exactly that test — a test of whether X-post sentiment helps predict next-day price moves for five memecoins (DOGE, SHIB, PEPE, BONK, FLOKI), reported honestly either way — with the market and the news source swapped.

**Starting point.** Suggested project **4.2.4 News headlines and the stock market** (AI + Finance
brief), which invites *"a different market, time period or news source"*. We changed the market
(five memecoins instead of the Dow Jones), the news source (X posts instead of newspaper headlines),
and the evaluation discipline — a pre-declared selection rule, purged partitions, an ablation, a
date-block uncertainty estimate and a failure analysis.

---

## Run it top to bottom

| Step | What it does | Roughly |
|---|---|---|
| 1 · Setup | Installs the pinned libraries (Colab only) | 1–2 min |
| 2 · Data | Fetches every input from its pinned public source and checks SHA-256 | 1–2 min |
| 3 · Code | Writes `pipeline.py` — the whole experiment, readable | instant |
| 4 · Train | Scores posts with frozen FinTwitBERT and fits the models | 3–10 min CPU |
| 5 · Results | Held-out metrics, baselines, per-coin detail, uncertainty | instant |
| 6 · Failures | The confident calls that were wrong, with the posts behind them | instant |
| 7 · Tests | Leakage, tamper and boundary checks | instant |
| 8 · Write-up | Our submission write-up, form answers and credits | — |

`Runtime → Run all` produces every number below from scratch. No paid API, no account, no credentials.

**Links.** [Dashboard](@@SITE@@) · [This notebook on Colab](@@COLAB@@) · [Repository](https://github.com/@@REPO@@)
""")

code("""
# 1 · Setup — a pinned environment, so a rerun reproduces our measured numbers.
import subprocess
import sys

IN_COLAB = "google.colab" in sys.modules
print(f"Python {sys.version.split()[0]} on {sys.platform} · running in Google Colab: {IN_COLAB}")

PINNED = [
    "numpy==2.5.4", "pandas==3.1.1", "pyarrow==26.0.1", "scikit-learn==1.9.1",
    "requests==2.34.2", "transformers==5.23.0", "safetensors==0.8.0",
    "huggingface-hub==2.4.0", "torch==2.15.0+cpu", "setuptools==84.2.1",
]

if IN_COLAB:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *PINNED,
         "--extra-index-url", "https://download.pytorch.org/whl/cpu"],
        check=True,
    )
    print("Installed the pinned libraries (CPU PyTorch build).")

import numpy
import pandas
import sklearn

print(f"numpy {numpy.__version__} · pandas {pandas.__version__} · scikit-learn {sklearn.__version__}")
""")

md("""
## 1 · The data — three public sources, pinned

Everything the model ever sees arrives here. Nothing is pulled from a live feed at run time, nothing
is invented, and nothing is forward-filled: a missing day stops the run instead of being papered over.

| # | Source | What we take from it | Pin |
|---|---|---|---|
| 1 | [HF `StephanAkkerman/financial-tweets-crypto`](https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto) (MIT) | 104,379 posts with text, UTC timestamps, ticker tags and a retrospective collection time | revision `@@DATASET_REV@@`, upstream LFS SHA-256 `@@POSTS_SHA@@` |
| 2 | [Binance public market data](https://github.com/binance/binance-spot-api-docs) (daily and 1-hour spot bars, USDT pairs) | 1,025 daily bars per coin and 1,040 complete UTC days of hourly bars | fixed 2024-01-01 → 2026-10-21 horizon |
| 3 | Our own derived files, published in the [repository](https://github.com/@@REPO@@) so this notebook runs anywhere | hourly aggregation, BTC context, and a second independent scoring pass | pinned by SHA-256 in `pipeline.py` |

The price files are the exact bytes our measured run used. The pipeline can also rebuild them live
from Binance's public REST API (`api.binance.com/api/v3/klines`) — that is how they were made — but we
fetch the published copies first, because Binance blocks some regions and some cloud IP ranges, and a
submission that only runs in certain countries is not a submission. Either way the digests are
verified before anything is trained.

**Honest note on sizing.** Only **1,124** of the 104,379 posts survive our eligibility rules
(original posts only, non-empty bounded text, an exact supported ticker, deduplicated, inside the
fixed window). That is small, and we treat it as small everywhere below.
""")

code("""
# 2 · Fetch every input and verify it. Bad bytes stop the run.
import hashlib
from pathlib import Path

import requests

ROOT = Path.cwd()
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

PINNED_POSTS_URL = ("https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto/resolve/"
                    "@@DATASET_REV@@/data/train.parquet")
PUBLISHED = @@SHA_JSON@@


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fetch(name, url, expected, min_bytes=4096):
    \"\"\"Download once, verify the digest off the wire, and keep the file only if it matches.\"\"\"
    dest = DATA / name
    if dest.exists() and sha256(dest) == expected:
        return dest, "already present"
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=(10, 180))
            response.raise_for_status()
            break
        except Exception as error:                                    # noqa: BLE001
            if attempt == 2:
                raise RuntimeError(f"Could not fetch {name}: {error}") from error
    if len(response.content) < min_bytes:
        raise ValueError(f"{name} is implausibly small; refusing to use it")
    part = dest.with_suffix(dest.suffix + ".part")
    part.write_bytes(response.content)
    if sha256(part) != expected:
        part.unlink(missing_ok=True)
        raise ValueError(f"{name} SHA-256 does not match the pinned copy; refusing to continue")
    part.replace(dest)
    return dest, "downloaded"


for name, expected in [("posts.parquet", "@@POSTS_SHA@@")] + list(PUBLISHED.items()):
    url = PINNED_POSTS_URL if name == "posts.parquet" else f"@@RAW@@/{name}"
    path, how = fetch(name, url, expected)
    print(f"  ok  {name:32s} {path.stat().st_size / 1e6:9.2f} MB  {sha256(path)[:16]}...  {how}")

missing = [name for name in ("posts.parquet", "market-hourly.csv", "expanded-daily.csv",
                             "independent-post-scores.csv") if not (DATA / name).exists()]
assert not missing, f"missing required inputs: {missing}"
print("\\nEvery input verified by SHA-256. Nothing here is invented.")
""")

code("""
# 3 · Look at the data before training on it — the guide's own advice.
import pandas as pd

posts = pd.read_parquet("data/posts.parquet")
print(f"Source posts: {len(posts):,}   columns: {list(posts.columns)}")

sample = posts.loc[posts.tweet_type.eq("tweet"), ["tweet_id", "created_at", "first_posted_at", "text", "tickers"]]
for row in sample.sample(5, random_state=7).to_dict("records"):
    tags = ", ".join(row["tickers"]) if row["tickers"] is not None else "(none)"
    print(f"\\n  tagged: {tags}")
    print(f"  posted: {row['created_at']}   collected: {row['first_posted_at']}")
    print(f"  text:   {str(row['text'])[:140]!r}")

print("\\nTwo clocks matter. `created_at` is when the post was written; `first_posted_at` is when the"
      "\\narchive first saw it. We use the LATER of the two as the moment the information was actually"
      "\\navailable — the conservative choice, and the one that makes this a real forecasting task.")
""")

md("""
## 2 · How the experiment works

`pipeline.py`, written in full by the next cell, is the entire model. The short version:

**1 · Availability, not just timestamps.** A post counts at cutoff `t` (UTC midnight) only if
`max(created_at, first_posted_at)` falls in `[t-24h, t)`. Retweets and quotes are dropped; exact
duplicate text keeps only its earliest appearance; a post must carry an exact supported ticker
(`$DOGE`, `#SHIB`, `PEPE`, ...). The word "bonk" in ordinary prose is not treated as BONK.

**2 · Frozen sentiment, never fine-tuned.** [FinTwitBERT-sentiment](https://huggingface.co/@@SENT_MODEL@@)
(`StephanAkkerman`, MIT, revision `@@SENT_REV@@`) scores each post bullish / bearish / neutral using
safe tensor weights and no remote code. The daily score is `bullish - bearish`, alongside the post
count, the disagreement (spread across posts) and the mean bullish and bearish shares. The checkpoint
was public before our experiment window opens, so it was not trained on our test period.

**3 · Prices are labels, never features.** For each coin we take continuous Binance USDT daily bars.
Every input is **lagged**: `ret1`, `ret3`, `ret7`, 7-day volatility, prior-day range and volume change
all describe days that had already closed. The label is `close/open - 1 > 0` for the **next** UTC day,
and flat counts as not-up. A missing or duplicated day raises instead of being filled.

**4 · Split by date, with a purge.** Train 2024-03-01 → 2025-09-29 · validation 2025-10-01 →
2026-04-10 · test 2026-04-12 → 2026-10-21, roughly 60/20/20. One adjacent date is dropped at each
boundary so no training window's target reaches into the next partition.

**5 · Pre-declared model choice, three baselines.** Pooled regularised logistic regressions share the
numeric features and a one-hot coin identity; the scaler is fitted on training data only;
`C` is chosen from `(0.01, 0.1, 1.0)` on **validation** Brier; threshold 0.5; no refit on validation
and no test-driven changes. The baselines are the ones the brief asks for — always up, training
majority and previous direction.

**6 · What we added beyond the brief.** A different market and news source; an ablation from
market-only → + attention → + sentiment; a secondary richer-context arm (BTC returns plus intraday
hourly features); a second independent scoring pass from a different checkpoint revision as a
reproducibility check; a circular date-block bootstrap so a 0.001 Brier gap is not read as a
discovery; and per-coin sparse-coverage accounting.

**7 · The rules we hold ourselves to.** No test-set tuning. Every claim traceable to a number in
`docs/report.json`. Failures printed, not hidden.
""")

code("%%writefile pipeline.py\n" + PIPELINE)

md("""
## 3 · Train and evaluate

This cell runs the whole experiment: fetch and validate, filter posts, score them with frozen
FinTwitBERT, build the lagged feature table, fit the models, and write `docs/report.json`.

CPU inference over ~1,100 posts is the slow part, and it checkpoints after every batch — if the cell
is interrupted, rerun it and it resumes from `data/sentiment.json` instead of starting over. A network
or validation failure stops the run; the pipeline never substitutes made-up data.
""")

code("!python pipeline.py --stage all --batch-size 16")


md("""
## 4 · Results on held-out data

@@N_TOTAL@@ test coin-days across @@N_DATES@@ shared UTC dates, 2026-04-12 → 2026-10-21. Every row above the line is
scored on exactly the same observations, so the comparison is like-for-like.
""")

code("""
import json
from pathlib import Path

import pandas as pd

try:
    from IPython.display import display
except ImportError:                                    # plain-python fallback
    def display(value):
        print(value)


report = json.loads(Path("docs/report.json").read_text())
print("Splits:", report["meta"]["splits"])

metrics = pd.DataFrame({k: v for k, v in row.items() if k != "per_coin"} for row in report["metrics"])
display(metrics[["name", "accuracy", "balanced_accuracy", "brier", "n", "n_dates"]].round(4))
print("Brier is a probability error — lower is better. Accuracy alone hides a near-50% problem.")

display(pd.DataFrame(report["coverage"]))

market = next(row for row in report["metrics"] if row["name"] == "Market only")["brier"]
social = next(row for row in report["metrics"] if row["name"] == "Market + X")["brier"]
uncertainty = report["uncertainty"]
print(f"\\nMarket only Brier {market:.4f}  →  Market + X Brier {social:.4f}")
print(f"Social-minus-market Brier difference: {uncertainty['mean_social_minus_market_brier']:+.6f}"
      f"   (95% date-block interval {uncertainty['interval95'][0]:+.6f} to {uncertainty['interval95'][1]:+.6f})")
print("Positive means sentiment made probability error WORSE. The interval crosses zero, so the"
      "\\nhonest reading is 'no demonstrated improvement', not 'proven harm'.")

print("\\nSecondary arm — richer market context, identical held-out rows:")
display(pd.DataFrame(report["extended"]["metrics"])[["name", "accuracy", "brier", "n"]].round(4))

rows = []
for coin in ["DOGE", "SHIB", "PEPE", "BONK", "FLOKI"]:
    a = next(r for r in report["metrics"] if r["name"] == "Market only")["per_coin"][coin]
    b = next(r for r in report["metrics"] if r["name"] == "Market + X")["per_coin"][coin]
    rows.append({"coin": coin, "n": a["n"], "market_acc": round(a["accuracy"], 3), "social_acc": round(b["accuracy"], 3),
                 "market_brier": round(a["brier"], 4), "social_brier": round(b["brier"], 4)})
print("\\nPer-coin detail:")
display(pd.DataFrame(rows))

print("\\nDoes more training history help? (validation only, never test)")
display(pd.DataFrame(report["learning_curve"]))
print("Independent second scoring pass, different checkpoint revision:")
print(" ", report["independent_check"])
""")

md("""
## 5 · Where it fails — the actually useful part

Two questions matter more than the headline number: does sentiment help where posts really exist, and
what do the confident mistakes look like?
""")

code("""
import json
from pathlib import Path

import pandas as pd

report = json.loads(Path("docs/report.json").read_text())
pred = pd.DataFrame(report["predictions"])
pred["up"] = pred.actual_up.astype(int)
pred["social_right"] = (pred.social_prob >= 0.5) == (pred.up == 1)
with_posts = pred[pred.post_count > 0]

print(f"Test coin-days with observed posts: {len(with_posts)} of {len(pred)} "
      f"({100 * len(with_posts) / len(pred):.1f}%). On the rest, market + X is a market model with a"
      "\\nzero bolted on — which is why the pooled number flatters sentiment.")
for label, frame in (("all rows", pred), ("rows WITH posts", with_posts), ("rows WITHOUT posts", pred[pred.post_count == 0])):
    market = ((frame.market_prob >= 0.5) == (frame.up == 1)).mean()
    social = ((frame.social_prob >= 0.5) == (frame.up == 1)).mean()
    print(f"  {label:20s} n={len(frame):4d}   market-only {100 * market:5.2f}%   market + X {100 * social:5.2f}%")

disagree = pred[(pred.market_prob >= 0.5) != (pred.social_prob >= 0.5)]
pooled_market = int(((disagree.market_prob >= 0.5) == (disagree.up == 1)).sum())
pooled_social = int(((disagree.social_prob >= 0.5) == (disagree.up == 1)).sum())
print(f"\\nThe two models disagreed on {len(disagree)} of {len(pred)} coin-days ({100 * len(disagree) / len(pred):.1f}%).")
print(f"  market-only right {pooled_market} · market + X right {pooled_social}")
posts_disagree = disagree[disagree.post_count > 0]
print(f"  on the {len(posts_disagree)} of those that had posts: market-only right "
      f"{int(((posts_disagree.market_prob >= 0.5) == (posts_disagree.up == 1)).sum())} · market + X right "
      f"{int(((posts_disagree.social_prob >= 0.5) == (posts_disagree.up == 1)).sum())}")

print("\\nMost confident wrong calls from market + X, on coin-days that had posts:")
wrong = with_posts[~with_posts.social_right].copy()
wrong["confidence"] = (wrong.social_prob - 0.5).abs()
for row in wrong.sort_values("confidence", ascending=False).head(5).to_dict("records"):
    example = (row["examples"] or [{}])[0].get("text", "").replace("\\n", " ")[:150]
    print(f"\\n  {row['date']}  {row['coin']:5s}  p(up)={row['social_prob']:.3f}  actual next-day return "
          f"{row['actual_return']:+.2%}  sentiment {row['sentiment']:+.2f}  posts {row['post_count']}")
    print(f"     post: {example!r}")

print("\\nReading these: the misses cluster on posts that are famous crypto-market commentary, whale"
      "\\nwatching or sector talk that merely mentions a ticker — not a directional opinion about that"
      "\\ncoin. The ticker tag comes from the archive's retrospective curation, so a bearish take on"
      "\\n'the crypto gaming sector' became a bearish FLOKI score on a day FLOKI rose.")
print("\\nThe model is also right for the wrong reason: daily prices are persistent, so 'yesterday's"
      "\\ndirection' already scores 49.22% and the market features do most of the work.")
""")

md("""
## 6 · Verification — the checks we ran against ourselves

A result that survives in only one direction is worthless, so the repository ships runnable gates. The
next two cells write `test_pipeline.py` and run it. These are adversarial tests, not decoration:

- **No future leakage.** Tripling prices from a chosen day onward must not change that day's market
  features *at all* — and must change the following day's, proving the test can detect a break.
- **Conservative availability.** `available_at = max(created_at, first_posted_at)`, and every retained
  post sits strictly inside the 24 hours before its own cutoff.
- **Tampering fails closed.** A corrupted `posts.parquet` raises on its SHA-256 instead of training.
- **No silent forward-filling.** Deleting one day of prices raises "Missing daily prices"; an
  out-of-range coin raises "Unknown coin".
- **Chronological purges.** The partition boundaries are asserted, including that no training target
  window reaches into validation.
- **Exported coefficients reproduce the report.** The plain-JSON parameters in
  `artifacts/coefficients.json` are re-scored against the published predictions, so the numbers are
  checkable without unpickling anything — we ship no pickles at all.

The published dashboard in `docs/` reads the same `docs/report.json`, so the interface cannot drift
from the measured results.
""")

code("%%writefile test_pipeline.py\n" + TESTS)

code("!python -m unittest -v test_pipeline.py")

md("""
## 7 · Write-up

**HypeCheck: Signal or Noise?**
**Track:** AI + Finance · **Team:** Nathaniel Wong (model), Aaden Yick (data), Alastair Chu (demo and write-up) — Malvern College Hong Kong

*This project is a prototype and is not financial advice.*

### The problem
A beginner crypto investor watching DOGE, SHIB, PEPE, BONK and FLOKI sees a timeline full of confident
posts and has no way to tell whether any of it is information or noise. The stakes are real: memecoin
moves are violent, and "the crowd knows something" is exactly the belief that hype monetises. We built
the tool we wanted as sceptics — one that answers *"does this actually predict anything?"* with a
measured number instead of a promise.

The guide's example of a well-scoped version of a too-broad idea is *"a test of whether news-headline sentiment helps predict next-day price moves for five stocks, reported honestly either way."* Our project is that test run on five memecoins and X posts rather than headlines, and we report the answer either way. It came out negative: sentiment did not improve the forecasts.

### What we built
In: the posts published in the 24 hours before a fixed UTC cutoff, plus that coin's own price history.
Out: a probability that the next UTC day closes up, next to three simple baselines and a market-only
model, with the option to replay any held-out date and see what happened. It runs as this notebook and
as a static dashboard ([@@SITE@@](@@SITE@@)) reading the same report.

### What we added beyond suggested project 4.2.4
The brief prescribes headlines → sentiment → predict the Dow. We changed the market (five memecoins),
the news source (X posts) and the evaluation discipline: a pre-declared selection rule, purged
partitions, an ablation, a date-block interval, a second independent scoring pass, and per-coin
sparse-coverage accounting. We did **not** invent FinTwitBERT or logistic regression, and we say so.

### How it works
- **Data.** `@@SENT_MODEL@@` is the model; the data is `StephanAkkerman/financial-tweets-crypto` at
  revision `@@DATASET_REV@@` — 104,379 posts reduced to **1,124 eligible** original deduplicated posts.
  Prices: continuous Binance USDT daily bars, 1,025 days per coin, 2024-01-01 → 2026-10-21.
- **Sentiment.** Frozen FinTwitBERT (`@@SENT_REV@@`), bullish / bearish / neutral, 128-token truncation,
  no fine-tuning of any kind.
- **Model.** Pooled regularised logistic regression on lagged market features plus a one-hot coin
  identity, `C` selected on validation Brier, decision threshold 0.5.

### Results
@@METRIC_TABLE@@

Social-minus-market Brier difference **@@GAP@@**, 95% date-block interval **[@@LOW@@, @@HIGH@@]** (1,000
circular 7-day moving-block resamples with coins kept together by date). Positive favours market-only
and the interval crosses zero, so **we do not demonstrate that sentiment improves these forecasts.**

The sharper finding is where sentiment should have helped most. On the **@@N_WITH@@ test coin-days that
actually had posts**, market-only scored 51.41% direction accuracy against 42.96% for market + X. On
the @@N_DISAGREE@@ coin-days where the two models disagreed, market-only was right @@RIGHT_MARKET@@ times against
@@RIGHT_SOCIAL@@ — and on the @@N_DISAGREE_POSTS@@ of those with posts, @@RIGHT_MARKET_POSTS@@ against @@RIGHT_SOCIAL_POSTS@@.

Secondary arm, identical held-out rows:

@@EXT_TABLE@@

Coverage, which is the real constraint:

@@COVERAGE_TABLE@@

### Where it fails
- Posts that are sector commentary, whale-watching or market-wide news but carry a ticker get read as
  coin sentiment. Our most confident error, 2026-10-10 DOGE (`p(up)=0.33`, actual **+0.45%**), rests on
  a post about Bitcoin liquidations; 2026-06-15 FLOKI (`p(up)=0.36`, actual **+3.14%**) rests on "is the
  crypto gaming sector over?" — a bearish sector take, not a call on FLOKI.
- Sparse coverage is structural: DOGE has 638 eligible posts across 306 days while SHIB has 45 across
  42. @@N_WITH@@ of @@N_TOTAL@@ test coin-days had no eligible post at all.
- The archive's ticker tags are retrospective curation, so we cannot prove the tag was right at the time.
- Sentiment here means "text about this coin", never "this coin will rise".

### What we'd do next
1. Collect a prospective, source-balanced stream instead of a retrospective archive, and evaluate on a
   genuinely later period, with the sentiment model fine-tuned on memecoin language.
2. Classify each post's *intent* — directional call vs sector news vs promotion — before scoring it.
   Our failure analysis says that is where most of the error lives.
3. Report a cost-sensitive measure, because a missed move and a false alarm cost a beginner very
   differently.

### Limitations
@@LIMITATIONS@@
""")

md("""
## 8 · Submission form answers

| Form question | Our answer |
|---|---|
| **Project title** | HypeCheck: Signal or Noise? |
| **Building as** | Team |
| **Team details** | Nathaniel Wong (model) · Aaden Yick (data) · Alastair Chu (demo and write-up) — Malvern College Hong Kong. *Before submitting, paste each member's email here in that order, spelled as it should appear on certificates.* |
| **Track** | AI + Finance |
| **Starting point** | 4.2.4 News headlines and the stock market |
| **Scope, in the guide's format** | A test of whether X-post sentiment helps predict next-day price moves for five memecoins (DOGE, SHIB, PEPE, BONK, FLOKI), reported honestly either way. |
| **One-sentence pitch** | HypeCheck takes a memecoin's X posts and its own price history and tells a beginner investor whether the hype carries any predictive signal at all — reporting honestly that, in our measured run, it does not. |
| **Code link** | [This Colab notebook](@@COLAB@@) or the [repository](https://github.com/@@REPO@@) |
| **Write-up** | Section 7 of this notebook, and the repository README |
| **Demo video link** | *optional — add the unlisted YouTube or Drive link* |
| **What we added** | A different market (five memecoins, not the Dow) and source (X posts, not headlines); a frozen pretrained sentiment model with conservative availability alignment; an ablation from market-only → +attention → +sentiment; a pre-declared selection rule with purged partitions; a date-block uncertainty estimate; a second independent scoring pass; and a failure analysis of the confident misses. |
| **Datasets, models and AI tools used** | HF `StephanAkkerman/financial-tweets-crypto`; HF `StephanAkkerman/FinTwitBERT-sentiment`; Binance public klines; scikit-learn, pandas, NumPy, PyArrow, Requests, PyTorch, Hugging Face Transformers, safetensors; Hermes Agent / OpenAI coding assistance. Full credits in section 9. |
| **New to AI?** | No — at least one of us has built with AI before |

*Prototype, not financial advice. The results are the ones we actually measured, including the negative one.*
""")

md("""
## 9 · Sources and credits

- **Dataset:** [StephanAkkerman/financial-tweets-crypto](https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto) (MIT) — revision `@@DATASET_REV@@`, upstream LFS SHA-256 `@@POSTS_SHA@@`.
- **Pretrained model:** [StephanAkkerman/FinTwitBERT-sentiment](https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment) (MIT), Stephan Akkerman and Tim Koornstra. Reused, not our invention; the independent check uses revision `@@IND_REV@@`.
- **Hourly market bars and BTC context:** Binance Vision spot klines, supplied in an independent HypeCheck data package; attributed to Binance Vision under its dataset terms and CC BY-NC-SA 4.0. The daily rows were cross-checked bar-for-bar against our own Binance bars and matched exactly.
- **Market data and field definitions:** [Binance spot API documentation](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#klinecandlestick-data). Binance market data is not relicensed as our code.
- **Libraries:** PyTorch, Hugging Face Transformers, safetensors, scikit-learn, pandas, NumPy, PyArrow, Requests — each keeps its own licence.
- **Evaluation guidance:** [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html).
- **Hackathon brief and rules:** [highschoolhackathon.org](https://highschoolhackathon.org/briefs.html) and the participant guide, sections 5–10.
- **AI tools:** Hermes Agent with OpenAI models was used for research, implementation, review and testing. Every number here is reproducible from this notebook, and the team can explain every part.

Code is MIT licensed (`LICENSE`). Third-party posts, model weights, market data and libraries keep
their original terms.

### Files this notebook produces
`docs/report.json` (every held-out prediction, metric, split, coverage figure and uncertainty
estimate), `artifacts/coefficients.json` (plain JSON model parameters — no pickles),
`data/features.csv` (the exact modelling table) and `artifacts/manifest.json` (SHA-256 of every input
and output).

### Data files this repository publishes, with their digests

@@DATA_TABLE@@

`posts.parquet` is fetched from Hugging Face at the pinned revision; the rest are the exact bytes our
measured run used, hosted in the [repository](https://github.com/@@REPO@@) and verified against these
digests on every run.
""")

code("""
# 10 · Take the results with you (optional; Colab only).
if IN_COLAB:
    from google.colab import files

    files.download("docs/report.json")
    files.download("data/features.csv")
    files.download("artifacts/coefficients.json")
else:
    print("Not in Colab — results are on disk at docs/report.json, data/features.csv, artifacts/.")
""")

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3", "language": "python"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = ROOT / "HypeCheck.ipynb"
out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
print(f"wrote {out.name} — {len(cells)} cells, {out.stat().st_size / 1024:.1f} KB")
unfilled = [t for t in VALUES if t in out.read_text()]
print("unfilled tokens:", unfilled or "none")
