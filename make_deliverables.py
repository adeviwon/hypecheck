"""Generate human-readable results and a self-contained Colab notebook from real artifacts."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
report = json.loads((ROOT / "docs/report.json").read_text())
meta = report["meta"]
rows = report["metrics"]
metrics = "\n".join(f"| {m['name']} | {m['accuracy']:.2%} | {m['balanced_accuracy']:.2%} | {m['brier']:.4f} |" for m in rows)
coverage = "\n".join(f"| {c['coin']} | {c['eligible_posts']} | {c['days_with_posts']} |" for c in report["coverage"])
curve = "\n".join(f"| {c['train_dates']} | {c['validation_brier']:.4f} |" for c in report["learning_curve"])
u = report["uncertainty"]
ext = report["extended"]
ic = report["independent_check"]
extended_metrics = "\n".join(f"| {m['name']} | {m['accuracy']:.2%} | {m['balanced_accuracy']:.2%} | {m['brier']:.4f} |" for m in ext["metrics"])
readme = f'''# HypeCheck — Signal or Noise?

**AI + Finance · Veritas AI Hackathon 2026**

**Team:** Nathaniel Wong (model) · Aaden Yick (data) · Alastair Chu (demo and write-up) · Malvern College Hong Kong

Does X sentiment add predictive information beyond market history for DOGE, SHIB, PEPE, BONK and FLOKI?

**Scoped the way the guide recommends** — a test of whether X-post sentiment helps predict next-day price moves for five memecoins, reported honestly either way.

**Finding:** this fixed retrospective experiment does **not demonstrate improved forecasts from sentiment**. Market + X has a higher (worse) held-out Brier score than market-only; the exploratory date-block uncertainty interval crosses zero. This is a research demo, not an edge, a live signal or a trading strategy.

- **Demo:** https://adeviwon.github.io/hypecheck/
- **Colab:** [open `HypeCheck.ipynb`](https://colab.research.google.com/github/adeviwon/hypecheck/blob/main/HypeCheck.ipynb) and press `Runtime → Run all`, or upload the file. No account credentials are needed by this repository. The notebook fetches every input itself — pinned Hugging Face revision plus the exact data bytes published in `data/`, each verified against its SHA-256 before anything is trained — so it reproduces the measured report from any machine, including regions where Binance's API is blocked.
- **Results:** `docs/report.json` contains every held-out prediction, metrics, splits, coverage and uncertainty. `artifacts/coefficients.json` contains safe JSON model parameters; no pickled estimator is needed.

## Who this is for

A beginner crypto investor encountering bullish posts and asking whether the hype deserves their trust. Select a historical coin/date, compare market-only and market-plus-X forecasts, then reveal the next UTC day's actual return. It helps question claims, not place orders.

## Results — measured, not promised

Test dates: **{meta['splits']['test'][0]} through {meta['splits']['test'][1]}**, {rows[0]['n']} coin-days across {rows[0]['n_dates']} shared UTC dates. Every method uses the same observations.

| Method | Direction accuracy | Balanced accuracy | Brier error ↓ |
|---|---:|---:|---:|
{metrics}

Social-minus-market Brier difference: **{u['mean_social_minus_market_brier']:.6f}**. Exploratory 95% interval **[{u['interval95'][0]:.6f}, {u['interval95'][1]:.6f}]** using 1,000 circular 7-day moving-block resamples with coins kept together by date. Negative favours sentiment. This is not a formal significance or profitability claim.

Only **{report['subgroups']['with_observed_posts']['n']}** test coin-days had observed eligible posts. Absence from this archive does not imply no discussion on X. Subgroup and per-coin results are in the JSON; the pooled result must not be presented as five independent discoveries.

### Secondary comparison — richer market context

Hourly Binance bars and BTC/USDT context were added **after** the primary run was frozen. They are reported separately, on the same {rows[0]['n']} held-out rows, and were not used to select the primary models.

| Method | Direction accuracy | Balanced accuracy | Brier error ↓ |
|---|---:|---:|---:|
{extended_metrics}

Intraday features are the previous UTC day's hourly return volatility, high-low range, final-hour return and taker-buy share — each computed only from hourly bars that closed before the cutoff. BTC context is a lagged 1-day and 7-day return.

Richer market inputs made the market baseline **worse**, not better, and sentiment still did not help: sentiment-minus-richer-market Brier difference **{ext['uncertainty']['mean_social_minus_market_brier']:.6f}**, exploratory 95% interval **[{ext['uncertainty']['interval95'][0]:.6f}, {ext['uncertainty']['interval95'][1]:.6f}]**, again crossing zero. Two feature sets, one conclusion.

### Independent scoring check

The same posts were scored a second time by a separately produced pass using a **different pinned revision** of the same model (`c079da76…` rather than `da059da3…`).

| Check | Result |
|---|---:|
| Shared posts | {ic['shared_posts']:,} |
| Same three-class label | {ic['class_agreement']:.1%} |
| Same net-sentiment direction | {ic['net_sentiment_sign_agreement']:.1%} |
| Mean absolute difference in P(bullish) | {ic['mean_abs_p_bullish_difference']:.3f} |
| Posts differing by more than 0.25 | {ic['share_p_bullish_differing_over_0_25']:.1%} |

Two revisions of the same checkpoint disagree on the direction of roughly one post in eight, and the disagreement is flat across text-length buckets ({ic['length_diagnostic']['mean_abs_difference_short_text']:.3f} for short texts vs {ic['length_diagnostic']['mean_abs_difference_long_text']:.3f} for long ones), so truncation does not explain it. This is a reproducibility check, not a ground-truth accuracy measure — and it is a reason to treat single-post sentiment as fragile rather than authoritative.

### Data we examined and rejected

Two supplied files were profiled and **not** used. A 34,192-row post expansion collapses to 10,545 posts across only 92 dates inside the experiment window, about 14,000 of them in a single ten-day burst in April 2024, with 33 posts in 2025 and 25 in 2026 — and its in-window content is largely ticker spam and promotional bots. A 200,956-row, 306-asset training file has matched posts on only 2.2% of rows, almost all of them in training, and samples currently listed trading pairs. Neither would have added opinion-bearing memecoin posts to the test period, which is the real bottleneck.

## How it works

1. Fetch a fixed Hugging Face dataset revision and verify its upstream LFS SHA-256. Keep original posts only, nonempty bounded text, exact supported ticker matches and earliest exact-text duplicates.
2. Use conservative source availability `max(created_at, first_posted_at)` where the latter exists. At cutoff `t` (UTC midnight), aggregate only availability in `[t-24h, t)`. Retweets and quotes are excluded. Ticker attribution itself comes from the retrospectively curated source.
3. Frozen `StephanAkkerman/FinTwitBERT-sentiment` yields bullish/bearish/neutral probabilities; daily sentiment is bullish minus bearish. The checkpoint is pinned, uses safe tensor weights, and runs without remote code. Text truncates at 128 tokens.
4. Build continuous daily Binance spot **USDT** bars. Predict whether day-t `close/open - 1` is positive; flat means not-up. Market inputs are strictly lagged returns, volatility, range and volume change. Prices observed later in day t are labels only, never features. The secondary comparison adds hourly-derived intraday features and BTC context, each lagged to the prior completed UTC day and asserted finite before use.
5. Pooled regularised logistic regressions share numeric features and coin identity. Train-only scaling/one-hot encoding. Fixed C candidates `(0.01, 0.1, 1.0)` selected using validation Brier. Decision threshold 0.5. No refit on validation. No test-driven feature or hyperparameter changes.

Train: {meta['splits']['train'][0]} → {meta['splits']['train'][1]}. Validation: {meta['splits']['validation'][0]} → {meta['splits']['validation'][1]}. Test: {meta['splits']['test'][0]} → {meta['splits']['test'][1]}. Shared date boundaries, roughly 60/20/20; one adjacent date is purged at each boundary so training/validation target windows end strictly before the next partition.

The experiment begins March 2024, after the pinned sentiment checkpoint was public in February 2024. This reduces one chronology risk but is not proof of an independently collected live backtest or an overlap-free pretraining corpus.

## What we added beyond the suggested project

A five-memecoin, original-post dataset; conservative source-availability alignment; a price-only vs attention-only-added vs sentiment-added ablation; a predeclared simple selection procedure; paired date-block uncertainty; historical replay with failures visible; and explicit sparse-coverage accounting. Do not claim we invented FinTwitBERT or logistic regression.

## Usable social coverage after filtering

Counts may exceed unique posts when one post mentions multiple coins. Some last-day posts fall outside the fixed label horizon.

| Coin | Eligible original posts | Observation dates |
|---|---:|---:|
{coverage}

## Did more training history help?

These are validation-only comparisons using recent portions of the training window, not independent experiments or test-set tuning. More history had lower validation Brier in this run; arbitrary extra tweets are not guaranteed to help.

| Training UTC dates | Validation Brier ↓ |
|---|---:|
{curve}

## Run locally (Python 3.12)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python pipeline.py --stage all
python -m unittest -v test_pipeline.py test_web.py
node docs/app.js --self-test
node test_dom.mjs
python serve.py --port 8766
```

Open `http://127.0.0.1:8766`. The published dashboard requires no Python or inference server. Initial inference is CPU-capable and checkpoints progress after each batch. It may take several minutes depending on hardware. `python pipeline.py --stage data` only checks data; `--stage sentiment` resumes scoring. Cached `--stage train` reuses frozen sentiment scores.

`requirements.txt` pins direct dependencies including patched setuptools. `requirements.lock.txt` records the tested full Python environment. The CPU PyTorch index is explicit; no paid API keys. Internet access is required for first downloads. Binance availability varies by region; the pipeline stops on errors and never substitutes fabricated market data.

## Verification and security

The Python tests cover lag invariance under future-price perturbation, source-availability bounds, tampered dataset rejection, missing price days, hourly-gap rejection, hourly-feature lag correctness against hand-computed values, BTC lag and same-provider bar cross-checks, chronological purges, exact recomputed metrics for both the primary and secondary models, pinned secondary-source tamper rejection, primary-metric regression pinning, independent-scoring reproduction, JSON coefficient prediction parity and HTTP isolation. Node tests cover cutoff/date/availability boundaries, absent values, the extended report shape and the reveal control (hidden to revealed, with the reported probabilities unchanged). `test_dom.mjs` renders the real report through `docs/app.js` against a stub DOM and asserts both result tables. Browser replay is also exercised against the actual generated report.

Read `SECURITY.md`. Public hosting serves only static files; no trading, uploads, secrets, account connection or arbitrary URL fetches. A restrictive CSP and safe DOM rendering reduce XSS risk. Header capabilities differ between local preview and GitHub Pages. Dependency auditing is not proof of absence of vulnerabilities, and the CPU PyTorch build is separately identified when the auditor cannot resolve it.

### Would better data fix this?

Tested after the run was frozen; it changed nothing about the model or the selection. Coverage is the binding constraint and raising it moves the headline the wrong way — sentiment exists on 14.7% of test days (142 of 965) and lowers accuracy on exactly those days, so `0.147 x 42.96 + 0.853 x 54.80 = 53.06%`, the reported figure. More posts per day and stronger sentiment behave the same (1 post: 52.5% -> 44.3%, n=122; |sentiment| >= 0.6: 47.5% -> 43.6%, n=101). Spearman rho of sentiment against next-day return is train +0.076 (p=0.070), validation -0.180 (p=0.335), test -0.027 (p=0.733) — no monotone relationship for more data to sharpen. The one relationship that reached p<0.05 (|sentiment| vs next-day range on test, rho +0.196) flipped sign across the other periods, so it is reported as a false positive rather than a lead. On direction accuracy market-only beats the training-majority rule by 0.21 points; on Brier it is 46.7% better (0.2481 vs 0.4653). The value is calibrated uncertainty, not direction.## Limitations and failures

''' + "\n".join("- " + item for item in meta["limitations"]) + '''

The model sometimes confidently forecasts the wrong direction. The UI supports every test date, not just successful examples. No user interviews or live deployment validation have been performed; do not claim them. With more time, collect a prospective, source-balanced stream and evaluate on a genuinely new future period.

**This project is a prototype and is not financial advice.**

## Credits and sources

- Dataset: https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto (MIT; fixed revision and SHA in `pipeline.py`).
- Pretrained model: https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment (MIT), Stephan Akkerman and Tim Koornstra. Their language model and sentiment work are reused, not our invention.
- Price data and field definitions: https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#klinecandlestick-data . Binance market data is not relicensed as our code.
- Hourly market bars and BTC context: Binance Vision spot klines, supplied in an independent HypeCheck data package. Attributed to Binance Vision under its dataset terms (https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md) and CC BY-NC-SA 4.0. The daily rows were cross-checked bar-for-bar against our own Binance bars and matched exactly.
- Independent sentiment-scoring pass: a second pinned FinTwitBERT revision (https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment/tree/c079da7618e551d182ffdf68ef7aa25cb9a79667), used only as a reproducibility check.
- Libraries: PyTorch, Hugging Face Transformers, safetensors, scikit-learn, pandas, NumPy, PyArrow and Requests. Library licenses remain their own.
- Evaluation guidance: https://scikit-learn.org/stable/common_pitfalls.html .
- AI tools: Hermes Agent/OpenAI coding assistance was used for research, implementation, review and testing. The team must understand and explain all code and results before submission.
- Hackathon rules: https://highschoolhackathon.org/briefs.html . Team: Nathaniel Wong (model), Aaden Yick (data), Alastair Chu (demo and write-up), Malvern College Hong Kong. The submission form also needs each member's email, spelled as it should appear on certificates.

Code is MIT licensed. Third-party posts, model weights, market data and libraries retain their original terms.
'''
(ROOT / "README.md").write_text(readme)
freeze = subprocess.run(["uv", "pip", "freeze", "--python", str(ROOT / ".venv/bin/python")], capture_output=True, text=True, check=True).stdout
(ROOT / "requirements.lock.txt").write_text("--extra-index-url https://download.pytorch.org/whl/cpu\n" + freeze)

# The submission notebook lives in build_colab.py so the README, this deliverable script and
# the notebook can never disagree about the measured numbers.
subprocess.run([sys.executable, str(ROOT / "build_colab.py")], check=True)
