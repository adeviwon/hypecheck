# HypeCheck — Signal or Noise?

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

Test dates: **2026-04-12 through 2026-10-21**, 965 coin-days across 193 shared UTC dates. Every method uses the same observations.

| Method | Direction accuracy | Balanced accuracy | Brier error ↓ |
|---|---:|---:|---:|
| Always up | 46.53% | 50.00% | 0.5347 |
| Training majority | 53.47% | 50.00% | 0.4653 |
| Previous direction | 49.22% | 48.95% | 0.5078 |
| Market only | 53.68% | 51.50% | 0.2481 |
| Market + attention | 54.40% | 52.23% | 0.2480 |
| Market + X | 53.06% | 51.23% | 0.2491 |

Social-minus-market Brier difference: **0.001065**. Exploratory 95% interval **[-0.000159, 0.002483]** using 1,000 circular 7-day moving-block resamples with coins kept together by date. Negative favours sentiment. This is not a formal significance or profitability claim.

Only **142** test coin-days had observed eligible posts. Absence from this archive does not imply no discussion on X. Subgroup and per-coin results are in the JSON; the pooled result must not be presented as five independent discoveries.

### Secondary comparison — richer market context

Hourly Binance bars and BTC/USDT context were added **after** the primary run was frozen. They are reported separately, on the same 965 held-out rows, and were not used to select the primary models.

| Method | Direction accuracy | Balanced accuracy | Brier error ↓ |
|---|---:|---:|---:|
| Market only (primary) | 53.68% | 51.50% | 0.2481 |
| Market + X (primary) | 53.06% | 51.23% | 0.2491 |
| Market + BTC + intraday | 50.26% | 48.80% | 0.2495 |
| Market + BTC + intraday + X | 49.53% | 48.39% | 0.2504 |

Intraday features are the previous UTC day's hourly return volatility, high-low range, final-hour return and taker-buy share — each computed only from hourly bars that closed before the cutoff. BTC context is a lagged 1-day and 7-day return.

Richer market inputs made the market baseline **worse**, not better, and sentiment still did not help: sentiment-minus-richer-market Brier difference **0.000926**, exploratory 95% interval **[-0.000196, 0.002227]**, again crossing zero. Two feature sets, one conclusion.

### Independent scoring check

The same posts were scored a second time by a separately produced pass using a **different pinned revision** of the same model (`c079da76…` rather than `da059da3…`).

| Check | Result |
|---|---:|
| Shared posts | 1,122 |
| Same three-class label | 80.4% |
| Same net-sentiment direction | 87.0% |
| Mean absolute difference in P(bullish) | 0.219 |
| Posts differing by more than 0.25 | 31.9% |

Two revisions of the same checkpoint disagree on the direction of roughly one post in eight, and the disagreement is flat across text-length buckets (0.219 for short texts vs 0.221 for long ones), so truncation does not explain it. This is a reproducibility check, not a ground-truth accuracy measure — and it is a reason to treat single-post sentiment as fragile rather than authoritative.

### Data we examined and rejected

Two supplied files were profiled and **not** used. A 34,192-row post expansion collapses to 10,545 posts across only 92 dates inside the experiment window, about 14,000 of them in a single ten-day burst in April 2024, with 33 posts in 2025 and 25 in 2026 — and its in-window content is largely ticker spam and promotional bots. A 200,956-row, 306-asset training file has matched posts on only 2.2% of rows, almost all of them in training, and samples currently listed trading pairs. Neither would have added opinion-bearing memecoin posts to the test period, which is the real bottleneck.

## How it works

1. Fetch a fixed Hugging Face dataset revision and verify its upstream LFS SHA-256. Keep original posts only, nonempty bounded text, exact supported ticker matches and earliest exact-text duplicates.
2. Use conservative source availability `max(created_at, first_posted_at)` where the latter exists. At cutoff `t` (UTC midnight), aggregate only availability in `[t-24h, t)`. Retweets and quotes are excluded. Ticker attribution itself comes from the retrospectively curated source.
3. Frozen `StephanAkkerman/FinTwitBERT-sentiment` yields bullish/bearish/neutral probabilities; daily sentiment is bullish minus bearish. The checkpoint is pinned, uses safe tensor weights, and runs without remote code. Text truncates at 128 tokens.
4. Build continuous daily Binance spot **USDT** bars. Predict whether day-t `close/open - 1` is positive; flat means not-up. Market inputs are strictly lagged returns, volatility, range and volume change. Prices observed later in day t are labels only, never features. The secondary comparison adds hourly-derived intraday features and BTC context, each lagged to the prior completed UTC day and asserted finite before use.
5. Pooled regularised logistic regressions share numeric features and coin identity. Train-only scaling/one-hot encoding. Fixed C candidates `(0.01, 0.1, 1.0)` selected using validation Brier. Decision threshold 0.5. No refit on validation. No test-driven feature or hyperparameter changes.

Train: 2024-03-01 → 2025-09-29. Validation: 2025-10-01 → 2026-04-10. Test: 2026-04-12 → 2026-10-21. Shared date boundaries, roughly 60/20/20; one adjacent date is purged at each boundary so training/validation target windows end strictly before the next partition.

The experiment begins March 2024, after the pinned sentiment checkpoint was public in February 2024. This reduces one chronology risk but is not proof of an independently collected live backtest or an overlap-free pretraining corpus.

## What we added beyond the suggested project

A five-memecoin, original-post dataset; conservative source-availability alignment; a price-only vs attention-only-added vs sentiment-added ablation; a predeclared simple selection procedure; paired date-block uncertainty; historical replay with failures visible; and explicit sparse-coverage accounting. Do not claim we invented FinTwitBERT or logistic regression.

## Usable social coverage after filtering

Counts may exceed unique posts when one post mentions multiple coins. Some last-day posts fall outside the fixed label horizon.

| Coin | Eligible original posts | Observation dates |
|---|---:|---:|
| DOGE | 638 | 306 |
| SHIB | 45 | 42 |
| PEPE | 244 | 167 |
| BONK | 137 | 94 |
| FLOKI | 150 | 114 |

## Did more training history help?

These are validation-only comparisons using recent portions of the training window, not independent experiments or test-set tuning. More history had lower validation Brier in this run; arbitrary extra tweets are not guaranteed to help.

| Training UTC dates | Validation Brier ↓ |
|---|---:|
| 144 | 0.2485 |
| 289 | 0.2466 |
| 578 | 0.2458 |

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

Tested after the run was frozen; it changed nothing about the model or the selection. Coverage is the binding constraint and raising it moves the headline the wrong way — sentiment exists on 14.7% of test days (142 of 965) and lowers accuracy on exactly those days, so `0.147 x 42.96 + 0.853 x 54.80 = 53.06%`, the reported figure. More posts per day and stronger sentiment behave the same (1 post: 52.5% -> 44.3%, n=122; |sentiment| >= 0.6: 47.5% -> 43.6%, n=101). Spearman rho of sentiment against next-day return is train +0.076 (p=0.070), validation -0.180 (p=0.335), test -0.027 (p=0.733) — no monotone relationship for more data to sharpen. The one relationship that reached p<0.05 (|sentiment| vs next-day range on test, rho +0.196) flipped sign across the other periods, so it is reported as a false positive rather than a lead. On direction accuracy market-only beats the training-majority rule by 0.21 points; on Brier it is 46.7% better (0.2481 vs 0.4653). The value is calibrated uncertainty, not direction.

### Would more specific data fix it? (author, wording, hype, past trend)

Asked and tested after the report was frozen: 40 features in seven blocks over the same 738 post-days, three ways of splitting. The split decides the answer.

| Split | Best block | Accuracy | Edge over majority class |
|---|---|---:|---:|
| Flat random 80/20 | + the words used | 0.6014 | +8.11pp |
| Chronological 80/20, 1-day purge | + hype lexicon | 0.5751 | +6.36pp |
| Walk-forward, 4 windows, 460 out-of-sample coin-days | + FinTwitBERT sentiment | 0.5511 | 2 of 4 windows |

The flat split is the one that flatters: 60.8% of its test rows share a calendar date with training rows and 99.6% of its test posts come from authors already seen in training. It manufactures roughly five accuracy points the walk-forward does not reproduce. Controls: the best walk-forward AUC (0.5596) clears a 20-draw label-permutation null (0.4991 +/- 0.0233, max 0.5526) and a feature placebo (0.5143 +/- 0.0143) - the first block in this project to do so - and it is still AUC 0.56 across 460 coin-days. Promising is not tradeable.

No follower counts exist in this data. The only reach proxy is engagement, present on 177 of 1,274 posts and every one of them dated 2026-04-11 or later, so it would teach the model the calendar rather than the author. Excluded, and reported as excluded.

### The hype meter - 234 curated words and phrases, measured

`data/hype-lexicon.csv`. Curated memecoin bull/bear/hype vocabulary, then measured against the corpus rather than trusted. Each term carries the share of posts containing it that the independent replication lexicon flags bullish or bearish, and a hype meter (0-100) = the share of those posts carrying an independent hype marker (exclamation mark, 30% or more capitals, promotion, urgency, extreme claim), against a 16.1% corpus baseline.

- **Validated bullish (7):** `bullish` (n=51, 100%), `breakout` (34, 100%), `send it` (10, 100%), `rally` (8, 100%), `pumping` (8, 100%), `bottoms` (6, 83%), `longs` (6, 50%).
- **Validated bearish (2):** `breakdown` (7, 100%), `bearish` (7, 100%).
- **Contradicted by the data (9):** `sold` is 62.5% bullish, `selling` 37.5%, `sell` 42.9%, `short` 30%, `downside` 40%, `lower` 29.4%, `downtrend` 20%, `buying` 28.6% bearish, `reclaim` 16.7% bearish. Naive bearish vocabulary is actively misleading here: this corpus talks about selling into strength.
- **Never appears at all (67):** `to the moon`, `100x`, `rugpull`, `scam`, `rekt`, `lambo`, `guaranteed`, `yolo`, `moonshot`. This is technical-analysis commentary from 49 accounts, not retail shilling.
- **The hype meter does not measure hype.** Spearman between the post-level hype score and the independent hype markers is +0.012. The direction lexicon, by contrast, agrees with the independent lexicon on 105 of the 145 posts where it fires (72%) and covers 30.8% of posts against 11.4% - nearly three times the reach, from the same corpus.

## Limitations and failures

- Selected influencers are not representative of all X. Missing observations do not mean no discussion occurred.
- Sparse and changing coverage; author/source changes can mimic sentiment effects. No causal claim.
- Pretrained sentiment checkpoint was public before the experiment period, but training-corpus overlap is not fully auditable.
- English sentiment model; source language is not independently verified. Multi-coin posts share one sentiment score.
- Sentiment is truncated to 128 tokens. Probabilities are not independently calibrated; sentiment certainty is not return certainty.
- Historical USDT exchange bars exclude fees, slippage, latency and execution constraints; no profitability claim.
- Only one held-out period. Date-block intervals are exploratory, and the final period must not be reused to tune models.
- Two pinned revisions of the same sentiment checkpoint disagree on the direction of about 13% of shared posts, so these scores are model-revision dependent rather than a stable ground truth.

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
