# HypeCheck — Signal or Noise?

**AI + Finance · Veritas AI Hackathon 2026**

Does X sentiment add predictive information beyond market history for DOGE, SHIB, PEPE, BONK and FLOKI?

**Finding:** this fixed retrospective experiment does **not demonstrate improved forecasts from sentiment**. Market + X has a higher (worse) held-out Brier score than market-only; the exploratory date-block uncertainty interval crosses zero. This is a research demo, not an edge, a live signal or a trading strategy.

- **Demo:** https://adeviwon.github.io/hypecheck/
- **Colab:** open `HypeCheck.ipynb` in Google Colab, or upload that file. No account credentials are needed by this repository.
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

## How it works

1. Fetch a fixed Hugging Face dataset revision and verify its upstream LFS SHA-256. Keep original posts only, nonempty bounded text, exact supported ticker matches and earliest exact-text duplicates.
2. Use conservative source availability `max(created_at, first_posted_at)` where the latter exists. At cutoff `t` (UTC midnight), aggregate only availability in `[t-24h, t)`. Retweets and quotes are excluded. Ticker attribution itself comes from the retrospectively curated source.
3. Frozen `StephanAkkerman/FinTwitBERT-sentiment` yields bullish/bearish/neutral probabilities; daily sentiment is bullish minus bearish. The checkpoint is pinned, uses safe tensor weights, and runs without remote code. Text truncates at 128 tokens.
4. Build continuous daily Binance spot **USDT** bars. Predict whether day-t `close/open - 1` is positive; flat means not-up. Market inputs are strictly lagged returns, volatility, range and volume change. Prices observed later in day t are labels only, never features.
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
python serve.py --port 8766
```

Open `http://127.0.0.1:8766`. The published dashboard requires no Python or inference server. Initial inference is CPU-capable and checkpoints progress after each batch. It may take several minutes depending on hardware. `python pipeline.py --stage data` only checks data; `--stage sentiment` resumes scoring. Cached `--stage train` reuses frozen sentiment scores.

`requirements.txt` pins direct dependencies including patched setuptools. `requirements.lock.txt` records the tested full Python environment. The CPU PyTorch index is explicit; no paid API keys. Internet access is required for first downloads. Binance availability varies by region; the pipeline stops on errors and never substitutes fabricated market data.

## Verification and security

The Python tests cover lag invariance under future-price perturbation, source-availability bounds, tampered dataset rejection, missing price days, chronological purges, exact recomputed metrics, JSON coefficient prediction parity and HTTP isolation. Node tests cover cutoff/date/availability boundaries and absent values. Browser replay is also exercised against the actual generated report.

Read `SECURITY.md`. Public hosting serves only static files; no trading, uploads, secrets, account connection or arbitrary URL fetches. A restrictive CSP and safe DOM rendering reduce XSS risk. Header capabilities differ between local preview and GitHub Pages. Dependency auditing is not proof of absence of vulnerabilities, and the CPU PyTorch build is separately identified when the auditor cannot resolve it.

## Limitations and failures

- Selected influencers are not representative of all X. Missing observations do not mean no discussion occurred.
- Sparse and changing coverage; author/source changes can mimic sentiment effects. No causal claim.
- Pretrained sentiment checkpoint was public before the experiment period, but training-corpus overlap is not fully auditable.
- English sentiment model; source language is not independently verified. Multi-coin posts share one sentiment score.
- Sentiment is truncated to 128 tokens. Probabilities are not independently calibrated; sentiment certainty is not return certainty.
- Historical USDT exchange bars exclude fees, slippage, latency and execution constraints; no profitability claim.
- Only one held-out period. Date-block intervals are exploratory, and the final period must not be reused to tune models.

The model sometimes confidently forecasts the wrong direction. The UI supports every test date, not just successful examples. No user interviews or live deployment validation have been performed; do not claim them. With more time, collect a prospective, source-balanced stream and evaluate on a genuinely new future period.

**This project is a prototype and is not financial advice.**

## Credits and sources

- Dataset: https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto (MIT; fixed revision and SHA in `pipeline.py`).
- Pretrained model: https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment (MIT), Stephan Akkerman and Tim Koornstra. Their language model and sentiment work are reused, not our invention.
- Price data and field definitions: https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#klinecandlestick-data . Binance market data is not relicensed as our code.
- Libraries: PyTorch, Hugging Face Transformers, safetensors, scikit-learn, pandas, NumPy, PyArrow and Requests. Library licenses remain their own.
- Evaluation guidance: https://scikit-learn.org/stable/common_pitfalls.html .
- AI tools: Hermes Agent/OpenAI coding assistance was used for research, implementation, review and testing. The team must understand and explain all code and results before submission.
- Hackathon rules: https://highschoolhackathon.org/briefs.html . Add the three team members' actual names in the submission form; this repository does not invent identities.

Code is MIT licensed. Third-party posts, model weights, market data and libraries retain their original terms.
