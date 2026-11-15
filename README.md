# HypeCheck — Signal or Noise?

**AI + Finance · Veritas AI International High School Hackathon 2026**

**Team:** Nathaniel Wong (model) · Aaden Yick (data) · Alastair Chu (demo and write-up) · Malvern College Hong Kong

*This project is a prototype and is not financial advice.*

## The question

Do prior-day headline-style X posts tell us anything about the next UTC day’s direction for **13 memecoins**? These are original social posts, not newspaper headlines or a complete X feed.

**Answer: not reliably in this experiment.** The sentiment-feature model beats always-up and previous-day direction on point accuracy, but not the training-majority rule. Adding sentiment to market history does not demonstrate an improvement. Failed models do not prove prediction impossible.

- **Colab:** https://colab.research.google.com/github/adeviwon/hypecheck/blob/main/HypeCheck.ipynb
- **Demo:** https://adeviwon.github.io/hypecheck/headlines.html
- **Report:** [`docs/headline-report.json`](docs/headline-report.json)
- **Every forecast:** [`artifacts/headline-predictions.csv`](artifacts/headline-predictions.csv)

## User and product

A beginner crypto investor sees a bullish post and wonders whether to trust it. HypeCheck compares prior-information forecasts with simple rules and reveals what happened in a historical replay. It helps inspect claims, not place orders.

## Expanded dataset

The naturally sampled expanded corpus contains 1,773 eligible original posts across 13 memecoins. Natural pretrained sentiment labels: {'bullish': 1238, 'bearish': 358, 'neutral': 177}. No 40-case or 15/25 quota. The same 193 dates remain dependent across coins; 2,509 coin-days are not that many independent tweets.

## Test-period results

**2026-04-12–2026-10-21: 2509 coin-days across 193 UTC dates.** All methods use identical test rows.

| Method | Accuracy | Brier error ↓ |
|---|---:|---:|
| Always up | 46.07% | 0.5393 |
| Previous-day direction | 48.51% | 0.5149 |
| Training majority | 53.93% | 0.4607 |
| Sentiment + coin | 53.09% | 0.2494 |
| Coin only | 53.93% | 0.2485 |
| Market only | 54.40% | 0.2477 |
| Market + sentiment | 53.61% | 0.2486 |
| Training prevalence | 53.93% | 0.2486 |

Primary model: frozen pretrained FinTwitBERT sentiment features plus a trained logistic-regression classifier and coin identity. The encoder is **not fine-tuned** in this experiment. Coin-only and market-only ablations test whether text adds anything. Training prevalence is the appropriate constant-probability reference for Brier error.

**Baseline correction:** previous-day direction is the preceding calendar UTC day’s **close > open**, matching the target. The archived original report/dashboard instead used previous close-to-close return; it is not this comparison.

### Paired accuracy differences

| Sentiment model minus baseline | Difference | Exploratory 95% interval |
|---|---:|---|
| Always up | +7.01 pp | [-1.67, +16.02] pp |
| Previous-day direction | +4.58 pp | [-0.04, +8.69] pp |
| Training majority | -0.84 pp | [-1.83, +0.16] pp |
| Market only | -1.32 pp | [-3.35, +0.80] pp |

Market + sentiment minus market-only Brier error: +0.000908, exploratory 95% interval [+0.000238, +0.001589]. Positive means worse probability error when sentiment is added.

Intervals use paired seven-observed-date circular moving-block resamples, retaining coins together. Every interval above crosses zero.

A secondary expanding-window model scores **53.17%** over the same final dates. It is a robustness check, not independent replication. Per-window results are in the notebook and JSON.

## Do some kinds of headline-style posts matter more?

Fixed keyword buckets: events → promotion → technical → other. Post categories are mutually exclusive by rule priority; coin-day slices overlap. These are approximate rules, not human-validated semantic labels. Counts are post-coin incidences, not necessarily unique tweets.

Each kind’s count and mean sentiment are added to the market-only model and evaluated on **all test rows**. Negative Brier change favours text.

| Kind | Test coin-days with kind | Post-coin incidences | Brier change | 95% interval | Holm-adjusted p |
|---|---:|---:|---:|---|---:|
| events | 16 | 16 | +0.000131 | [-0.000179, +0.000582] | 1.000 |
| promotion | 12 | 13 | -0.000029 | [-0.000326, +0.000246] | 1.000 |
| technical | 53 | 57 | +0.000124 | [-0.000052, +0.000309] | 0.586 |
| other | 161 | 185 | +0.000587 | [-0.000003, +0.001205] | 0.396 |

**No kind demonstrates a reliable improvement here.** All intervals cross zero; promotion and event samples are particularly sparse. Do not promote a flattering small slice into a strategy.

## Method and leakage checks

- Original deduplicated posts; availability is max(publication, archive collection time) where available.
- Only previous UTC-day posts. Predict next UTC-day close > open; flat counts as not up.
- Chronological train/validation/test with one-day purge gaps. Training-only scaling and fitting; select C on validation Brier from 0.01, 0.1 and 1.0. Threshold fixed at 0.5.
- Zero shared train/test UTC dates. Earlier target windows end before the next partition.
- Twenty training-label permutation refits, twenty feature-placebo refits and four expanding windows.
- Real accuracy 53.09%; permuted-label maximum 54.48%; feature-placebo maximum 54.44%. The real model does not beat these null maxima. Accuracy nulls need not be 50% under class imbalance.

## Limitations and investor interpretation

Only **219 of 2509 test coin-days** contain eligible observed posts. Missing data does not mean there was no discussion on X.
- This period has already been inspected. The new experiment and text-kind comparisons are exploratory, not a fresh confirmatory test.
- Selected accounts are not all X. English sentiment, truncated text and collection/edit history limit generalisation. Pretraining overlap is not fully auditable.
- Same-day association is not causation or a forecast. No fees, slippage, execution or profit evaluation was performed.

**Justifies:** question claims that bullish social posts reliably predict tomorrow; demand baseline comparisons and uncertainty.

**Does not justify:** buying, selling, shorting, leverage, profit expectations, causal claims about tweets moving prices, or universal claims that prediction cannot work.

## Reproduce

Open the Colab notebook and Run all. By default it hash-checks cached scores from the pinned pretrained sentiment model and **trains direction classifiers afresh**. Set `RECOMPUTE_SENTIMENT=True` in the input cell to recompute frozen sentiment inference. No keys are required. A verified server run is not reported as actual Colab training.

```bash
python -m pip install -r requirements.txt
python headline_study.py
python test_headline_study.py
python test_pipeline.py
```

`build_headline_colab.py` generates the notebook from executable source; `make_headline_deliverables.py` generates this README from the report. Legacy entrypoints delegate to these generators to prevent restoring abandoned framing.

## What we added

Memecoin-specific market and social source, availability-aware time splits, correctly matched baselines, coin/market ablations, uncertainty, controls, text-kind tests, complete saved predictions and a historical replay.

## Sources and credits

- Posts: https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto ; revision `c4b17a94a667af3da1c9ed5144f998e3a4f29966`.
- Sentiment: https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment ; revision `da059da3b3bbcb43f9ed1aeb5ae61644010c7e1e`.
- Binance USDT daily bars and the independently supplied hourly/BTC inputs; source provenance and input hashes in `pipeline.py`, the notebook and archived `docs/report.json`.
- NumPy, pandas, scikit-learn, PyTorch, Transformers, Hugging Face Hub, Requests.
- Hermes Agent / OpenAI coding assistance for research, implementation and testing. Team members must understand and explain the work.
- Veritas project brief Finance §4.2.4: https://drive.google.com/file/d/15tYD48XRmCgWbk1_lLDtc8lebi6qm-4-/view

*This project is a prototype and is not financial advice.*
