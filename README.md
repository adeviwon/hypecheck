# HypeCheck — test the crowd

**Live demo:** [paste a tweet and get the forecast](https://adeviwon.github.io/hypecheck/live.html) — the trained wording + coin model runs in your browser, no server, with its own held-out scores shown beside the forecast.

Supply a batch of public X posts about one memecoin. HypeCheck separates estimated sample sentiment, next-day direction and the evidence for trusting that prediction. No claim of all-X coverage or causal price impact.

[Current Colab](https://colab.research.google.com/github/adeviwon/hypecheck/blob/main/HypeCheck.ipynb)

## What is actually trained
Default word TF-IDF + logistic regression; then the same features with balanced class weights; then a character-pattern alternative. Aggregate pinned FinTwitBERT sentiment and coin identity stay fixed between comparisons. The classifiers train on actual subsequent market outcomes, not model guesses. No transformer fine-tuning is claimed.

## Honest results

| Model | Later-time accuracy | Unseen-coin accuracy |
|---|---:|---:|
| Model 1 | 36.09% | 49.03% |
| Model 2 | 36.75% | 43.83% |
| Character trial | 40.03% | 45.13% |
| Coin only | 51.18% | 55.84% |
| Always up | 50.39% | 44.16% |
| Training majority | 49.61% | 55.84% |
| Training prevalence | 49.61% | 55.84% |
| Previous completed day | 47.24% | 55.84% |

All accuracy is coin-day weighted. Training: 991 snapshots / 556 coin-days. Later-time test: 155 snapshots / 127 coin-days. Unseen-coin test: 99 snapshots / 77 coin-days. The word-balanced model fails the coin-only control in both tests. No demonstrated forecasting edge.

## Use
Runtime → Run all. Section 5 visibly fits Model 1; Section 8 fits balanced Model 2 and the character trial. Section 9 runs a genuine historical batch and opens the multi-post/CSV app automatically. In Section 9, choose a coin, paste your own public posts separated by blank lines (or select CSV text), leave UTC cutoff blank and click Analyse batch. The first new-text run downloads the pinned sentiment network; allow that download to finish. No API key or paid service is required. New text runs the actual pinned pretrained sentiment network. Missing/invalid models or inputs block forecasting rather than fabricate scores.

The primary target is the next complete UTC day after the snapshot cutoff, open to close, not the next 24 hours. Blank runtime cutoff means the actual submission time; old pasted text is not an automatic backtest.

## Reproduction
Use Python 3.12 and the pinned requirements. `python prepare_verification.py` restores exact legacy cache inputs; `python sample_pipeline.py` trains the current batch experiment; `python verify_submission.py` verifies the submission. Source/data downloads are version-pinned and hash-checked. Server notebook execution is not an actual Colab runtime test.

## Verification
Current evidence: [artifacts/crowd-verification.json](artifacts/crowd-verification.json): recorded cell counts, test counts and genuine execution evidence. The model implementation received independent review; the presentation-only changes were rerun and regression-tested. Older submission-verification.json is explicitly archived.

## Sources and limits
[Public post archive](https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto), [FinTwitBERT](https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment), actual Binance market bars, scikit-learn and dependencies; Veritas AI participant reference notebook and Finance brief; Hermes Agent/OpenAI coding assistance. Sparse, biased archive and previously explored test dates mean exploratory evidence, not independent confirmation. No complete live feed, independently calibrated probabilities, causal effect or trading recommendation.

**This project is a prototype and is not financial advice.**
