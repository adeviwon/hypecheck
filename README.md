# HypeCheck — test the crowd

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
**Try one tweet:** open Colab, run the two setup cells, then paste a tweet in **Check a custom tweet**. It uses the published model, so it works before training.

**Rate several tweets:** Choose **Runtime → Run all**, then replace the quoted entries in Section 9's **Enter tweets here** list. Rerun the list cell and the scoring cell below it. A supported `TICKER | ` prefix takes precedence over cashtags and the default coin. The 40 default entries have known outcomes. A pasted entry is scored against an archived outcome only when its coin and exact text match that example.

For new text, the notebook runs the pinned FinTwitBERT checkpoint. The prediction target is the next complete UTC day, open to close.

## Reproduction
Use Python 3.12 and the pinned requirements. `python -m scripts.prepare_verification` restores exact legacy cache inputs; `python sample_pipeline.py` trains the current batch experiment; `python verify_submission.py` verifies the submission. Source/data downloads are version-pinned and hash-checked. Server notebook execution is not an actual Colab runtime test.

## Sources and limits
[Public post archive](https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto), [FinTwitBERT](https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment), actual Binance market bars, scikit-learn and dependencies; Veritas AI participant reference notebook and Finance brief; Hermes Agent/OpenAI coding assistance. Sparse, biased archive and previously explored test dates mean exploratory evidence, not independent confirmation. No complete live feed, independently calibrated probabilities, causal effect or trading recommendation.

**This project is a prototype and is not financial advice.**
