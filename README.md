# HypeCheck

Can the wording and sentiment of public X posts improve next-day memecoin forecasts? In our retrospective tests, the coin-only baseline was more accurate on both later dates and coins excluded from training.

[Run in Colab](https://colab.research.google.com/github/adeviwon/hypecheck/blob/main/HypeCheck.ipynb) · [Read the technical write-up](WRITEUP.md)

## Try your own posts

- **One tweet in Colab.** Run the first two setup cells, paste the text into **Check a custom tweet**, select its coin, and run the cell. The first novel tweet may download FinTwitBERT.
- **Several tweets in Colab.** Choose **Runtime → Run all**, then edit the list in Section 9 under **Enter tweets here**. Run that cell and the scoring cell directly below it. Each list entry is rated separately. The section also demonstrates a historical same-coin batch.

The target is whether the coin closes above its open on the next complete UTC day. These are experimental estimates. The earlier [five-coin replay](https://adeviwon.github.io/hypecheck/) and [13-coin sentiment replay](https://adeviwon.github.io/hypecheck/headlines.html) show historical forecasts from separate experiments.

## Reproduce

## Use
**Try one tweet:** open Colab, run the two setup cells, then paste a tweet in **Check a custom tweet**. It uses the published model, so it works before training.

**Rate a list of 40 posts:** Section 9's **Enter tweets here** block holds 40 real posts held out of training, shows what the model got right, and accepts your own list.

Runtime → Run all. Section 5 visibly fits Model 1; Section 8 fits balanced Model 2 and the character trial. Section 9 runs a genuine historical batch and offers the optional multi-post/CSV app. New text runs the actual pinned pretrained sentiment network. Missing/invalid models or inputs block forecasting rather than fabricate scores.

`verify_submission.py` reconstructs the historical sentiment cache, runs the Python and JavaScript tests, and checks notebook source hashes and regeneration. See the [write-up](WRITEUP.md) for training commands, data provenance, and model details.

## Reproduction
Use Python 3.12 and the pinned requirements. `python prepare_verification.py` restores exact legacy cache inputs; `python sample_pipeline.py` trains the current batch experiment; `python verify_submission.py` verifies the submission. Source/data downloads are version-pinned and hash-checked. Server notebook execution is not an actual Colab runtime test.

## Verification
Current evidence: [artifacts/crowd-verification.json](artifacts/crowd-verification.json); judge instructions and limitations: [JUDGE_READINESS.md](JUDGE_READINESS.md). The model implementation received independent review; the presentation-only changes were rerun and regression-tested. Older submission-verification.json is explicitly archived.

## Sources and limits
[Public post archive](https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto), [FinTwitBERT](https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment), actual Binance market bars, scikit-learn and dependencies; Veritas AI participant reference notebook and Finance brief; Hermes Agent/OpenAI coding assistance. Sparse, biased archive and previously explored test dates mean exploratory evidence, not independent confirmation. No complete live feed, independently calibrated probabilities, causal effect or trading recommendation.

**This project is a prototype and is not financial advice.**
