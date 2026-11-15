# HypeCheck — public-post market forecast

Paste a public tweet, choose its memecoin and get an estimated next-UTC-day direction. Optional earlier public posts supply recent context. No live X feed is connected.

[Open the submission in Colab](https://colab.research.google.com/github/adeviwon/hypecheck/blob/main/HypeCheck.ipynb) · [Try the trained browser model](https://adeviwon.github.io/hypecheck/predict.html)

## Actual training
Raw wording → train-only character TF-IDF → trained logistic regression for direction. A separate ridge regression estimates whole-coin return, not a tweet-caused effect. The notebook visibly fits the models using actual market outcomes. This is not transformer fine-tuning.

## Honest held-out results

| Method | Coin-day-weighted accuracy |
|---|---:|
| tweet_only | 49.28% |
| tweet_and_context | 45.02% |
| coin_only | 50.68% |
| always_up | 48.40% |
| training_majority | 51.60% |
| training_prevalence | 51.60% |
| previous_completed_day | 48.86% |

Training: 1,451 post/coin events across 859 coin-days. Validation: 71 events across 55 coin-days. Test: 271 events across 219 coin-days and 112 shared dates. Reused archive: exploratory, not independent confirmation. Neither wording model beats the training-majority baseline. Return estimates do not beat the training-constant error baseline.

## Use
In Colab select Runtime → Run all, then edit Section 9. The static browser demo runs the same safe-JSON model locally, without uploading text. The target is the next complete UTC day, open to close, not the next 24 hours.

## Local reproduction
Use **Python 3.12** (verified on 3.12.14) and Node.js 22+ for the UI checks. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify_submission.py
python tweet_forecast.py
python serve.py --port 8000
```

Open http://localhost:8000/predict.html. The gate restores the archived sentiment-cache subset from the pinned expanded cache, requiring an exact match to its original manifest hash; it does not synthesize scores or rerun neural inference. The public source archive downloads on first use if absent. Server verification does not constitute a live Google Colab test. Optional cross-browser regression instructions are in `qa_playwright.py`; its Playwright dependency is separate from model-training requirements.

## Sources and limits
Public financial-tweets-crypto archive, actual Binance daily market bars, scikit-learn and dependencies; pinned revisions and hashes are in source. Hermes Agent/OpenAI assisted coding. No causal impact identification, no independently calibrated probabilities, no validated trading edge, no complete live sentiment feed. Older files are retrospective research, not an alternative current submission.

**This project is a prototype and is not financial advice.**
