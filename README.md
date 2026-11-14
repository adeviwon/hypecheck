# HypeCheck — Signal or Noise?

**Veritas AI International High School Hackathon 2026 · AI + Finance**

Does archived X sentiment improve next-day forecasts beyond market history? **Not demonstrably in our test.** A research prototype, not a trading system.

## Start here
- **Submission notebook:** [Open in Colab](https://colab.research.google.com/github/adeviwon/hypecheck/blob/main/HypeCheck.ipynb). Run top to bottom; slow exploratory search is optional. Saved primary-run outputs are included.
- **Approximately one-page write-up:** [WRITEUP.md](WRITEUP.md)
- **Historical replay demo:** https://adeviwon.github.io/hypecheck/
- **Readiness and experiment comparison:** [SUBMISSION_AUDIT.md](SUBMISSION_AUDIT.md)
- **Demo script and form checklist:** [SUBMISSION.md](SUBMISSION.md)

## Concrete tweet tests
[Read the complete real archived bearish-tweet ledger](tweet-tests/README.md): full texts, source links, fresh sentiment scores, frozen-model direction calls and correct/incorrect outcomes. This is an illustrative subset of already examined data, not a new independent accuracy claim.

## Measured results
965 held-out coin-days across 193 shared UTC dates; earlier training/validation dates with one-day boundary purges. Lower Brier error is better.

| Method | Accuracy | Brier error |
|---|---:|---:|
| Always up | 46.53% | 0.5347 |
| Training majority | 53.47% | 0.4653 |
| Previous direction | 49.22% | 0.5078 |
| Training prevalence | 53.47% | 0.2491 |
| Market only | 53.68% | 0.2481 |
| Market + attention | 54.40% | 0.2480 |
| Market + X | 53.06% | 0.2491 |

The primary comparison is market + X against market-only, not the best of many experiments. Its paired Brier difference is +0.001065, exploratory 95% date-block interval [−0.000159, +0.002483]. We do not demonstrate incremental value from sentiment. Only 142/965 test coin-days have eligible posts.

Training prevalence is the fair constant-probability baseline. The hard majority forecast is retained for accuracy, not used to claim calibration. Same-day correlations do not establish that posts cause price changes. No profitability or durable trading-edge claim is made.

## Reproduce (Python 3.12)
The self-contained Colab downloads and hash-checks all inputs, scores pinned FinTwitBERT, trains the models and runs tests. No paid API or project credentials.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_nb_capture.py HypeCheck.ipynb /tmp/hypecheck-fresh-run
```

Run `test_pipeline.py` inside the generated working directory after execution: it needs downloaded inputs and the regenerated sentiment cache. A fresh clone intentionally excludes caches and model weights. Local notebook execution does not verify Google Colab's package-install step; package mirrors and Python compatibility can differ.

Dashboard checks and localhost preview:
```bash
python -m unittest -v test_web.py
node docs/app.js --self-test
node test_dom.mjs
python serve.py --port 8766
```

## Security and limitations
Static dashboard: no account connection, trading, uploads or arbitrary URL fetches. Input hashes, pinned model revisions, safe tensor weights, no remote model code, no pickled estimator, adversarial leakage/tamper tests. See [SECURITY.md](SECURITY.md); this is not a security certification.

The archive is retrospective, sparse and not representative of all X. Missing posts do not prove silence. Subsequent exploratory work repeatedly examined the test period; no further tuning or independent-confirmation claim is justified. No prospective deployment or user study was performed. Older exploratory notebooks/scripts remain for transparency, not as endorsed alternative results.

## Credits
Dataset: [financial-tweets-crypto](https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto); model: [FinTwitBERT-sentiment](https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment), Stephan Akkerman and Tim Koornstra; market data: Binance/Binance Vision. Revisions and third-party terms are in the notebook and source. Libraries: PyTorch, Transformers, safetensors, scikit-learn, pandas, NumPy, PyArrow, Requests and dependencies. Hermes Agent/OpenAI assisted research, implementation, review and testing. The team must understand and explain the work.

**Team:** Nathaniel Wong, Aaden Yick, Alastair Chu — Malvern College Hong Kong. Code MIT; third-party data and models retain their terms. [Official brief](https://highschoolhackathon.org/briefs.html).

**This project is a prototype and is not financial advice.**
