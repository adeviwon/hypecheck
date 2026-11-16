# Judge readiness — HypeCheck

## Verdict

**Technically runnable, genuine trained ML, with a usable own-input demo. Not evidence of useful trading predictions.** Actual Google Colab execution and the required video are still submission checklist items, not completed by this audit.

## What judges can actually use

- **Browser:** https://adeviwon.github.io/hypecheck/live.html — paste a public post, choose/detect its coin, click **Get forecast**. No login, API key, backend or paid inference. This is a separately fitted wording + coin logistic-regression model, not the full neural-sentiment model.
- **Full model:** https://colab.research.google.com/github/adeviwon/hypecheck/blob/main/HypeCheck.ipynb — Runtime → Run all, then Section 9. The input app now opens automatically. Paste posts separated by blank lines, or select CSV text, choose a coin, leave cutoff blank and click **Analyse batch**. Novel text runs pinned FinTwitBERT then the trained direction classifier. First use may download the checkpoint.

Training labels are actual next-complete-UTC-day close > open outcomes. The notebook visibly fits TF-IDF/logistic-regression models; it does not fine-tune FinTwitBERT. Neither interface is a live X feed or a causal tweet-impact estimator.

## Executed checks

| Check | Observed result |
|---|---|
| Clean Python 3.12.14 install | Documented pinned requirements installed successfully |
| Full submission gate | **107 Python tests passed**, plus all Node parser/DOM/parity checks |
| Empty-directory full notebook | **15/15 code cells passed**, real saved outputs |
| Simulated Colab setup | Fresh environment initially without torch installed it; **15/15 cells passed**. Module sentinel only — not Google infrastructure |
| Own new text through full-model widget | Actual pinned neural inference, then trained Model 2; invalid input cleared stale output |
| Browser/Python parity | 18 held-out cases; maximum probability difference **1.11e-16** |
| Real Chromium interaction | Desktop 1280px and mobile 375px: own-input inference, coin detection, evidence warnings, no inference network call, invalid-input clearing, no horizontal overflow; missing model blocks forecasts |
| Browser model reproduction | Model JSON, parity fixtures and generated page byte-identical after retraining |
| Full-model reproduction | Both fresh runs reproduced every model/report JSON value and every prediction-ledger value exactly; JSON key/CSV column order differed |

Detailed measured evidence and hashes: `artifacts/crowd-verification.json`.

## Fixes

- Enabled the existing full-model input app by default; fixed the stale submission handoff and preserved the newer browser demo.
- Added explicit PyTorch installation fallback for the Colab dependency branch.
- Fixed shared inference: NaN/Infinity, invalid scaling or corrupt feature blocks now block forecasts instead of turning NaN into 100% UP. Regression also checks stale widget output is cleared.
- Fixed inherited vocabulary keys/nonfinite browser scores; local `serve.py` now serves the live demo assets.
- Updated generator sources as well as notebook/README, and pinned changed runtime sources to a published immutable commit. No model-performance numbers were improved or concealed.

## Important limitations

The full balanced model scored **36.75% later-time / 43.83% unseen-coin accuracy**, versus coin-only **51.18% / 55.84%**. The browser wording + coin model scored **41.21% / 50.22%**. These reused historical tests show no demonstrated forecasting edge; the project should be presented as an honest research prototype, not a reliable trading tool.

## Recheck locally

Use Python 3.12, then:

```bash
python -m pip install -r requirements.txt
python verify_submission.py
python serve.py
```

Open `http://127.0.0.1:8766/live.html`. To reproduce fresh full-model inference rather than an archived-text cache hit:

```bash
python -c 'from sample_pipeline import batch_check; print(batch_check("DOGE community momentum is fading, and holders are worried about another selloff.", "DOGE"))'
```

This sentence is synthetic test wording, not an attributed public post or fabricated market data.

**Before Submit:** run the current notebook in actual Google Colab and enter your own text; record/upload the required video of at most two minutes; complete the organiser's form and verify team/eligibility details. A script and a public repo are not a completed submission.
