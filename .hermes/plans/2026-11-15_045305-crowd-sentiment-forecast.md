# HypeCheck Crowd Signal Implementation Plan

> **For Hermes:** Use subagent-driven-development to execute task-by-task, with spec review followed by quality review. Respect the designated Cline engine for delegated swarm coding; if unavailable, report it and implement directly rather than invent execution.

**Goal:** Build a stand-out, honest hackathon product that accepts a collection of public X posts about one memecoin, estimates sentiment in that supplied collection, and tests whether wording plus sentiment improves a next-day market-direction forecast.

**Architecture:** A batch snapshot is the unit of both training and inference. A pinned frozen FinTwitBERT estimates post sentiment; trainable wording and aggregate-sentiment features predict real Binance outcomes. A Python inference engine serves the runnable Colab interface; the public static page must not pretend it can execute a neural sentiment model without a working backend.

**Tech Stack:** Existing Python 3.12 environment, pandas/NumPy, PyTorch/Transformers safe pinned weights, scikit-learn, safe-JSON model exports, IPython/ipywidgets for the working Colab demo. Keep the existing verified input-fetch and notebook-generation pipeline.

---

## 1. Product contract

### User journey
1. Select a supported memecoin.
2. Paste a batch of public posts, separated by blank lines, or upload a CSV with text and optional publication/availability timestamps.
3. Confirm the snapshot cutoff in UTC. Default: actual forecast submission time. Show exact forecast start/end dates.
4. Inspect the supplied-sample sentiment and data-quality summary.
5. Run the trained market forecast, with the matched baseline and evidence warning beside it.
6. Optionally explore a frozen historical snapshot and reveal its actual later market result.

### Three distinct output panels
- **Crowd Pulse:** estimated bullish/neutral/bearish distribution, mean directional sentiment, disagreement, unique-post count, duplicates removed and unknown timing. These describe only supplied posts, not the entire X population.
- **Market Forecast:** estimated next-complete-UTC-day UP probability and a direction label. Show exact UTC target window, the baseline forecast, and the held-out difference against the baseline. Do not label a model probability as independently calibrated confidence.
- **Evidence Check:** coverage limitations, repeated-post concentration, timestamp verification status and actual held-out performance. If no reliable incremental skill was demonstrated, say **No demonstrated forecasting edge** prominently.

Do not require users to invent a public-sentiment score. Derive it from their batch. External sentiment numbers may be displayed as sourced reference information, but must not enter a trained model unless its training data contains the same defined feature.

### Scope and honesty
- The interface accepts a sample or collection, never promises **all tweets on X**.
- No X API subscription, scraping, login or private account access is required.
- User-supplied context is not automatically authentic, complete, same-coin or correctly timestamped. Mark declarations and verified metadata separately.
- Do not claim an individual post caused a market movement.
- Remove the failed return-magnitude estimate from the primary product card. Retain it only in a labelled research appendix if useful; no precise price-impact claim.
- No buy/sell recommendation, expected-profit claim, financial advice or transformer-fine-tuning claim.

## 2. Timing and training unit

Use a rolling batch snapshot, not a single tweet plus an ad hoc string.

- Snapshot key: `(coin, cutoff_utc)`.
- Historical cutoff: conservative archive availability, `max(created_at, first_posted_at)`, at each observed same-coin update. Deduplicate coincident coin/cutoff snapshots.
- Snapshot contents: eligible unique same-coin texts with availability in `[cutoff − 24 hours, cutoff]`. Including the triggering post at the cutoff is intentional; no later posts are allowed.
- Target start: midnight UTC immediately following the cutoff date. Target end: the following midnight, exclusive.
- Label: actual target-day `close > open`; flat is not up.
- Runtime default: cutoff is actual submission time in UTC. Pasting an old post analyses its wording at this submission time; it does not create a historical backtest. Explicit historical mode is separate.
- If CSV timestamps exist, validate timezone, reject future rows and report excluded out-of-window rows. If pasted texts have no timestamps, mark **Timing unverified — supplied snapshot**; do not claim the 24-hour rule was independently checked.
- Each `(coin, target_day)` receives total training/evaluation weight one across its snapshots. Multiple snapshots share a market outcome and are not independent successes.
- Keep existing calendar-based train/validation/test boundaries and a purge at each horizon boundary. Snapshot contents must remain strictly prior to the target.

This supports a forecast at an arbitrary current time without pretending an unfinished daily sentiment summary is a completed-day input.

## 3. Model contract

### Features
- Train-only TF-IDF on the joined unique batch texts, using the same preprocessing in training and runtime.
- Frozen FinTwitBERT probabilities from the pinned revision: mean bullish, mean bearish, mean neutral, mean bullish-minus-bearish, disagreement and log unique-post count.
- Coin identity.
- Optional market context only when the feature definition uses bars completed before the cutoff. No unfinished posting-day close or later engagement.
- Bound batch size and text length consistently in training and inference. State truncation and excluded rows. Apply the same deterministic duplicate handling in both paths.

### Comparisons
Predeclare the primary comparison before running:
**Wording + estimated sentiment + available market context versus the same model without wording/sentiment.**
Also report always up, train-majority/prevalence, previous completed-day direction, coin-only, wording-only and sentiment-only. Fit transforms on train; choose regularisation on validation Brier error only. Do not select a product model or horizon from whichever has the highest test score.

Train on actual subsequent market labels, never on model-generated price guesses. Frozen sentiment inference is feature extraction, not transformer training. Keep the actual downstream `.fit(...)` code visible in Colab.

### Release criteria
- Same test snapshots for every model; publish snapshot count, distinct coin-days and distinct dates.
- Report accuracy, Brier error and paired date-block intervals for the primary difference. Preserve missing calendar dates when constructing calendar-week blocks.
- Refit label-permutation and feature-placebo controls; do not merely shuffle predictions.
- Report full-panel missingness separately from observed-batch performance. Missing archive posts are not evidence that X was silent.
- Reused archive/test dates make this exploratory. Even a positive result is not independent confirmation.
- If the combined model loses, retain the honest negative result. Do not change horizons, assets or sample quotas to rescue it.

## 4. Stand-out demo without invented claims

**Demo sequence:** import a real archived batch → show sentiment/disagreement → reveal forecast and baseline → reveal actual next-day direction → show whether text added value on the complete held-out sample.

- Use a clearly labelled historical replay, not a fake live feed.
- Provide the full eligible replay list, not a carousel of selected wins.
- Show an example where the crowd is bullish but the model or actual market disagrees. Label any chosen illustration as selected and keep population metrics visible.
- Explain why **bullish sentiment** and **probability of a price increase** are different quantities.
- Add a counterfactual sentiment/wording ablation only as **model sensitivity**, never real-world causal influence. Defer it if the core product is not complete.
- Keep loading code short, tables readable and infrastructure collapsed. Preserve the existing escaping of public text and saved rich outputs.

## 5. Implementation tasks

### Task 1 — Freeze the batch schema and parser
**Create:** `crowd_snapshot.py`, `test_crowd_snapshot.py`.
- Define bounded text-only and CSV inputs, supported coins, snapshot cutoff, provenance and verification status.
- Add failing tests for empty input, duplicate texts, future timestamps, exact 24-hour boundary, equal cutoff timestamps, mixed timezones, unsupported coins and size limits.
- Implement only the parser and deterministic snapshot rules.
- Run: `/home/ubuntu/Projects/hypecheck/.venv/bin/python -m unittest -v test_crowd_snapshot.py`.
- Acceptance: parser returns post records and an audit report; no invented timestamps or coverage guarantee.

### Task 2 — Construct actual rolling historical batches
**Create:** `crowd_dataset.py`, `test_crowd_dataset.py`.
**Reuse read-only:** `pipeline.py`, `expanded_inputs.py`, `data/expanded-sentiment.json`, `data/expanded-memecoin-daily.csv`.
- Add failing fixtures for target-date alignment, coincident cutoffs and exclusion of later text.
- Reconstruct batches from the verified original source and cached frozen sentiment scores. Verify cache signature before reuse.
- Assert exact raw-bar label reconstruction, shared-date partitioning and total weight one per coin-day.
- Audit and publish actual eligible counts; do not assume the old 271-event denominator remains unchanged.

### Task 3 — Train the batch forecast with ablations
**Create:** `crowd_model.py`, `test_crowd_model.py`.
- Add tests proving preprocessing sees only training texts, regularisation selection ignores test labels, and feature order is preserved on reload.
- Fit the predeclared models and refitted controls; save a safe-JSON checkpoint, prediction ledger and measured report.
- Test safe-checkpoint reconstruction against actual sklearn predictions, not just a handcrafted fixture.
- Publish `artifacts/crowd-model.json`, `artifacts/crowd-predictions.csv`, `docs/crowd-report.json`.
- A failed baseline comparison is an acceptable scientific result, not a reason to alter the target.

### Task 4 — Implement genuine batch inference
**Create:** `crowd_inference.py`, `test_crowd_inference.py`.
- Load pinned FinTwitBERT once, using safe weights and `trust_remote_code=False`; process new public texts in bounded batches.
- Reuse the identical feature-building and checkpoint code as training. Do not substitute a phrase list or a single-post model.
- Show exact submission/target UTC timestamps and whether timing was verified or user-declared.
- Exercise both cached archived text and genuinely new text through real neural inference. If inference or checkpoint loading fails, visibly block the forecast; never fabricate a sentiment score.

### Task 5 — Build the three-panel Colab app
**Create:** `crowd_app.py`, `test_crowd_app.py`.
**Modify:** `build_headline_colab.py`, `tweet-forecast-source.json` or a clearly renamed single source manifest.
- Use ipywidgets for paste/CSV, coin, cutoff and a real Predict button connected to the Python inference engine. Keep this as the primary free runnable app.
- Preserve separate sentiment and price panels, the baseline comparison, data-quality audit and the no-demonstrated-edge warning when applicable.
- The browser landing page can link to the functioning Colab. Do not imply that a static page executes the neural model unless a real backend or verified browser implementation exists.
- Main notebook still follows the submission sample and visibly trains. Keep loading to a compact cell rather than embedding source/data walls.

### Task 6 — Add the complete historical replay
**Modify:** `docs/predict.html`, `docs/predict.js`, `docs/predict.css` or replace them with one unambiguous current landing/replay.
- Render all eligible held-out batch snapshots from a frozen artifact.
- Add forecast-then-reveal interaction with the same saved probabilities and actual outcomes used in evaluation.
- Test default replay state, outcome reset, missing data and input escaping.
- Avoid collecting handles, author profiles or new personal information.

### Task 7 — Integrate, verify and review
**Modify:** `verify_submission.py`, `run_nb_capture.py`, `test_notebook_readability.py`, required dependency pins.
- Run the new focused tests first, then the full suite once.
- Execute the actual notebook from an empty directory with genuine outputs; compare its report against the audited report.
- Verify new-input neural inference, saved-model reload, rich table/chart capture, input escaping, source/data hashes and byte-identical regeneration.
- Dispatch spec review first, then code-quality review. Fix blockers and rerun affected gates.
- Record server execution separately from an actual Google Colab run. Do not claim Colab training without runtime evidence.

### Task 8 — Publish one coherent submission
**Modify:** `README.md`, `WRITEUP.md`, `SUBMISSION.md`, current notebook and demo links.
- Explain the exact user/input/output, primary comparison, measured result and limitations.
- Credit public data, pretrained teacher, libraries and AI coding assistance.
- Remove the obsolete alternative notebook only after verifying it is not a dependency; do not force-push or erase Git history.
- Keep unfinished work from other sessions untouched and stage explicit paths only.
- After normal publication, read back the exact remote notebook/model/report and verify their hashes. Verify public access and the actual runnable demo path.

## 6. Non-goals
No trading bot, real-money recommendations, causal-impact score, guaranteed forecast, complete X archive, follower/author profiling, paid API dependency, or new 15/25 sentiment quota. Do not fine-tune a large transformer merely to make the project look more advanced.

## 7. Risk register
- **Sparse archive:** sample sentiment may not represent public sentiment. Show sample limitations; more complicated models do not fix missing data.
- **Weak existing results:** both current raw-post models underperform a simple majority rule. Batch aggregation is a new tested hypothesis, not a promised improvement.
- **Runtime parity:** single-post training cannot back a batch UI; match the units before displaying a forecast.
- **Temporal ambiguity:** use an explicit cutoff and exact next-UTC-day window. Historical and current-time modes must not be confused.
- **CPU/model download costs:** bound inference, reuse safely pinned local weights and provide honest timing. No made-up fallback if neural inference is unavailable.
- **Deadline/quality:** prioritise a working batch pipeline, honest baseline table and one clear demo over extra models or visual flourishes.

## Definition of done
One current notebook accepts a coin-specific batch, derives clearly labelled sample sentiment, trains and runs a matching next-day predictor, exposes real out-of-sample results versus baselines, and has verified runtime/source/model parity. The demo never turns a weak or negative result into a confident trading recommendation.
