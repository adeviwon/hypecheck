# HypeCheck technical write-up

Memecoin traders often see posts claiming that a coin is about to surge. We tested whether the wording and estimated tone of those posts improve the next-day direction forecast. The comparison uses actual Binance outcomes and the same evaluation rows for every model.

## Data and prediction target

- **Post archive.** `expanded_inputs.select_posts` keeps original posts with a supported ticker and 8–4,000 characters, then deduplicates exact trimmed text by earliest availability. Availability is the later of publication and the archive timestamp when both are present. A missing archive timestamp falls back to publication. A ticker tag can be a passing mention.
- **Snapshots.** For each coin and observed cutoff time `t`, retain posts in the inclusive interval `[t − 24h, t]`, deduplicate whitespace-normalized text, and keep the latest 200 posts. The cutoff is driven by observed posts, so the evaluation describes coin-days with eligible posts. It does not estimate performance on days without posts.
- **Label.** Set `D = floor_UTC(t) + 1 day` and label the snapshot `1` if Binance spot `close[D] > open[D]`, otherwise `0`. Snapshots for the same coin and target day share a label. The prior-day baseline uses the last completed UTC bar at the cutoff, two calendar days before `D`.

The pinned [post archive](https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto/tree/c4b17a94a667af3da1c9ed5144f998e3a4f29966) and [FinTwitBERT checkpoint](https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment/tree/da059da3b3bbcb43f9ed1aeb5ae61644010c7e1e) are declared in [pipeline.py](pipeline.py). [The loader](colab_setup.py) pins the 13-coin OHLCV table and frozen sentiment cache by SHA-256. The public archive is sparse and its availability timestamps have incomplete coverage.

## Split and leakage controls

The test boundary comes from the 80% point of the eligible daily price calendar. A training row must satisfy `target_day + 1 day < test_boundary`. The five training coins are DOGE, SHIB, PEPE, BONK, and FLOKI. A post mentioning any of the eight excluded coins is also removed from the established-coin cohort, even if it tags a training coin. Preprocessing and classifiers fit on training rows. The code checks the time gap and verifies that training source IDs do not appear in the unseen-coin cohort.

| Partition | Snapshots | Coin-days | Observed target dates (UTC) |
|---|---:|---:|---|
| Training | 991 | 556 | 23 Mar 2024 – 2 Dec 2025 |
| Later dates, established coins | 155 | 127 | 21 Apr – 20 Oct 2026 |
| Later dates, excluded coins | 99 | 77 | 23 Apr – 3 Oct 2026 |

Each snapshot gets weight `1 / n`, where `n` is the number of snapshots for its coin and target day in that partition. This gives each observed coin-day total weight one in fitting and evaluation before any class weighting. Calendar dates and overlapping windows still create dependence. The notebook includes a stratified random 80/20 diagnostic on training rows only. Its two parts share 112 dates.

## Models and inference

1. **Wording.** Join snapshot text with newlines and apply scikit-learn word TF-IDF with smoothed IDF and L2 row normalization. The character trial uses word-boundary character n-grams of lengths 2–5 and sublinear term frequency.
2. **Estimated sentiment.** Frozen FinTwitBERT softmax scores each post with a 128-token limit. The six numerical features are mean bullish-minus-bearish, its population standard deviation, mean bullish, mean bearish, mean neutral, and `log1p(post_count)`. StandardScaler fits on training rows. Exact archive texts can reuse cached sentiment; novel text uses the pinned checkpoint with safetensors.
3. **Coin identity.** One-hot encode the training coins. An excluded coin contributes zero in this feature block.
4. **Direction.** Fit L2 logistic regression and classify UP at `p ≥ 0.5`. Model 1 uses `C=1` and the snapshot weights. Model 2 keeps the same features and applies scikit-learn's balanced class weights. The character trial applies balanced class weights with `C=10`.

The safe JSON export records the intercept, vocabulary, IDF, feature coefficients, means, scales, and categories. At inference, `z = intercept + Σ feature_value × coefficient` and `p = sigmoid(z)`. The displayed words are the largest signed TF-IDF contributions to this logit. Their magnitudes describe the fitted model.

The editable list in Section 9 applies the batch-trained model to one post at a time. Its 40-post historical illustration uses 20 single-post snapshots from each test partition, filtered for readability and selected across coins. Model 2 got 14 of 40 correct, compared with 22 for coin-only and 16 for always-up. These counts describe the selected posts. The primary evaluation below weights coin-days and accounts for shared dates in its intervals.

## Results

| Model | Later dates accuracy | Later dates Brier | Unseen coins accuracy | Unseen coins Brier |
|---|---:|---:|---:|---:|
| Word TF-IDF + sentiment + coin | 36.09% | 0.2766 | 49.03% | 0.2650 |
| Balanced word model | 36.75% | 0.2768 | 43.83% | 0.2662 |
| Balanced character trial | 40.03% | 0.3418 | 45.13% | 0.2970 |
| Coin only | 51.18% | 0.2532 | 55.84% | 0.2484 |
| Training prevalence | 49.61% | 0.2503 | 55.84% | 0.2487 |
| Always up | 50.39% | 0.4961 | 44.16% | 0.5584 |
| Previous completed day | 47.24% | 0.5276 | 55.84% | 0.4416 |

Brier error is the weighted mean squared error of the predicted probability. The balanced model's paired accuracy difference against coin-only was **−14.44 percentage points** on later dates (95% interval **−23.76 to −3.94**) and **−12.01 points** on unseen coins (**−24.14 to +1.49**). We average paired snapshot differences within coin-days and resample seven-observed-date circular blocks 1,000 times, keeping coins on the same date together. These archive dates were explored before. The results did not demonstrate a forecasting edge, and the probabilities have no independent calibration.

The unseen-coin result combines a time shift and a coin shift. Coverage also varies greatly by coin, from 35 PENGU snapshots to one TURBO snapshot. The results support conclusions about these tested samples. They do not identify price impact, profitability after fees, or performance on a prospective sample.

## Reproduction and artifacts

Run these from the repo root with the environment described in the [README](README.md).

```sh
python sample_pipeline.py
python build_headline_colab.py
```

- [Batch report](docs/crowd-report.json), [held-out predictions](artifacts/crowd-predictions.csv), and [model parameters](artifacts/crowd-model.json) come from `sample_pipeline.py`. `crowd-source.json` pins the published model used by the quick Colab form. A newly generated export is separate from that immutable release until its revision and hash are updated together.
- The notebook verifies pinned source files before import. For a clean local execution, copy it to a scratch path and run `python run_nb_capture.py <scratch-notebook> <empty-runtime-directory>`. The runner overwrites the scratch notebook with executed output.
- The five-coin and 13-coin replays remain available with their published predictions. Retired experiments, earlier verification records, and the original manifest remain in [Git history](https://github.com/adeviwon/hypecheck/tree/e1ee51c0b61d0f955244e1a6a52bff53a167c84d).

The project adapts the Veritas AI Finance brief and participant reference notebook. Data comes from the public crypto-post archive and Binance spot bars. Modelling uses scikit-learn, PyTorch, and Hugging Face Transformers. Hermes Agent and OpenAI provided coding assistance.
