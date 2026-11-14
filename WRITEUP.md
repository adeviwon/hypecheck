# HypeCheck: Signal or Noise?

**Track:** AI + Finance  
**Team:** Nathaniel Wong (model), Aaden Yick (data), Alastair Chu (demo and write-up) — Malvern College Hong Kong

## The problem
A beginner crypto investor sees confident social posts but cannot tell whether sentiment adds information beyond price history. We test that claim rather than recommend trades.

## What we built
A reproducible notebook and historical replay dashboard. Input: archived posts available before UTC midnight and completed market bars. Output: next-day up probabilities from market-only and sentiment-added models, alongside simple baselines and the actual historical outcome.

## What we added
We adapt suggested project **4.2.4, News headlines and the stock market** from Dow/headlines to five memecoins/X posts. We add conservative source-availability alignment, purged chronological dates, market/attention/sentiment ablations, date-block uncertainty and explicit failure analysis.

## How it works
104,379 public archived posts yield 1,124 eligible original deduplicated posts about DOGE, SHIB, PEPE, BONK and FLOKI. Frozen, revision-pinned FinTwitBERT scores sentiment. Logistic regression uses lagged market features and coin identity, with regularisation selected on validation Brier error. Train, validation and test dates are shared across coins, with one-day boundary purges. The final test contains 965 coin-days across 193 dates; models are not retuned on that period.

## Results
Market-only accuracy: **53.68%**; market + sentiment: **53.06%**; training-majority baseline: **53.47%**. Brier error (lower is better): **0.2481** market-only versus **0.2491** with sentiment. The paired difference is **+0.001065**, exploratory 95% date-block interval **[−0.000159, +0.002483]**. We did not demonstrate incremental forecasting value from sentiment. The 54.40% attention-arm accuracy is descriptive, not a demonstrated edge.

Only **142 of 965** test coin-days have eligible posts. One confident miss was DOGE on 2026-10-10: the social model assigned about 33% probability of a rise, but the coin rose about 0.45%; its attached post concerned Bitcoin liquidations. Sparse coverage, retrospective ticker tags and checkpoint dependence limit interpretation. This result does not establish causality, profitability or that better data cannot help.

## What we'd do next
Collect a prospective, source-balanced archive and test once on a genuinely new future period. Hand-label post intent to separate coin-specific opinions from sector news.

## Sources
[Dataset](https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto), [FinTwitBERT](https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment), [Binance bars](https://github.com/binance/binance-spot-api-docs), scikit-learn, PyTorch, Transformers and dependencies. Hermes Agent/OpenAI assisted implementation and review. Full revisions and credits are in the notebook; the team must understand and explain the work.

**This project is a prototype and is not financial advice.**
