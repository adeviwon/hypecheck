# HypeCheck — test the crowd
**Track:** AI + Finance
**Team:** Nathaniel Wong (model), Aaden Yick (data), Alastair Chu (demo)

## The problem
Memecoin traders encounter confident, bullish public posts without knowing whether that language predicts future prices. We investigated whether a batch of public X posts and its estimated sentiment can help predict the next complete UTC day’s price direction.

## What we built
A runnable Colab notebook with a batch-post/CSV interface. Users select a coin and supply public posts; HypeCheck reports estimated sample sentiment, a next-day direction estimate, word-level model explanations and evidence warnings. It does not collect a live X feed, represent all public sentiment or estimate how much a post caused prices to move.

## How it works
We use an archived public crypto-post dataset and real Binance daily prices across 13 memecoins. Each snapshot contains same-coin posts available in the preceding 24 hours. We reject invalid supplied timestamps and remove duplicate text; missing timestamps remain unknown. Training uses **991 snapshots covering 556 coin-days**. Repeated snapshots share a coin-day’s total evaluation weight, so frequently posted days do not dominate.

Frozen, pretrained FinTwitBERT estimates language sentiment; it is not fine-tuned. We train logistic regression using word TF-IDF, sentiment and coin identity. We then change only the class weights to balanced, and separately test character features. Earlier dates train the models; a one-day purge separates later tests. Eight coins are excluded from training for the unseen-coin test. We use simple models because the sample is small and word contributions are inspectable.

## Results
| Test | Balanced word model | Coin-only baseline | Always up | Previous completed day |
|---|---:|---:|---:|---:|
| Later dates: 127 coin-days | 36.75% | 51.18% | 50.39% | 47.24% |
| Unseen coins: 77 coin-days | 43.83% | 55.84% | 44.16% | 55.84% |

**The wording/sentiment models did not demonstrate a forecasting advantage.** Neither balancing nor character features beat the coin-only control on either test. The notebook includes actual errors: strongly bullish wording can precede a falling price. A language sentiment score is not the probability of a price rise. Sparse, biased data and market shifts are plausible limitations, not established explanations for each error.

The archive dates were previously explored, so these are exploratory results, not independent confirmation. Forecast probabilities are not independently calibrated. No profitable strategy, causal effect or user benefit has been validated; these results do not justify trading on the model. Execution was verified on a server, not an actual Google Colab runtime.

## What we would do next
Freeze the protocol and evaluate a genuinely new timestamped sample prospectively. Test whether users correctly distinguish bullish language, an uncertain forecast and the evidence needed to trust it.

## Sources
[Public post dataset](https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto); [FinTwitBERT](https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment); [Binance market data](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#klinecandlestick-data); [scikit-learn](https://scikit-learn.org/), PyTorch and Transformers; Veritas AI Finance brief and participant reference notebook; [Hermes Agent](https://github.com/NousResearch/hermes-agent) and OpenAI coding assistance. Exact source revisions, hashes and dependencies accompany the notebook.

**This project is a prototype and is not financial advice.**
