# HypeCheck - test the crowd, not just a tweet

**Track:** Finance
**Team:** Malvern Little Geniuses - Nathaniel Wong (model), Aaden Yick (data), Alastair Chu (demo)

## The problem

In 2026 memecoins move on attention, and traders constantly see public X posts claiming a coin is about to run. These posts are hard to judge: they are written confidently, they cannot be verified, and a new trader has no way to tell language that carries information from language that is only loud. So we investigated whether the wording of public posts, and public sentiment about a coin, is useful information for predicting that coin's next-day price direction - and reported the answer honestly, even though it turned out to be "not really".

## What we built

HypeCheck is a research prototype that takes a batch of public X posts about one memecoin and returns three separate things:

1. **Crowd Pulse** - the estimated language sentiment of the batch (bullish, bearish, neutral). This describes how the posts read, not whether the price will rise.
2. **Market Forecast** - an estimated probability that the coin's next complete UTC day closes above its open, with the exact UTC start and end of that day.
3. **Evidence Check** - the input audit, the coin-only baseline for comparison, and a plain statement that our own tests found no forecasting edge.

A user runs our supplied Google Colab notebook, pastes public posts (or CSV) for a coin, and reads the result. The notebook ships fully executed with saved outputs, so a judge can see every number without running anything.

The prototype covers 13 memecoins. Five are used for training - **DOGE, SHIB, PEPE, BONK, FLOKI**. Eight are held out completely - **WIF, TRUMP, NEIRO, PENGU, PNUT, TURBO, BOME, MEME** - so we can test coins the model has never seen.

### What we added

Our own idea was to test whether social-media wording carries information beyond the market data itself. We added two tests that go past a standard within-sample evaluation: a **later-time** test (train on earlier dates, test on the most recent ones with a purge gap) and an **unseen-coin** test (coins excluded from training entirely). We also built a real input-to-output app rather than a notebook that only prints numbers, and we inspect the model's actual mistakes instead of only its accuracy.

## How it works

### Data

- **Market data:** actual daily Binance spot bars, frozen and hash-checked, so results are reproducible from the same bytes.
- **Posts:** an archived public crypto-financial post dataset ([financial-tweets-crypto](https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto)), pinned to one revision.
- **Batch construction:** each example is one coin's posts available in the 24 hours before a specific UTC cutoff. This produced **991 training snapshots covering 556 distinct coin-days** (3,032 posts, 23 Mar 2024 - 2 Dec 2025), **155 snapshots / 127 coin-days** in the later-time test (222 posts), and **99 snapshots / 77 coin-days** in the unseen-coin test (146 posts).
- **Cleaning:** timestamps normalised to UTC; duplicate post text removed so a repeated post cannot inflate the count; posts outside the 24-hour window excluded; invalid supplied timestamps rejected instead of being silently treated as missing. Posts with no usable timestamp stay marked unknown rather than assumed neutral.
- **No leakage:** features use only information available before the cutoff; one day is removed at each split boundary so a label cannot cross from training into test; multi-coin posts that mention held-out assets are removed from the training cohort.
- **Weighting:** several snapshots can share one coin-day outcome, so each coin-day carries total weight one. A busy posting day cannot outvote a quiet one.

### Model

- **Sentiment:** the pretrained financial language model **FinTwitBERT** estimates each post as bullish, bearish or neutral. It is used **frozen** - we did not fine-tune it. Its scores are model estimates, not human labels.
- **Prediction:** logistic regression over word TF-IDF features plus the sentiment summary and coin identity. We chose logistic regression because it is honest with a sparse archive and because its word coefficients can be read and explained.
- **Three variants, one change at a time:** Model 1 is the default word model. Model 2 changes **only** the class weights to balanced. A third trial replaces words with character n-grams (char_wb 2-5, sublinear TF, balanced, C=10). Sentiment and coin features are identical across all three, so any difference comes from the text representation.
- **Target:** the next complete UTC day after the cutoff, close greater than open. Flat counts as not up. This is not a rolling 24 hours and not a same-day move.

## Results

All figures are coin-day weighted, on data the model never trained on.

| Method | Later-time (127 coin-days) | Unseen coins (77 coin-days) |
|---|---:|---:|
| Model 1 - word TF-IDF | 36.09% | 49.03% |
| **Model 2 - balanced word model** | **36.75%** | **43.83%** |
| Character n-gram trial | 40.03% | 45.13% |
| Coin-only baseline | 51.18% | 55.84% |
| Always up | 50.39% | 44.16% |
| Training majority | 49.61% | 55.84% |
| Previous completed day | 47.24% | 55.84% |

Probability error (Brier, lower is better): Model 2 scored **0.2768** later-time and **0.2662** on unseen coins, against the coin-only baseline's **0.2532** and **0.2484**.

**Our baseline scored higher than our model.** The wording and sentiment models did not beat knowing the coin alone in either test. In the later-time test the balanced word model was 14.4 accuracy points *worse* than the coin-only baseline (95% interval -23.8 to -3.9 points); on unseen coins it was 12.0 points worse (interval -24.1 to +1.5). Balancing the classes did not create an edge, and neither did character features.

We also ran a stratified random 80/20 split, which scores 66.77% for the balanced model. We deliberately do **not** headline that number: related snapshots and the same market days sit on both sides of a random split, so it measures memorisation of the archive rather than forecasting. It is exactly the kind of result that looks too good to be true.

### Where it fails, and why we think so

- The notebook prints the model's **actual wrong calls**, with the post text, the model's probability and the words that drove it. We find strongly bullish batches followed by falling prices. A high sentiment score is not a high probability of a rise.
- **No forecasting edge:** text features added nothing over coin identity in either held-out test, and made the probability estimates slightly worse.
- **Balanced weights trade one error for another.** They raise how often the model catches UP days, but also raise false alarms and lower accuracy. They are not a fix.
- **The archive is sparse and concentrated.** In the later-time test 76 of 155 snapshots are DOGE, 45 are PEPE and only 7 are SHIB. This is a sample of public posts, not all of X.
- **Our test dates had been examined before.** They are exploratory, not independent confirmation, and the notebook says so.
- **Probabilities are not independently calibrated**, and we estimate no causal effect - the model does not tell you how much a post moved a price.
- Execution was verified on a server, not inside an actual Google Colab runtime. We state that rather than imply otherwise.

## What we would do next

1. **Collect a genuinely new timestamped sample** and evaluate this frozen protocol on it prospectively, so the test is no longer a re-used archive.
2. **Test whether users understand the evidence warnings** - specifically whether they distinguish bullish language, an uncertain forecast, and the evidence needed to trust it.

With more time we would also increase post coverage across the full period, since the binding limit is the data rather than the model.

## Sources

- [financial-tweets-crypto](https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto) - archived public crypto-financial posts; pinned revision, hash-checked.
- [FinTwitBERT sentiment model](https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment) - pretrained, used frozen, not fine-tuned.
- [Binance spot market data](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#klinecandlestick-data) - actual daily bars behind the price labels.
- [scikit-learn](https://scikit-learn.org/), PyTorch and Hugging Face Transformers - modelling and pretrained inference.
- Veritas AI Finance brief and the participant reference notebook, credited as a methodological source.
- AI tools: [Hermes Agent](https://github.com/NousResearch/hermes-agent) (Nous Research) with OpenAI models, used as coding assistance. The team understands and can explain every part of the project.

**This project is a prototype and is not financial advice.**
