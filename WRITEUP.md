# HypeCheck - Signal or Noise?

This project is a prototype and is not financial advice.

We test whether prior-day headline-style X posts add information about next-day open-to-close direction for 13 memecoins.

## Results

Always up: 46.07%
Previous-day direction: 48.51%
Sentiment + coin: 53.09%
Training majority: 53.93%
Market only: 54.40%
Market + sentiment: 53.61%

The sentiment model does not beat training majority; adding sentiment does not establish improvement beyond market history. No text kind demonstrated reliable improvement. Sparse archive coverage and exploratory reuse limit inference.

No trading, profit or causal claim is justified. This does not prove prediction is impossible.

Full method, paired intervals, sources and credits: README.md. Measured report: docs/headline-report.json. Colab: HypeCheck.ipynb.

Expanded corpus: 1773 original posts; 13 memecoins; 2509 held-out coin-days. Natural sentiment distribution; no fixed 15/25 split.
