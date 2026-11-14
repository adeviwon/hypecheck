# HypeCheck - Signal or Noise?

This project is a prototype and is not financial advice.

We test whether prior-day headline-style X posts add information about next-day open-to-close direction for DOGE, SHIB, PEPE, BONK and FLOKI.

## Results

Always up: 46.53%
Previous-day direction: 49.33%
Sentiment + coin: 52.95%
Training majority: 53.47%
Market only: 53.68%
Market + sentiment: 53.06%

The sentiment model does not beat training majority; adding sentiment does not establish improvement beyond market history. No text kind demonstrated reliable improvement. Sparse archive coverage and exploratory reuse limit inference.

No trading, profit or causal claim is justified. This does not prove prediction is impossible.

Full method, paired intervals, sources and credits: README.md. Measured report: docs/headline-report.json. Colab: HypeCheck.ipynb.
