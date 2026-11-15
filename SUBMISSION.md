# HypeCheck — current submission

Track: AI + Finance

Code: https://colab.research.google.com/github/adeviwon/hypecheck/blob/main/HypeCheck.ipynb

Demo: https://adeviwon.github.io/hypecheck/predict.html

Write-up: WRITEUP.md

Nathaniel Wong — model; Aaden Yick — data; Alastair Chu — demo. Team member emails and school details must be confirmed in the submission form.

The model is trained from raw public-post wording against actual next-UTC-day Binance outcomes. Section 5 visibly trains; Section 9 accepts a tweet and coin. The forecast is not a causal impact estimate. No live X feed is connected. Training and notebook execution were verified on the server, not an actual Colab runtime.

Held-out coin-day-weighted accuracy: tweet-only 49.28%; tweet plus prior context 45.02%; training majority 51.60%. No demonstrated trading edge. Return error also fails its constant baseline. Full numbers and limitations: docs/tweet-forecast-report.json.

This project is a prototype and is not financial advice.
