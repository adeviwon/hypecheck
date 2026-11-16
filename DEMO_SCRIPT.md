# HypeCheck — two-minute recording script

Record the real notebook/app, not a mock-up. Run all cells before recording; the input app opens automatically in Section 9. Use a real public batch or the notebook’s first eligible historical batch; label a historical replay explicitly. Do not claim newly pasted text has a known future outcome. Keep the final recording under two minutes and check its actual duration.

| Time | Show | Suggested narration |
|---|---|---|
| 0:00–0:15 | Notebook title and question | Memecoin posts often sound confident. We asked whether their wording and estimated sentiment actually help predict the next day’s direction. |
| 0:15–0:40 | Select coin, paste real public batch/CSV, click Analyse batch | Users supply a batch about one coin. The app checks the input and runs a pretrained sentiment model, followed by our trained direction classifier. This is a user-provided sample, not all of X. |
| 0:40–1:00 | Sentiment, forecast dates and evidence panels | These are different outputs: what the language sounds like, what the forecast model estimates, and whether our tests support trusting it. We do not estimate a tweet’s causal price impact. |
| 1:00–1:15 | Word-contribution panel and saved failure | These words influenced this model. That does not make them economic laws. Our failure analysis shows bullish language can precede a falling price. |
| 1:15–1:40 | Results table | We trained word-based logistic regression, changed class weights and tried character features. The balanced model scored 36.75 percent on later dates and 43.83 percent on unseen coins, below coin-only baselines of 51.18 and 55.84 percent. |
| 1:40–1:55 | Limitations and next steps | We found no forecasting edge in this sparse archive. The dates were previously explored. Next we would freeze the method and test genuinely new posts prospectively. This is a prototype, not financial advice. |

Upload the recording as unlisted on YouTube or viewable by anyone with the link on Drive. Test it logged out, including audio. A script is not a recorded video. If inference is slow, use a clearly labelled edit rather than suggesting it completed instantly.
