# HypeCheck - two-minute recording script

Record the real Colab notebook, not a mock-up. Open
https://colab.research.google.com/github/adeviwon/hypecheck/blob/main/HypeCheck.ipynb
and press Runtime -> Run all before you start recording, so the outputs are already on screen. Keep the final recording under two minutes and check its actual duration.

Run all to train the models and produce the default 40-post scorecard. The quick form near the top scores one custom tweet after the two setup cells.

| Time | Show | Suggested narration |
|---|---|---|
| 0:00-0:12 | Notebook title | Memecoin traders see confident posts all day. We tested whether that wording actually helps predict the next day's direction. |
| 0:12-0:30 | Section 5, the training cell | This is the model being trained: word features, sentiment, coin identity, fitted on real next-day market outcomes. No invented labels. |
| 0:30-0:50 | The results tables | On later dates and on coins it never saw, the model loses to a coin-only baseline: 36.75 percent against 51.18. We report that rather than dress it up. |
| 0:50-1:05 | The mistake analysis | These are the model's real wrong calls - strongly bullish posts followed by falling prices. |
| 1:05-1:35 | Section 9, the "Enter tweets here" block, and its scoring cell | Forty real public posts held out of training. The model rates every one, and the table shows what it got right: 14 of 40. Saying "up" every time would have scored 16, and coin-only 22. |
| 1:35-1:50 | Replace the list with two or three of your posts, then rerun the list and scoring cells | You can replace the list with your own posts. They get rated the same way, but they are not scored, because their outcome has not happened yet. |
| 1:50-1:58 | Limitations cell | Sparse archive, reused dates, no calibrated probabilities and no causal claim. This is a prototype, not financial advice. |

Do not claim the model has a trading edge, and do not claim a forecast on your own pasted posts was correct - its outcome is in the future. If inference is slow on the recording machine, use a clearly labelled edit rather than implying it was instant.

Upload the recording as unlisted on YouTube or as a Drive link viewable by anyone. Test it logged out, including audio. A script is not a recorded video.
