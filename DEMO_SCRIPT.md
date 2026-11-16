# HypeCheck - two-minute recording script

Record the real Colab notebook, not a mock-up. Open
https://colab.research.google.com/github/adeviwon/hypecheck/blob/main/HypeCheck.ipynb
and press Runtime -> Run all before you start recording, so the outputs are already on screen. Keep the final recording under two minutes and check its actual duration.

For the full neural-sentiment model, run the notebook; its Section 9 input app opens automatically after Run all.

| Time | Show | Suggested narration |
|---|---|---|
| 0:00-0:12 | Notebook title | Memecoin traders see confident posts all day. We tested whether that wording actually helps predict the next day's direction. |
| 0:12-0:30 | Section 5, the training cell | This is the model being trained: word features, sentiment, coin identity, fitted on real next-day market outcomes. No invented labels. |
| 0:30-0:50 | The results tables | On later dates and on coins it never saw, the model loses to a coin-only baseline: 36.75 percent against 51.18. We report that rather than dress it up. |
| 0:50-1:05 | The mistake analysis | These are the model's real wrong calls - strongly bullish posts followed by falling prices. |
| 1:05-1:35 | Section 9, the "Enter tweets here" block, then run that cell | Forty real public posts held out of training. The model rates every one, and the table shows what it got right: 14 of 40. Saying "up" every time would have scored 16, and coin-only 22. |
| 1:35-1:50 | Delete the list, paste two or three of your own posts, run the cell | You can replace the list with your own posts. They get rated the same way, but they are not scored, because their outcome has not happened yet. |
| 1:50-1:58 | Limitations cell | Sparse archive, reused dates, no calibrated probabilities and no causal claim. This is a prototype, not financial advice. |

Do not claim the model has a trading edge, and do not claim a forecast on your own pasted posts was correct - its outcome is in the future. If inference is slow on the recording machine, use a clearly labelled edit rather than implying it was instant.

Upload the recording as unlisted on YouTube or as a Drive link viewable by anyone. Test it logged out, including audio. A script is not a recorded video.
