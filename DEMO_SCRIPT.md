# HypeCheck - two-minute recording script

Record the real live page and the real notebook, not a mock-up. Open
https://adeviwon.github.io/hypecheck/live.html in a normal browser window, paste a real public post, and let the forecast render on camera. Keep the final recording under two minutes and check its actual duration.

| Time | Show | Suggested narration |
|---|---|---|
| 0:00-0:12 | Live page, empty box | Memecoin traders see confident posts all day. We tested whether that wording actually helps predict the next day's direction. |
| 0:12-0:30 | Paste a real public post into **Paste tweet here**, click **Get forecast** | One box. I paste a public post and the trained model runs in my browser - nothing is uploaded, there is no server. |
| 0:30-0:50 | The rendered result | It reads the coin from the cashtag, gives the estimated chance of an up day, the exact UTC window it is forecasting, and the words that moved the score. |
| 0:50-1:05 | Coin-only baseline line and the evidence line | That is the honest part: the same pipeline without the wording scores higher than the model with it. 41 percent against 51 on later dates. So we tell users not to trust the number. |
| 1:05-1:25 | Scroll to "What this page runs, exactly" | The page runs the wording and coin model in the browser. The full submitted model adds a frozen neural sentiment block, and that needs the Colab - it is all in the notebook with its error analysis. |
| 1:25-1:45 | Notebook results table | We trained word TF-IDF logistic regression, changed class weights, and tried character features. Balanced model: 36.75 percent later dates, 43.83 percent unseen coins, against coin-only 51.18 and 55.84. |
| 1:45-1:58 | Limitations cell | No forecasting edge in this sparse archive, the dates were examined before, and we estimate no causal effect. This is a prototype, not financial advice. |

Do not claim the live forecast was "correct" - its outcome is in the future. Do not claim a trading edge. If inference is slow on the recording machine, use a clearly labelled edit rather than implying it was instant.

Upload the recording as unlisted on YouTube or as a Drive link viewable by anyone. Test it logged out, including audio. A script is not a recorded video.
