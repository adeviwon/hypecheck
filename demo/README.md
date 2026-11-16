# HypeCheck demo inputs

Two verified demo paths. Both run the real trained model and real pretrained sentiment inference - nothing is mocked.

## 1. Live batch (`live-batch.txt`)
Paste the contents into the app textarea, choose the matching coin, leave **UTC cutoff blank**, click **Analyse batch**. The cutoff becomes the actual current time, so the forecast target is the next complete UTC day - an outcome nobody can know yet. Use this to show the interface, the three panels and the word explanations.

These are real archived public posts supplied as a demonstration batch. They are labelled as an archive sample, not a live feed, and not as proof the forecast was correct.

## 2. Historical replay (`historical-batch.csv`)
Paste as **CSV text** with the matching coin and its UTC cutoff. This snapshot outcome is already known, so you can reveal whether the forecast matched the market - the notebook does this automatically in Section 9.

| Input | Coin | UTC cutoff | Target day (UTC) | Model P(UP) | Direction | Actual |
|---|---|---|---:|---:|---|---|
| historical-batch.csv | PEPE | 2026-04-20T20:54:44+00:00 | 21 Apr 2026 | 57.95% | UP | rose |
| live-batch.txt | PEPE | blank (current time) | next complete UTC day | see the app | see the app | unknown by design |

Show both: the replay proves the pipeline produces a real, checkable forecast; the live batch proves a user can enter new posts and get a result. Neither shows a forecasting edge, and the notebook says so.

**This project is a prototype and is not financial advice.**
