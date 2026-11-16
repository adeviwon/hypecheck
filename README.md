# HypeCheck

Can the wording and sentiment of public X posts improve next-day memecoin forecasts? In our retrospective tests, the coin-only baseline was more accurate on both later dates and coins excluded from training.

[Run in Colab](https://colab.research.google.com/github/adeviwon/hypecheck/blob/main/HypeCheck.ipynb) · [Read the technical write-up](WRITEUP.md)

## Try your own posts

- **One tweet in Colab.** Run the first two setup cells, paste the text into **Check a custom tweet**, select its coin, and run the cell. The first novel tweet may download FinTwitBERT.
- **Several tweets in Colab.** Choose **Runtime → Run all**, then edit the list in Section 9 under **Enter tweets here**. Run that cell and the scoring cell directly below it. Each list entry is rated separately. The section also demonstrates a historical same-coin batch.

The target is whether the coin closes above its open on the next complete UTC day. These are experimental estimates. The earlier [five-coin replay](https://adeviwon.github.io/hypecheck/) and [13-coin sentiment replay](https://adeviwon.github.io/hypecheck/headlines.html) show historical forecasts from separate experiments.

## Reproduce

Use Python 3.12 and Node.js. The lockfile targets Linux CPU. For macOS, install the same PyTorch release without its Linux-only `+cpu` suffix.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python verify_submission.py
```

`verify_submission.py` reconstructs the historical sentiment cache, runs the Python and JavaScript tests, and checks notebook source hashes and regeneration. See the [write-up](WRITEUP.md) for training commands, data provenance, and model details.

Nathaniel Wong, Aaden Yick, and Alastair Chu · Malvern College Hong Kong · Veritas AI International High School Hackathon 2026, AI + Finance
