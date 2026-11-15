# HypeCheck — current crowd-batch submission

Track: AI + Finance

Notebook: https://colab.research.google.com/github/adeviwon/hypecheck/blob/main/HypeCheck.ipynb

Write-up: WRITEUP.md

Nathaniel Wong — model; Aaden Yick — data; Alastair Chu — demo. Confirm individual emails and school details in the submission form.

Use Runtime → Run all. Sections 5 and 8 visibly train the actual word, balanced and character-model sequence. Section 9 explains a real batch and offers RUN_APP=True for new public posts or CSV. The app derives sentiment from supplied content, then predicts next-full-UTC-day direction. No complete X-feed coverage or causal impact is claimed.

Our method: inspect → baseline → Model 1 word TF-IDF → errors → later-time/unseen-coin tests → Model 2 balanced → char-ngram trial → word-contribution demo → full results/limits. No synthetic price labels or forced quotas.

Actual held-out Model 2 accuracy: 36.75% later-time vs 51.18% coin-only; 43.83% unseen coins vs 55.84% coin-only. No demonstrated forecasting edge. Details and all alternatives: docs/crowd-report.json.

Training and notebook execution were verified on the server, not an actual Colab runtime. These archive dates were previously examined; results are exploratory.

This project is a prototype and is not financial advice.
