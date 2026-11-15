# ScamCheck reference → HypeCheck experiment map

The complete original notebook was recovered from the attached user message, not inferred from its headings. ScamCheck compares older UK SMS with 40 team-written modern US-style texts. It does not test another financial market. We separately test later dates and excluded memecoin markets, using actual prices rather than invented financial outcomes.

1. Load real messages → load original public posts and actual Binance labels.
2. Inspect old examples → inspect real historical batches, dates and estimated sentiment.
3. Stratified 80/20 random_state=42 → reproduce as a labelled diagnostic on historical training data; main financial test is chronological 80/20 with purge.
4. Always not-spam plus accuracy/precision/recall → always-up, train-majority, prevalence, coin-only and prior completed-day controls; accuracy, UP precision/recall, confusion matrices and Brier error.
5. Default word TfidfVectorizer + LogisticRegression(max_iter=1000) → same wording learner, with aggregate pretrained sentiment and coin identity fixed across comparisons.
6. Five wrong predictions → actual held-out batch errors and false-UP/missed-UP counts.
7. Modern US-style challenge → all eligible later-date established-coin batches and all eligible later-date held-out-coin batches. No forced 40-case or sentiment quota. Held-out asset mentions are excluded from supervised training.
8. One change: class_weight="balanced" → exactly the same word features and training examples, then evaluate the balanced model on both tests.
9. Alternative char_wb(2,5), sublinear_tf=True, balanced LogisticRegression(C=10,max_iter=1000) → reproduce the full trial and report its results, not just mention it.
10. Top 12 words and TF-IDF × coefficient explanations → actual fitted word contributions, with sentiment/coin contributions separated. These explain model scores, not causal price effects.
11. Optional app off during Run all → real multi-post input and a separate optional interactive app.
12. Side-by-side results and limitations → publish every model/test comparison, including negative results and class-weight tradeoffs.

The word-balanced model is predeclared as the sample-aligned readable demo, not selected for test accuracy. Financial chronology, actual labels, batch units and correlated-row weighting are intentional documented adaptations. The archive/test periods were inspected before, so this is exploratory, not an independent replication. Frozen FinTwitBERT produces language features; downstream classifiers train against market outcomes. No transformer fine-tuning or complete X-feed coverage is claimed.
