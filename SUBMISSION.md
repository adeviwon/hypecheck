# HypeCheck — Signal or Noise?

**Track:** AI + Finance, adapted from brief §4.2.4.

**Code:** https://colab.research.google.com/github/adeviwon/hypecheck/blob/main/HypeCheck.ipynb
**Write-up:** https://github.com/adeviwon/hypecheck
**Demo:** https://adeviwon.github.io/hypecheck/headlines.html

**Problem:** A beginner investor sees bullish memecoin posts and wonders whether they forecast tomorrow. We test rather than assume that claim.

**Approach:** Prior-day original X-post sentiment for DOGE, SHIB, PEPE, BONK and FLOKI. Frozen pretrained FinTwitBERT features plus logistic regression; chronological splits with purge gaps, both requested baselines, coin/market ablations and text-kind comparisons.

**Results:** Read the generated README and `docs/headline-report.json`; those are the source of truth. Do not copy an old headline number from a different model or split. Beating weak baselines on point accuracy does not demonstrate useful sentiment information. The study is exploratory on a reused period, not independent confirmation.

**Investor boundary:** This project is a prototype and is not financial advice. No trade, profit, or causal-effect claim.

**Team:** Nathaniel Wong (model), Aaden Yick (data), Alastair Chu (demo and write-up), Malvern College Hong Kong. Confirm each member's exact email/name/school in the form. Email addresses and a demonstration video have not been supplied here.

**Execution evidence:** Notebook outputs were captured during a real clean-working-directory server run. Actual Google Colab execution is not claimed without Colab runtime evidence. The notebook installs its dependencies conditionally on Colab, fetches verified public inputs, and retrains the direction models.

**Credits:** See README and notebook sources. AI coding assistance disclosed. Every team member must understand and explain the model and results.
