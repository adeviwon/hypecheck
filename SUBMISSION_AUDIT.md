# Final submission audit — current crowd-batch experiment

Source: [official participant guide](https://pub-4ab5e9a15dfa442f83f50fdee0d26970.r2.dev/hackathon-guide.pdf), sections 8–10. This audit supersedes the old daily-study audit.

| Requirement | Evidence | Status |
|---|---|---|
| Working AI input-to-output | HypeCheck.ipynb Section 9; real neural inference and trained batch classifier | Server verified; actual Colab runtime remains unverified |
| Fair comparisons and failures | Chronological purge, unseen-coin cohort, baseline tables, actual error analysis | Present; reused archive explicitly exploratory |
| Judge-readable notebook | Plain-English findings, named columns, percentages and rose/fell outcomes | Present with saved outputs |
| Short write-up in organiser format | WRITEUP.md, generated from submission_writeup.md and measured report | Present |
| Dataset, model and code credits | Notebook Sources and WRITEUP.md | Present |
| Financial prototype disclaimer | Notebook, README, write-up, packet | Present |
| Publicly accessible code/write-up | Anonymous link checks recorded in artifacts/final-public-checks.json | Check evidence file |
| Demo video up to two minutes | DEMO_SCRIPT.md is preparation only | Outstanding: recording and accessible URL required |
| Team details and eligibility | Names/roles in packet; registration details must be confirmed privately | Team action required |
| Build-window and AI-tool rules | Commit history and AI credits; team must confirm originality and understanding | Not independently certified |
| Submission confirmation | Official form receipt | Not submitted by this agent |

## Evidence boundary
Current machine-readable execution/test evidence: artifacts/crowd-verification.json. Saved notebook code cells were executed on the server, not Google Colab. Dependencies were not independently security-audited during finalisation. No calibrated trading edge, causal price effect or profitability is claimed. Freezing documentation does not create independent scientific confirmation.

The strengths are reproducible evaluation and clear negative findings. The limits are sparse data, exploratory reused dates, no demonstrated trading value, and no validated real-user impact.
