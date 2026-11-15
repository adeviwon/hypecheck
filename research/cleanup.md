# Conservative dead-code cleanup

Workflow: ECC [/refactor-clean](https://github.com/affaan-m/ecc/blob/main/commands/refactor-clean.md).
The upstream command is named refactor-clean, not refactor-cleanse.

## Removed

- Twelve unused imports across eight Python source files, plus one unused import
  in the notebook generator setup cell (thirteen in total).
- The unused `real` assignment in `headline_study.py`.
- `README.template.md`: an unreferenced, obsolete research template. The current
  README is generated from the measured report, not this template.

## Preserved deliberately

- All experimental data, trained model parameters, reports and predictions.
- Historical scripts and notebooks: absence of imports does not make standalone
  entry points dead code, and these files document earlier hypotheses.
- Compatibility entry points `build_colab.py` and `make_deliverables.py`.
- `imp3.py:mats(..., fit_text=True)`: Vulture flags the unused parameter, but this
  archived script uses dynamic execution outside the tested primary path.
  No interface change or broad experimental refactor was attempted.

## Verification

- Baseline: 42 Python tests and three JavaScript checks passed before editing.
- The submission gate was rerun after each non-primary source cleanup and after
  removing the obsolete template.
- The primary notebook executed locally with all ten code cells. Its fresh
  report, prediction CSV and model JSON matched the existing artifacts
  byte-for-byte. Cached pretrained sentiment scores were used, not recomputed.
- Final `python -m unittest discover -v`: 42 tests passed.
- Final `python verify_submission.py`: 38 Python tests, three JavaScript checks,
  embedded-source parity and repeatable artifact generation passed.
- Final `ruff check . --select F401,F811,F841`: no findings.
- Local execution used Python 3.11, not a live Google Colab runtime.

No scientific conclusions, model selection, split dates, baseline definitions,
public demo assets or published accuracies were changed.
