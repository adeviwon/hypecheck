"""Regenerate README and notebook without overwriting the lockfile or execution evidence."""
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parent
report = json.loads((ROOT/'docs/report.json').read_text())
rows = ['| Method | Accuracy | Brier error |', '|---|---:|---:|']
rows += [f"| {m['name']} | {m['accuracy']:.2%} | {m['brier']:.4f} |" for m in report['metrics']]
text = (ROOT/'README.template.md').read_text().replace('@@RESULTS@@', '\n'.join(rows))
(ROOT/'README.md').write_text(text)
subprocess.run([sys.executable, str(ROOT/'build_colab.py')], check=True)
