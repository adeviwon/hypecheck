"""Verify the current study, readable notebook and preserved security gates."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

if sys.version_info < (3, 12):
    raise SystemExit('Use Python 3.12+ with pinned requirements; tested on Python 3.12.14.')

import colab_setup
from prepare_verification import prepare

ROOT = Path(__file__).resolve().parent


def run(args):
    subprocess.run(args, cwd=ROOT, check=True)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


prepare()

run([sys.executable, '-m', 'unittest', '-v', 'test_pipeline.py', 'test_web.py',
     'test_headline_study.py', 'test_headline_artifacts.py', 'test_headline_http.py',
     'test_prepare_verification.py', 'test_integration_readiness.py', 'test_expanded_inputs.py', 'test_colab_setup.py', 'test_notebook_readability.py','test_tweet_forecast.py','test_crowd_snapshot.py','test_sample_pipeline.py','test_crowd_app.py'])
run(['node', 'docs/app.js', '--self-test'])
run(['node', 'test_dom.mjs'])
run(['node', 'test_headline_dom.cjs'])
run(['node', 'test_tweet_ui.cjs'])
nb = json.loads((ROOT / 'HypeCheck.ipynb').read_text())
code = [cell for cell in nb['cells'] if cell['cell_type'] == 'code' and ''.join(cell['source']).strip()]
metadata = nb['metadata']['hypecheck']
assert len(code) == metadata['expected_code_cells'], len(code)
assert all(cell.get('execution_count') and cell.get('outputs') for cell in code)
assert not any(output.get('output_type') == 'error' for cell in code for output in cell['outputs'])
# Union of both source, input, bootstrap and readability guarantees.
for name, expected in colab_setup.FILE_DIGESTS.items():
    assert digest(ROOT / name) == expected, name
for name, expected in metadata['source_sha256'].items():
    assert digest(ROOT / name) == expected, name
    assert json.loads((ROOT / "crowd-source.json").read_text())["files"][name] == expected,name
config = json.loads((ROOT / 'colab-loader.json').read_text())
assert config['revision'] == metadata['bootstrap_revision']
assert config['sha256'] == metadata['bootstrap_sha256']['colab_setup.py']
assert config['presentation_sha256'] == metadata['bootstrap_sha256']['notebook_support.py']
for name, expected in metadata['bootstrap_sha256'].items():
    assert digest(ROOT / name) == expected, name
assert any("model1.fit(X_train,train.actual_up" in "".join(c["source"]) for c in code)
assert any("model2.fit(X_train,train.actual_up" in "".join(c["source"]) for c in code)
assert not any("EMBEDDED_INPUTS =" in "".join(c["source"]) or "SOURCES =" in "".join(c["source"]) for c in code)
paths = [ROOT / name for name in ['HypeCheck.ipynb', 'README.md', 'requirements.lock.txt', 'docs/headline-predictions.json']]
before = {path.name: digest(path) for path in paths}
run([sys.executable, 'make_deliverables.py'])
run([sys.executable, 'make_deliverables.py'])
assert before == {path.name: digest(path) for path in paths}, 'Generator changed delivered artifacts or lockfile'
print('Submission gate passed: real rich outputs, all-source/input parity, verified compact loader and repeatable artifact generation.')
