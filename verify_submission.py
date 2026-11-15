"""Verify the current study, judge-facing notebook and preserved security gates."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import colab_setup

ROOT = Path(__file__).resolve().parent


def run(args):
    subprocess.run(args, cwd=ROOT, check=True)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


run([sys.executable, '-m', 'unittest', '-v', 'test_pipeline.py', 'test_web.py',
     'test_headline_study.py', 'test_headline_artifacts.py', 'test_headline_http.py',
     'test_expanded_inputs.py', 'test_colab_setup.py', 'test_notebook_readability.py'])
run(['node', 'docs/app.js', '--self-test'])
run(['node', 'test_dom.mjs'])
run(['node', 'test_headline_dom.cjs'])
nb = json.loads((ROOT / 'HypeCheck.ipynb').read_text())
code = [cell for cell in nb['cells'] if cell['cell_type'] == 'code' and ''.join(cell['source']).strip()]
metadata = nb['metadata']['hypecheck']
assert len(code) == metadata['expected_code_cells'], len(code)
assert all(cell.get('execution_count') and cell.get('outputs') for cell in code)
assert not any(output.get('output_type') == 'error' for cell in code for output in cell['outputs'])
# Moving source out of the notebook must not remove source/input parity checks.
for name, expected in metadata['source_sha256'].items():
    assert digest(ROOT / name) == expected, name
    assert colab_setup.FILE_DIGESTS[name] == expected, name
for name, expected in metadata['bootstrap_sha256'].items():
    assert digest(ROOT / name) == expected, name
loader = next(''.join(cell['source']) for cell in code if 'VERIFIED_FILES = ' in ''.join(cell['source']))
assignments = {node.targets[0].id: ast.literal_eval(node.value)
               for node in ast.parse(loader).body
               if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
assert assignments['REVISION'] == metadata['bootstrap_revision']
assert assignments['VERIFIED_FILES'] == metadata['bootstrap_sha256']
paths = [ROOT / name for name in ['HypeCheck.ipynb', 'README.md', 'requirements.lock.txt', 'docs/headline-predictions.json']]
before = {path.name: digest(path) for path in paths}
run([sys.executable, 'make_deliverables.py'])
run([sys.executable, 'make_deliverables.py'])
assert before == {path.name: digest(path) for path in paths}, 'Generator changed delivered artifacts or lockfile'
print('Submission gate passed: real rich notebook outputs, pinned source parity, safe model parity and repeatable artifact generation.')
