"""Submission gate: tests, embedded source parity, saved outputs and generator idempotence."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parent

def run(args):
    subprocess.run(args, cwd=ROOT, check=True)

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

run([sys.executable, '-m', 'unittest', '-v', 'test_pipeline.py', 'test_web.py'])
run(['node', 'docs/app.js', '--self-test'])
run(['node', 'test_dom.mjs'])
nb = json.loads((ROOT/'HypeCheck.ipynb').read_text())
code = [c for c in nb['cells'] if c['cell_type']=='code' and ''.join(c['source']).strip()]
assert len(code) == 12, len(code)
assert all(c.get('execution_count') and c.get('outputs') for c in code)
assert not any(o.get('output_type')=='error' for c in code for o in c['outputs'])
for name in ['pipeline.py', 'postmodel.py', 'test_pipeline.py']:
    embedded = next(''.join(c['source']).split('\n',1)[1] for c in code if ''.join(c['source']).startswith('%%writefile '+name+'\n'))
    assert embedded.rstrip() == (ROOT/name).read_text().rstrip(), name
paths = [ROOT/n for n in ['HypeCheck.ipynb','README.md','requirements.lock.txt']]
before = {p.name:digest(p) for p in paths}
run([sys.executable, 'make_deliverables.py'])
assert before == {p.name:digest(p) for p in paths}, 'Generator changed delivered files or lockfile'
print('Submission gate passed: 12 output-bearing code cells, embedded-source parity, regeneration idempotence.')
