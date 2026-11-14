"""Verify current headline study while preserving legacy research/security gates."""
import ast,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def run(args):subprocess.run(args,cwd=ROOT,check=True)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

run([sys.executable,'-m','unittest','-v','test_pipeline.py','test_web.py','test_headline_study.py','test_headline_artifacts.py','test_headline_http.py'])
run(['node','docs/app.js','--self-test']);run(['node','test_dom.mjs']);run(['node','test_headline_dom.cjs'])
nb=json.loads((ROOT/'HypeCheck.ipynb').read_text());code=[c for c in nb['cells'] if c['cell_type']=='code' and ''.join(c['source']).strip()]
assert len(code)==10,len(code)
assert all(c.get('execution_count') and c.get('outputs') for c in code)
assert not any(o.get('output_type')=='error' for c in code for o in c['outputs'])
source=next(''.join(c['source']) for c in code if ''.join(c['source']).startswith('SOURCES = '))
embedded=ast.literal_eval(ast.parse(source).body[0].value)
for name,text in embedded.items():assert text.rstrip()==(ROOT/name).read_text().rstrip(),name
paths=[ROOT/n for n in ['HypeCheck.ipynb','README.md','requirements.lock.txt','docs/headline-predictions.json']]
before={p.name:digest(p) for p in paths}
run([sys.executable,'make_deliverables.py']);run([sys.executable,'make_deliverables.py'])
assert before=={p.name:digest(p) for p in paths},'Generator changed delivered artifacts or lockfile'
print('Submission gate passed: 10 output-bearing code cells, source parity, safe model parity, regenerated artifacts unchanged.')
