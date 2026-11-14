"""Execute project notebooks locally; capture real outputs, fail on shell errors.
Usage: python run_nb_capture.py notebook.ipynb empty-workdir
Not a substitute for a live Google Colab test.
"""
import ast, contextlib, io, json, os, shlex, subprocess, sys, time, traceback
from pathlib import Path

def execute(nb_path, workdir):
    nb_path = Path(nb_path).resolve()
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    nb = json.loads(nb_path.read_text())
    os.chdir(workdir)
    sys.path.insert(0, str(workdir))
    g = {'__name__': '__main__'}
    start = time.monotonic()
    count = 0
    for (i, c) in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        s = ''.join(c['source'])
        if not s.strip():
            continue
        count += 1
        print(f'CELL {i}: {s.splitlines()[0][:85]}', flush=True)
        buf = io.StringIO()
        c['execution_count'] = count
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                if s.startswith('%%writefile '):
                    (first, body) = s.split('\n', 1)
                    p = Path(shlex.split(first)[1])
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(body)
                    print(f'Wrote {p}')
                elif s.startswith('!'):
                    args = shlex.split(s[1:])
                    if args[0] in ('python', 'python3'):
                        args[0] = sys.executable
                    r = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    print(r.stdout, end='')
                    r.check_returncode()
                else:
                    t = ast.parse(s)
                    if t.body and isinstance(t.body[-1], ast.Expr):
                        exec(compile(ast.fix_missing_locations(ast.Module(body=t.body[:-1], type_ignores=[])), '<cell>', 'exec'), g)
                        v = eval(compile(ast.Expression(t.body[-1].value), '<cell>', 'eval'), g)
                        if v is not None:
                            print(repr(v))
                    else:
                        exec(compile(t, '<cell>', 'exec'), g)
            c['outputs'] = [{'output_type': 'stream', 'name': 'stdout', 'text': buf.getvalue().splitlines(keepends=True)}]
        except Exception:
            c['outputs'] = [{'output_type': 'stream', 'name': 'stdout', 'text': buf.getvalue().splitlines(keepends=True)}, {'output_type': 'error', 'ename': type(sys.exception()).__name__, 'evalue': str(sys.exception()), 'traceback': traceback.format_exc().splitlines()}]
            nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
            print(buf.getvalue(), flush=True)
            raise
        nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
        print(buf.getvalue(), end='', flush=True)
    summary = {'code_cells': count, 'seconds': round(time.monotonic() - start, 2), 'python': sys.version, 'status': 'passed'}
    (workdir / 'notebook-execution.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
if __name__ == '__main__':
    execute(sys.argv[1], sys.argv[2])
