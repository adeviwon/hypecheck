"""Execute a notebook like Colab does, capturing stdout and the last expression's value back into it."""
import ast, io, json, os, sys, time
from contextlib import redirect_stdout
from pathlib import Path

nb_path = Path(sys.argv[1]).resolve()
workdir = Path(sys.argv[2]).resolve()
workdir.mkdir(parents=True, exist_ok=True)
nb = json.loads(nb_path.read_text())
os.environ["PATH"] = "/home/ubuntu/Projects/hypecheck/.venv/bin:" + os.environ.get("PATH", "")
os.chdir(workdir)
g = {"__name__": "__main__", "__file__": str(nb_path)}
counter = 0

def to_lines(text):
    return text.splitlines(keepends=True) or [""]

def run_source(src):
    tree = ast.parse(src)
    buf = io.StringIO()
    result = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        head = ast.Module(body=tree.body[:-1], type_ignores=[])
        ast.fix_missing_locations(head)
        with redirect_stdout(buf):
            exec(compile(head, "<cell>", "exec"), g)
        expr = ast.Expression(body=tree.body[-1].value)
        ast.fix_missing_locations(expr)
        with redirect_stdout(buf):          # the last expression can print too
            result = eval(compile(expr, "<cell>", "eval"), g)
    else:
        with redirect_stdout(buf):
            exec(compile(tree, "<cell>", "exec"), g)
    return buf.getvalue(), result

def fmt(result):
    import numpy as _np
    import pandas as _pd
    if result is None:
        return None
    data = {}
    if isinstance(result, (_pd.DataFrame, _pd.Series)):
        data["text/plain"] = repr(result)
        try:
            data["text/html"] = result.to_html()
        except Exception:
            pass
    else:
        data["text/plain"] = repr(result)
    return {"output_type": "execute_result", "execution_count": None, "data": data, "metadata": {}}

start = time.time()
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if not src.strip():
        continue
    counter += 1
    head = src.splitlines()[0][:66]
    print(f"\n{'='*72}\nCELL {i}  ({time.time()-start:.1f}s)  {head}\n{'='*72}", flush=True)
    stdout, result = run_source(src)
    outs = []
    if stdout:
        outs.append({"output_type": "stream", "name": "stdout", "text": to_lines(stdout)})
    r = fmt(result)
    if r is not None:
        r["execution_count"] = counter
        outs.append(r)
    cell["execution_count"] = counter
    cell["outputs"] = outs
    print(stdout, end="", flush=True)
    if r is not None:
        print(f"[execute_result] {r['data'].get('text/plain','')[:300]}", flush=True)

nb_path.write_text(json.dumps(nb, indent=1))
print(f"\nALL CELLS EXECUTED OK in {time.time()-start:.1f}s  ({counter} code cells, outputs saved into the notebook)")
