"""Generate the focused original-question notebook from executable sources."""
import base64,hashlib,json,zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE='b0d05adcc394810f1ff6c72dba43a4c5d7e95985'
FILES=['market-hourly.csv','expanded-daily.csv']+[f'{c}-prices.csv' for c in ['DOGE','SHIB','PEPE','BONK','FLOKI']]
SHA={n:hashlib.sha256((ROOT/'data'/n).read_bytes()).hexdigest() for n in FILES}
SOURCES={n:(ROOT/n).read_text() for n in ['pipeline.py','headline_study.py','test_headline_study.py']}
CACHE=(ROOT/'data/sentiment.json').read_bytes()
CACHE64=base64.b64encode(zlib.compress(CACHE)).decode()
cells=[]
def md(s):cells.append({'cell_type':'markdown','metadata':{},'source':s.strip().splitlines(keepends=True)})
def code(s):cells.append({'cell_type':'code','execution_count':None,'outputs':[],'metadata':{},'source':s.strip().splitlines(keepends=True)})
md("""# HypeCheck — Do headlines tell us anything about memecoin direction?
**Veritas AI International High School Hackathon 2026 · AI + Finance**

**Team:** Nathaniel Wong (model), Aaden Yick (data), Alastair Chu (demo and write-up) · Malvern College Hong Kong.

*This project is a prototype and is not financial advice.*

We test whether sentiment in the previous UTC day's **original X posts** adds information about the next day's direction for **DOGE, SHIB, PEPE, BONK and FLOKI**. These are headline-style social posts, not newspaper headlines or a complete X feed.

**The answer may be “not really.”** We report our model beside **always up** and **previous-day direction**, plus harder training-majority, coin-only and market-only comparisons. We also test fixed text-kind buckets without treating their best slice as an investment strategy.

**Run:** Runtime → Run all. No keys or uploads. The default uses hash-checked cached outputs of a pinned pretrained sentiment model, then **trains logistic-regression direction models afresh**. The sentiment encoder is frozen, not fine-tuned. The optional recompute switch reruns that encoder. A server verification is not claimed as actual Colab execution.

**Study status:** the final period was examined in previous work. This is a reproducible exploratory extension, not a new untouched confirmatory test. Empty-workspace execution and results are retained below.
""")
code("""import sys, subprocess, json, hashlib, urllib.request, base64, zlib
from pathlib import Path
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    pins = ['numpy==2.5.4','pandas==3.1.1','pyarrow==26.0.1','scikit-learn==1.9.1','requests==2.34.2','transformers==5.23.0','safetensors==0.8.0']
    subprocess.run([sys.executable,'-m','pip','install','-q',*pins],check=True)
Path('data').mkdir(exist_ok=True); Path('docs').mkdir(exist_ok=True); Path('artifacts').mkdir(exist_ok=True)
print('Runtime:',sys.version,'Colab:',IN_COLAB)
print('Prototype only; no orders or financial advice.')
""")
md("""## 1. Verified public inputs
Market inputs are the exact bytes used by the original experiment, pinned to a public Git commit. No live Binance access is required. Original posts come from a pinned Hugging Face dataset revision. Every downloaded input is checked against SHA-256.

The compressed sentiment cache contains IDs and model scores, not author profiles. It is embedded for fast reproducibility, hash-checked, and tied to the checkpoint, text and tokenizer settings through the original pipeline's signature. It is not human sentiment ground truth. To independently recompute it, set `RECOMPUTE_SENTIMENT=True` below; this downloads the pinned public pretrained encoder and takes longer.
""")
fetch='BASE = '+repr(BASE)+chr(10)+'SHA = '+repr(SHA)+chr(10)
fetch+="""def download(url,path,expected):
    path=Path(path)
    if not path.exists():
        req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(req,timeout=120) as response:payload=response.read()
        if hashlib.sha256(payload).hexdigest()!=expected:raise ValueError('Download digest mismatch: '+str(path))
        path.write_bytes(payload)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Input digest mismatch: '+str(path))
    print('Verified',path)
for name,sha in SHA.items():
    download(f'https://raw.githubusercontent.com/adeviwon/hypecheck/{BASE}/data/{name}','data/'+name,sha)
download('https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto/resolve/c4b17a94a667af3da1c9ed5144f998e3a4f29966/data/train.parquet',
         'data/posts.parquet','81b2f7563a44c1a103e9468a83d52319dfeae0fc616422b4a4df90d8d684b960')
"""
fetch+='cache = zlib.decompress(base64.b64decode('+repr(CACHE64)+'))'+chr(10)
fetch+='assert hashlib.sha256(cache).hexdigest() == '+repr(hashlib.sha256(CACHE).hexdigest())+chr(10)
fetch+="""Path('data/sentiment.json').write_bytes(cache)
RECOMPUTE_SENTIMENT=False
if RECOMPUTE_SENTIMENT:
    Path('data/sentiment.json').unlink()
print('Verified pretrained score cache; direction models are trained afresh.')
"""
code(fetch)
md("""## 2. Read the executable method
The following cell writes the actual source used by this notebook; it is embedded, not fetched from a moving branch. `pipeline.py` supplies conservative timestamp filtering, pretrained sentiment inference and market loading. `headline_study.py` fits this focused study and constructs the **correct previous-day open-to-close baseline**.
""")
code('SOURCES = '+repr(SOURCES)+chr(10)+"""for name,source in SOURCES.items():
    Path(name).write_text(source)
    print('Wrote',name,len(source),'characters')
""")
md("""## 3. Inspect the data before fitting
Posts must be original, deduplicated, and available before the target cutoff. `available_at` is the later of publication and archive collection time where available. Only the previous UTC day's posts feed the forecast. The target is **close > open** on the next UTC day; a flat day counts as not up. No same-day return enters the predictor.
""")
code("""import pandas as pd
import pipeline as core
raw,selected=core.load_posts()
print('Source rows:',len(raw),'Eligible original posts:',len(selected))
print(selected[['available_at','text']].head(5).to_string(index=False))
print('All eligible post windows prior-only:',bool((selected.available_at<selected.date).all()))
""")
md("""## 4. Taxonomy fixed before this run
Exclusive post buckets, in priority order: **events → promotion → technical → other**. They are deterministic keyword rules, not verified human labels. A day can contain multiple kinds, so coin-day subsets overlap. Events such as listings may also be promotional; rule priority is a disclosed simplification. Counts shown below are post-coin incidences, not necessarily unique tweets.

Test all four kind-specific feature additions on identical test rows; report 7-date block intervals and Holm-adjusted block-sign tests. Sparse slices are descriptive, not strategies. The taxonomy is exploratory because this archive has already been inspected.
""")
code("""from headline_study import headline_kind, PATTERNS
print(json.dumps(PATTERNS,indent=2))
print('Example classifications:',[(x,headline_kind(x)) for x in ['ETF approval announced','buy now 100x','rejected at resistance','ordinary conversation']])
""")
md("""## 5. Train, validate and test
Chronological train/validation/test with a one-day purge; scaler and coefficients fit on training only. Select regularisation C from 0.01, 0.1 and 1.0 using validation Brier error, not test accuracy. Predict up at probability ≥0.5.

The primary model uses post count, presence, aggregate pretrained sentiment and disagreement, plus coin identity. Compare with coin-only to distinguish identity from text. Market-only and market+sentiment are a second ablation.

The previous-day baseline predicts **yesterday's close > open**, not yesterday's close-to-close return. Twenty refitted label-permutation and feature-placebo controls check whether apparent gains are easy to reproduce without the real text signal. Accuracy nulls need not be 50% when labels are imbalanced. Four expanding-window evaluations are a secondary robustness check on the same final period, not independent replications.
""")
code("""completed=subprocess.run([sys.executable,'headline_study.py'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
print(completed.stdout)
completed.check_returncode()
report=json.loads(Path('docs/headline-report.json').read_text())
print('LEAKAGE CHECKS:',json.dumps(report['leakage_checks'],indent=2))
""")
md('## 6. Accuracy on the same held-out rows — including the requested baselines')
code("""order=['Always up','Previous-day direction','Training majority','Sentiment + coin','Coin only','Market only','Market + sentiment']
table=pd.DataFrame([{k:v for k,v in report['metrics'][name].items() if k in ['name','accuracy','balanced_accuracy','brier','n','n_dates']} for name in order])
print(table.to_string(index=False))
print('PAIRED ACCURACY DIFFERENCES:')
print(json.dumps(report['paired_accuracy_differences'],indent=2))
print('COVERAGE:',report['coverage'])
""")
md('## 7. Do some kinds of headline-style posts matter more?')
code("""rows=[]
for row in report['kinds']:
    ci=row['paired_brier_vs_market']
    rows.append({'kind':row['kind'],'observed_test_coin_days':row['test_coin_days_with_kind'],
                 'post_coin_incidences':row['test_posts'],'all_row_accuracy':row['all_test_rows']['accuracy'],
                 'Brier_change_vs_market':ci['mean'],'CI_low':ci['interval95'][0],'CI_high':ci['interval95'][1],
                 'Holm_p':row['holm_adjusted_p']})
print(pd.DataFrame(rows).to_string(index=False))
print('Negative Brier change favours the kind feature. An interval spanning zero is inconclusive.')
print('Do not promote tiny subsets into trading rules; no manually validated headline-type ground truth.')
""")
md('## 8. Robustness checks and confident mistakes')
code("""print('WALK-FORWARD WINDOWS:')
for window in report['walk_forward']['windows']:
    print(window['start'],window['end'],window['n'],{k:v['accuracy'] for k,v in window['metrics'].items()})
controls=report['controls']
print('REAL / PERMUTED MAX / PLACEBO MAX:',controls['real_accuracy'],controls['label_permuted_max'],controls['feature_placebo_max'])
pred=pd.read_csv('artifacts/headline-predictions.csv')
wrong=pred[(pred['Sentiment + coin']>=.5)!=pred.actual_up].copy()
wrong['confidence']=(wrong['Sentiment + coin']-.5).abs()
print(wrong.nlargest(5,'confidence')[['date','coin','actual_up','Sentiment + coin','post_count']].to_string(index=False))
print('A wrong probability forecast is not evidence that the underlying post was fraudulent.')
""")
md('## 9. Automated protocol tests')
code("""checked=subprocess.run([sys.executable,'test_headline_study.py'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
print(checked.stdout)
checked.check_returncode()
print('Focused protocol tests passed.')
""")
md("""## 10. What an investor can and cannot conclude
A point estimate beating always-up and persistence is not enough to establish useful text information. Compare against training majority, coin-only and market-only too; inspect paired uncertainty and the two controls.

**Justified:** treat bullish social posts as opinions rather than demonstrated forecasts; question claims of reliable text-based price prediction.

**Not justified:** placing trades, shorting, leverage, profit expectations, causal claims that posts move prices, or claims that prediction is impossible in every dataset. No fees/slippage/execution test was performed.

Sparse coverage is critical: absence from the archive does not mean nobody discussed the coin. Text-kind samples may be too small for a decisive answer. Pretraining-corpus overlap and later text edits cannot be fully audited. This is exploratory research on a reused period.

*This project is a prototype and is not financial advice.*

## Sources and credits
- Posts: https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto (revision pinned above).
- Pretrained sentiment: https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment (checkpoint pinned in pipeline.py).
- Binance daily bars and independently supplied hourly/BTC data: exact pinned files and source provenance in the repository README and original report.
- scikit-learn, pandas, NumPy, PyTorch, Transformers and their dependencies.
- Hermes Agent / OpenAI coding assistance for implementation and testing. The team must understand and explain every part.
""")
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'cells':cells}
# Preserve outputs only when every code cell is unchanged, in order.
path=ROOT/'HypeCheck.ipynb'
if path.exists():
 old=json.loads(path.read_text());oc=[c for c in old['cells'] if c['cell_type']=='code'];nc=[c for c in cells if c['cell_type']=='code']
 if [c['source'] for c in oc]==[c['source'] for c in nc]:
  for a,b in zip(nc,oc):a['outputs']=b.get('outputs',[]);a['execution_count']=b.get('execution_count')
path.write_text(json.dumps(nb,indent=1))
print('Wrote HypeCheck.ipynb',len(cells),'cells')
