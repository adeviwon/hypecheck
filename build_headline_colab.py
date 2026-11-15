"""Generate the expanded study in the organiser sample's narrative sequence."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
LOADER=json.loads((ROOT/'colab-loader.json').read_text())
cells=[]
def md(s):cells.append({'cell_type':'markdown','metadata':{},'source':s.strip().splitlines(keepends=True)})
def code(s):cells.append({'cell_type':'code','execution_count':None,'outputs':[],'metadata':{},'source':s.strip().splitlines(keepends=True)})
md("""# HypeCheck: do headline-style posts predict memecoin direction?
**Veritas AI International High School Hackathon 2026 · AI + Finance**

**Team:** Nathaniel Wong (model), Aaden Yick (data), Alastair Chu (demo and write-up) · Malvern College Hong Kong.

**Pitch:** HypeCheck helps a beginner crypto investor test whether bullish social posts deserve forecasting trust, by comparing sentiment-based next-day forecasts against simple rules and actual market outcomes.

**How to run:** Runtime → Run all. No keys, uploads or live Binance API calls. The default uses verified cached outputs from a pinned pretrained sentiment model and trains direction classifiers afresh. A recompute switch reruns frozen sentiment inference. The encoder is not fine-tuned.

This expands the original study to **13 memecoins**. There is no 40-tweet test and no 15/25 sentiment quota: all eligible naturally occurring posts are retained. X posts are our headline-style source, not newspaper headlines or a complete X feed.

The section sequence follows the sample: load → inspect → split → baseline → first model → mistakes → later-data test → measured improvement → demo → results → limitations and sources. New experiment, same archive: this is exploratory, not independent confirmation.

*This project is a prototype and is not financial advice.*
""")
code("""import sys,subprocess,json,hashlib,urllib.request
from pathlib import Path
IN_COLAB='google.colab' in sys.modules
if IN_COLAB:
    pins=['numpy==2.5.4','pandas==3.1.1','pyarrow==26.0.1','scikit-learn==1.9.1','requests==2.34.2','transformers==5.23.0','safetensors==0.8.0']
    subprocess.run([sys.executable,'-m','pip','install','-q',*pins],check=True)
for name in ['data','docs','artifacts']:Path(name).mkdir(exist_ok=True)
print('Runtime:',sys.version,'Colab:',IN_COLAB)
print('The default direction classifiers are trained during this run.')
""")
md("""## 1. Load the data
Load archived posts, real daily prices and pretrained sentiment scores. The short loader fetches a **fixed version** from GitHub and verifies every file with SHA-256. No API keys.

[Setup code](https://github.com/adeviwon/hypecheck/blob/main/colab_setup.py) · [Model code](https://github.com/adeviwon/hypecheck/blob/main/headline_study.py)
""")
url="https://raw.githubusercontent.com/adeviwon/hypecheck/"+LOADER["revision"]+"/colab_setup.py"
loader="setup_url="+repr(url)+chr(10)
loader+="payload=urllib.request.urlopen(setup_url,timeout=30).read()"+chr(10)
loader+="if hashlib.sha256(payload).hexdigest()!="+repr(LOADER["sha256"])+":raise ValueError(\"Setup file changed\")".replace('\\\"','"')+chr(10)
loader+="""Path("colab_setup.py").write_bytes(payload)
from colab_setup import prepare
RECOMPUTE_SENTIMENT=False
prepare(recompute_sentiment=RECOMPUTE_SENTIMENT)
import pipeline as core
from expanded_inputs import select_posts,load,split_data,COINS
raw,_=core.load_posts()
print(f"Loaded {len(raw):,} archived posts.")
"""
code(loader)
md("""## 2. Look at the data before training
Keep original posts, bounded nonempty text, exact supported ticker tags and earliest exact-text duplicates. The later of publication and archive collection time is conservative availability. A post from day d feeds only the forecast for day d+1.

Sentiment classes are outputs of the pretrained model, not human labels. Bullish does not mean the coin actually rose. Multi-coin posts can contribute to multiple coin-days; tweet count, market-bar count and prediction-row count are different units.
""")
code("""from expanded_inputs import inspect_posts
selected=inspect_posts(raw)
import pandas as pd,numpy as np
""")
md("""## 3. Split into training, validation and later-date test data
**Never split these financial rows randomly.** Use shared chronological boundaries, approximately 60/20/20 across UTC dates, and purge the adjacent day at each boundary. Earlier target windows must end before the next partition begins.

The label is next-UTC-day **close > open**; flat counts as not up. Only strictly prior-day sentiment and lagged market features can be predictors. Scaling and coefficients are fitted on training rows; regularisation is selected by validation Brier error. The expanded universe and period were not a pristine new preregistration; we disclose archive reuse.
""")
code("""raw,scored,prices,frame=load()
print('Natural sentiment counts:',scored[['bullish','bearish','neutral']].idxmax(axis=1).value_counts().to_dict())
train,validation,test=split_data(frame)
print(pd.DataFrame([{'partition':name,'first':str(x.date.min().date()),'last':str(x.date.max().date()),'coin_days':len(x),'UTC_dates':x.date.nunique()} for name,x in [('train',train),('validation',validation),('test',test)]]).to_string(index=False))
assert train.date.max()+pd.Timedelta(days=1)<validation.date.min()
assert validation.date.max()+pd.Timedelta(days=1)<test.date.min()
assert not set(train.date)&set(test.date)
print('Boundary and shared-date leakage checks passed.')
""")
md("""## 4. Baselines: always up and previous-day direction
Always up ignores all text. Persistence copies the previous calendar UTC day's **open-to-close** direction, matching the target—not previous close-to-close return. We also report training majority and a training-prevalence probability baseline so a falling market cannot masquerade as text skill.
""")
code("""from headline_study import previous_day
persistence=np.full(len(test),np.nan)
for coin in COINS:
    mask=test.coin.eq(coin).to_numpy()
    persistence[mask]=test.loc[mask,'date'].map(previous_day(prices[coin])).to_numpy()
assert np.isfinite(persistence).all()
y=test.actual_up.to_numpy()
baselines={'Always up':np.ones(len(test)),'Previous-day direction':persistence,'Training majority':np.full(len(test),float(train.actual_up.mean()>=.5)),'Training prevalence':np.full(len(test),train.actual_up.mean())}
print(pd.DataFrame([{'method':name,**core.scores(y,p)} for name,p in baselines.items()]).to_string(index=False))
""")
md("""## 5. Model 1: pretrained sentiment + logistic regression
The primary model uses daily count/presence, mean bullish-minus-bearish sentiment, disagreement, bullish/bearish probabilities and coin identity. Frozen **FinTwitBERT** scores text; **logistic regression** learns the market-direction target. The encoder is not fine-tuned.

Fit C candidates 0.01, 0.1 and 1.0 on training only, choose using validation Brier, threshold at 0.5. Compare with coin-only to test whether sentiment adds anything beyond asset identity. All methods use the same later-date rows.
""")
code("""done=subprocess.run([sys.executable,'headline_study.py','--universe','expanded'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
print(done.stdout)
done.check_returncode()
report=json.loads(Path('docs/headline-report.json').read_text())
pred=pd.read_csv('artifacts/headline-predictions.csv')
print('CHECKS:',report['leakage_checks'])
""")
md("""## 6. Where Model 1 goes wrong
Show the most confident wrong forecasts, not only successes. A probability is not a guarantee; a mistaken market forecast is not evidence the underlying author was fraudulent. These examples are deliberately selected for error analysis, not used to adjust the trained model.
""")
code("""wrong=pred[(pred['Sentiment + coin']>=.5)!=pred.actual_up].copy()
wrong['confidence']=(wrong['Sentiment + coin']-.5).abs()
print(wrong.nlargest(5,'confidence')[['date','coin','actual_up','Sentiment + coin','post_count','actual_return']].to_string(index=False))
for row in wrong.nlargest(3,'confidence').to_dict('records'):
    day=pd.Timestamp(row['date'])
    relevant=scored[(scored.date==day)&scored.tickers.apply(lambda a:row['coin'] in a)]
    print('INPUT CONTEXT',row['coin'],str(day.date()),relevant.text.head(2).tolist())
""")
md("""## 7. Our addition: the larger later-date evaluation
No hand-selected 40 cases, no 15/25 sentiment quota. Evaluate the **entire natural test panel** and show results by coin, plus four expanding-date windows. Assets on the same date are dependent; the number of dates is not multiplied by the number of coins to claim independent evidence.

This is a larger version of the same archive, not an independent external dataset. Sparse text coverage remains a limitation. Missing archive posts do not prove the coin was not discussed.
""")
code("""print('TEST PANEL:',report['protocol']['splits']['test'])
print('TEXT COVERAGE:',report['coverage'])
print('NATURAL SENTIMENT COUNTS:',report['tweet_sentiment_counts'])
print(pd.DataFrame([{'coin':coin,**values} for coin,values in report['metrics']['Sentiment + coin']['per_coin'].items()]).to_string(index=False))
for window in report['walk_forward']['windows']:
    print(window['start'],window['end'],'n=',window['n'],{name:row['accuracy'] for name,row in window['metrics'].items()})
print('Expanding-window checks reuse these final dates; not independent replications.')
""")
md("""## 8. One measured improvement: market context and kinds of posts
Compare sentiment alone, market history alone, and the combination. Then ask whether **events, promotion, technical/price action, or other** posts contribute differently.

Keyword buckets are exclusive per post in that priority order, approximate and not human-validated. Coin-day slices can overlap. Each kind's count and sentiment are added to market history, evaluated on all test rows. Report support, paired date-block intervals and Holm-adjusted block-sign tests. Negative Brier change favours adding the kind. A high score from a tiny slice is not a strategy.
""")
code("""from headline_study import PATTERNS
print('Fixed keyword rules:',json.dumps(PATTERNS,indent=2))
rows=[]
for item in report['kinds']:
    ci=item['paired_brier_vs_market']
    rows.append({'kind':item['kind'],'coin_days_with_kind':item['test_coin_days_with_kind'],'post_coin_incidences':item['test_posts'],'Brier_change':ci['mean'],'CI_low':ci['interval95'][0],'CI_high':ci['interval95'][1],'Holm_p':item['holm_adjusted_p']})
print(pd.DataFrame(rows).to_string(index=False))
print('Market + sentiment vs market-only paired Brier:',report['paired_accuracy_differences']['sentiment_increment_brier'])
""")
md("""## 9. The demo: inspect a historical forecast and reveal the market result
This uses the saved logistic model parameters, reconstructed from safe JSON—not a fabricated score. It is a historical forecast from prior information, not a live trade signal or a predictor of an individual tweet's causal effect. Change coin/date to another held-out row. Sentiment probabilities are not independently calibrated return probabilities.
""")
code("""saved=json.loads(Path('artifacts/headline-model.json').read_text())
def predict_saved(row):
    numeric=(row[saved['features']].to_numpy(dtype=float)-np.array(saved['numeric_mean']))/np.array(saved['numeric_scale'])
    categories=np.array([float(row['coin']==name.removeprefix('coin__coin_')) for name in saved['feature_names'][len(numeric):]])
    logit=float(np.r_[numeric,categories]@np.array(saved['coefficient'])[0]+saved['intercept'][0])
    return float(1/(1+np.exp(-logit)))
def replay(coin,date,reveal=False):
    day=pd.Timestamp(date,tz='UTC')
    row=test[(test.coin==coin)&(test.date==day)]
    if len(row)!=1:raise ValueError('Choose a supported coin and held-out UTC date.')
    row=row.iloc[0];prob=predict_saved(row)
    checked=pred[(pred.coin==coin)&(pd.to_datetime(pred.date,utc=True)==day)].iloc[0]
    assert np.isclose(prob,checked['Sentiment + coin'],atol=1e-12)
    context=scored[(scored.date==day)&scored.tickers.apply(lambda a:coin in a)]
    print(coin,date,'P(up), uncalibrated:',round(prob,4),'direction:','up' if prob>=.5 else 'not up')
    print('Eligible context:',context.text.head(3).tolist() or 'No eligible archive posts; not proof of silence.')
    if reveal:print('ACTUAL:',bool(row.actual_up),'return:',float(row.actual_return))
    else:print('Outcome hidden. Set reveal=True to inspect the real market result.')
first=pred[pred.post_count>0].sort_values(['date','coin']).iloc[0]
replay(first['coin'],str(first['date'])[:10],reveal=True)
print('Saved-model reconstruction verified against recorded prediction.')
""")
md("""## 10. Results summary — model and baselines together
Use the full test-period comparison, not the most flattering coin, sentiment class or confidence band. Paired uncertainty groups coins by date. Both training-label permutation and feature-placebo controls actually refit the classifier; their accuracy nulls need not equal 50% under class imbalance.
""")
code("""order=['Always up','Previous-day direction','Training majority','Training prevalence','Sentiment + coin','Coin only','Market only','Market + sentiment']
print(pd.DataFrame([{'method':name,**{k:v for k,v in report['metrics'][name].items() if k in ['accuracy','balanced_accuracy','brier','n','n_dates']}} for name in order]).to_string(index=False))
print('Paired model-minus-baseline accuracy differences:',json.dumps(report['paired_accuracy_differences'],indent=2))
print('Controls:',report['controls'])
ci=report['paired_accuracy_differences']['Market only']['interval95']
print('Interpretation:', 'positive exploratory accuracy difference; needs future confirmation' if ci[0]>0 else ('model underperforms market-only here' if ci[1]<0 else 'no clear accuracy improvement over market-only demonstrated'))
""")
md("""## 11. What this would and would not justify an investor doing
**Justified:** question claims that bullish posts reliably predict tomorrow; compare them with simple rules, uncertainty and actual outcomes.

**Not justified:** buying, selling, leverage, shorting, profit expectations, or saying posts caused returns. This result also does not prove all prediction impossible. No fees, slippage, execution or user-outcome study was performed.

The archive is sparse and selected, not a random sample of all X. Symbol collisions and retrospectively assigned ticker metadata may misattribute a post; this is not contract-address-verified pump.fun coverage. Current exchange listings create selection/survivorship limits. Later-listed coins have shorter training histories. Pretraining overlap, text edits and collection timing are not fully auditable. English sentiment truncates at 128 tokens. The period has been repeatedly explored; wider coverage is not independent confirmation.

*This project is a prototype and is not financial advice.*
""")
md('## 12. Protocol tests and reproducibility')
code("""for filename in ['test_headline_study.py','test_expanded_inputs.py']:
    checked=subprocess.run([sys.executable,filename],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(checked.stdout);checked.check_returncode()
print('Protocol tests passed. Runtime evidence above distinguishes actual Colab from server execution.')
""")
md("""## Sources and credits
- Public original posts: https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto ; dataset revision and SHA are in pipeline.py. Third-party post terms remain their own.
- Frozen sentiment: https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment ; pinned revision in pipeline.py, original work by Stephan Akkerman and Tim Koornstra.
- Actual Binance spot daily OHLCV bars, exact verified panel downloaded above. Field definitions: https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#klinecandlestick-data . Market data is not relicensed as our code.
- Original source and older independent hourly/BTC research remain credited in the repository; expanded primary models use daily bars, not invented hourly inputs.
- PyTorch, Transformers, scikit-learn, pandas, NumPy, PyArrow, Requests and dependencies.
- Hermes Agent / OpenAI coding assistance for research, implementation, review and testing. The team must understand and explain every part.
- Veritas AI official sample structure and Finance brief §4.2.4. We changed market/source and added time-aware evaluation, broader memecoins, text-kind checks, controls and a historical replay.
""")
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'hypecheck':{'universe':'expanded','expected_code_cells':sum(c['cell_type']=='code' for c in cells),'quota_sampling':False}},'cells':cells}
path=ROOT/'HypeCheck.ipynb'
if path.exists():
 old=json.loads(path.read_text());a=[c for c in cells if c['cell_type']=='code'];b=[c for c in old['cells'] if c['cell_type']=='code']
 if [c['source'] for c in a]==[c['source'] for c in b]:
  for x,y in zip(a,b):x['outputs']=y.get('outputs',[]);x['execution_count']=y.get('execution_count')
path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+chr(10));print('Wrote sample-format notebook:',len(cells),'cells;',nb['metadata']['hypecheck']['expected_code_cells'],'code cells')
