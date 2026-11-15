"""Reproduce the actual ScamCheck experiment sequence for crowd forecasts."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
cfg=json.loads((ROOT/"crowd-source.json").read_text());bootstrap=json.loads((ROOT/"colab-loader.json").read_text());cells=[]
def md(s):cells.append({"cell_type":"markdown","metadata":{},"source":s.strip().splitlines(keepends=True)})
def code(s,form=False):cells.append({"cell_type":"code","metadata":{"cellView":"form"} if form else {},"source":s.strip().splitlines(keepends=True),"execution_count":None,"outputs":[]})
md("""# HypeCheck — test the crowd, not just a tweet
**Veritas AI International High School Hackathon 2026 · AI + Finance**
**Team:** Nathaniel Wong · Aaden Yick · Alastair Chu — Malvern College Hong Kong

**Our user:** a trader who sees many public posts about one memecoin and wonders whether the wording and crowd sentiment deserve forecasting trust. **Input:** a batch of public posts. **Output:** estimated sample sentiment, a next-day direction estimate, and the evidence against simple baselines.

**Bottom line:** this experiment does not demonstrate a forecasting edge. We report that rather than turn a weak model into trading advice.

**Runtime → Run all.** Section 5 visibly trains Model 1; Section 8 makes the exact balanced-weight change and character-ngram trial from ScamCheck. Section 9 runs the trained batch forecast and offers an optional interactive app. Source setup is collapsed; data loading stays short.

**ScamCheck mapping:** original examples → baseline → word model → mistakes → out-of-distribution tests → balanced model → character trial → word explanations → results and limits. Its challenge used team-written modern US-style messages against older UK SMS. We use actual later prices and excluded memecoins, not fictional financial labels. [Exact experiment map](https://github.com/adeviwon/hypecheck/blob/main/docs/scamcheck-pipeline-map.md).

*This project is a prototype and is not financial advice. No complete or live X feed, causal impact or trading skill is claimed. This reused archive is exploratory, not independent confirmation.*
""")
code("""#@title Automatic environment setup
import sys,subprocess,json,hashlib,urllib.request,contextlib,io
from pathlib import Path
IN_COLAB="google.colab" in sys.modules
if IN_COLAB:
    pins=["numpy==2.5.4","pandas==3.1.1","pyarrow==26.0.1","scikit-learn==1.9.1","requests==2.34.2","ipython==9.18.1","ipywidgets==8.1.9","transformers==5.23.0","safetensors==0.8.0"]
    subprocess.run([sys.executable,"-m","pip","install","-q",*pins],check=True)
print("Environment ready; actual Google Colab runtime:",IN_COLAB)
""",True)
setup="SETUP_REV="+repr(bootstrap["revision"])+"\nBOOTSTRAP="+repr({"colab_setup.py":bootstrap["sha256"],"notebook_support.py":bootstrap["presentation_sha256"]})+"\nSOURCE_REV="+repr(cfg["revision"])+"\nSOURCE_FILES="+repr(cfg["files"])+"\n"
setup+="""for name,(revision,expected) in {**{n:(SETUP_REV,h) for n,h in BOOTSTRAP.items()},**{n:(SOURCE_REV,h) for n,h in SOURCE_FILES.items()}}.items():
    url=f"https://raw.githubusercontent.com/adeviwon/hypecheck/{revision}/{name}"
    with urllib.request.urlopen(url,timeout=60) as response:payload=response.read()
    if hashlib.sha256(payload).hexdigest()!=expected:raise ValueError("Changed verified file: "+name)
    Path(name).write_bytes(payload)
from colab_setup import prepare
prepare()
import numpy as np,pandas as pd
from IPython.display import display,HTML
from notebook_support import table,accuracy_chart
from sample_pipeline import *
print("Pinned sources and actual market inputs verified; no downloaded forecasts.")
"""
code(setup,True)
md("""## 1. Load the data
Real original public X posts and actual daily Binance prices. Frozen FinTwitBERT estimates language sentiment; next-day labels come from the **actual market**, not the sentiment model. The downstream forecast models are trained during this run. Transformers are **not fine-tuned**.
""")
code("""frame,train,later,unseen=dataset()
print(f"Training: {len(train):,} snapshots; later-time test: {len(later):,}; unseen-coin test: {len(unseen):,}.")
""")
md("""## 2. Look at the data first
Like ScamCheck, read the source examples before fitting. Each row is a same-coin batch observed by a particular UTC cutoff, using the preceding 24 hours. Several snapshots can share one coin-day outcome, so each coin-day gets total weight one. Sentiment describes this supplied archive sample, not all of X.
""")
code("""table(train[["cutoff","date","coin","text","post_count","bullish","bearish","actual_up"]].head(5).assign(text=lambda x:x.text.str.slice(0,350)),"Real historical inputs and market labels")
print("Established training coins:",BASE_COINS)
print("Excluded from supervised training:",NEW_COINS)
""")
md("""## 3. Split into training and test sets
ScamCheck uses stratified random 80/20 with random_state=42. We reproduce that **diagnostic** on historical training data, but do not use its score as forecasting evidence: related snapshots and market dates can appear on both sides.

The primary financial test instead uses earlier calendar dates for training and the final 20% of the price calendar for testing, with a purge. The later-time set uses established coins; the other set uses coins excluded from training. Multi-coin posts mentioning excluded assets are removed from the established training cohort. This is new-to-training data, **not independent confirmation**: the archive was explored before.
""")
code("""from sklearn.model_selection import train_test_split
random_train,random_test=train_test_split(train,test_size=.2,stratify=train.actual_up,random_state=42)
table(pd.DataFrame([{ "set":name,"snapshots":len(d),"coin_days":len(d[["date","coin"]].drop_duplicates()),"dates":d.date.nunique(),"first":str(d.date.min().date()),"last":str(d.date.max().date())} for name,d in [("Training",train),("Later dates",later),("Unseen coins",unseen)]]))
assert train.date.max()+pd.Timedelta(days=1)<min(later.date.min(),unseen.date.min())
print("Random diagnostic shared dates:",len(set(random_train.date)&set(random_test.date)))
""")
md("""## 4. Baselines: simple rules before AI
ScamCheck always predicts not-spam. We report always-up, training majority/prevalence, coin-only and the previous **completed** UTC-day direction at the snapshot cutoff. No unfinished-day close is allowed.

Report accuracy, precision and recall for UP, confusion matrices and **Brier error** (probability error; lower is better). Methods use identical rows and coin-day weights. A high accuracy alone does not establish forecasting value.
""")
code("""from sklearn.linear_model import LogisticRegression
coin_encoder=encoder(coin_only=True)
coin_model=LogisticRegression(max_iter=1000).fit(coin_encoder.fit_transform(train),train.actual_up,sample_weight=weight(train))
base_models={"Coin only":(coin_encoder,coin_model)}
for label,part in [("Later dates",later),("Unseen coins",unseen)]:
    scores,_=evaluate(base_models,train,part)
    table(pd.DataFrame([{ "method":name,**{k:v for k,v in row.items() if not k.startswith("confusion")}} for name,row in scores.items()]),label)
""")
md("""## 5. Model 1: TF-IDF + logistic regression
The same default word TF-IDF and logistic regression as ScamCheck. Our fixed extra features represent estimated batch sentiment and coin identity; they remain unchanged between Model 1 and balanced Model 2. TF-IDF and scaling fit only training data. The label is whether the next complete UTC day closes above its open.
""")
code("""from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
vec=TfidfVectorizer()
word_encoder=ColumnTransformer([("text",vec,"text"),("numeric",StandardScaler(),SOCIAL),("coin",OneHotEncoder(handle_unknown="ignore"),["coin"])])
X_train=word_encoder.fit_transform(train)
model1=LogisticRegression(max_iter=1000)
model1.fit(X_train,train.actual_up,sample_weight=weight(train))
models={"Model 1":(word_encoder,model1),**base_models}
score1,pred1=evaluate(models,train,later)
table(pd.DataFrame([{ "method":name,**{k:v for k,v in row.items() if not k.startswith("confusion")}} for name,row in score1.items()]),"Model 1 on later dates")
table(pd.DataFrame(score1["Model 1"]["confusion_coin_day_weighted"],columns=["Predicted DOWN/FLAT","Predicted UP"]),"Rows: actual DOWN/FLAT, actual UP; counts are coin-day weighted")
""")
md("""## 6. Where Model 1 goes wrong
Like the sample, inspect five actual held-out errors. A missed UP and a false-UP forecast are different mistakes. Neither has a universal financial cost: losses depend on returns, fees and execution. These are errors, not fabricated successful demonstrations.
""")
code("""def inspect_errors(part,prep,model,title):
    p=model.predict_proba(prep.transform(part))[:,1]
    bad=part[((p>=.5)!=part.actual_up)].copy()
    bad["estimated_up"]=p[((p>=.5)!=part.actual_up)]
    bad["error_type"]=np.where(bad.actual_up.eq(1),"Missed UP","False UP")
    bad["estimated_language"]=np.where(bad.bullish>bad.bearish,"More bullish","More bearish/neutral")
    saved=export(prep,model)
    def reasons(row):
        explanation=infer(saved,row.to_dict())
        key="toward_up" if row.estimated_up>=.5 else "toward_down"
        return ", ".join(x["word"] for x in explanation[key][:3])
    shown=bad.head(5).copy();shown["actual_word_reasons"]=shown.apply(reasons,axis=1)
    table(shown[["date","coin","error_type","actual_up","estimated_up","actual_word_reasons","text"]].assign(text=lambda x:x.text.str.slice(0,250)),title)
    table(bad.groupby(["error_type","estimated_language"]).size().reset_index(name="raw_snapshot_errors"),"Observed failure patterns; snapshots are dependent")
    print(title+":",len(bad),"wrong of",len(part),"raw snapshots")
inspect_errors(later,word_encoder,model1,"Model 1: later-time errors")
""")
md("""## 7. Our addition: later time and another coin market
ScamCheck tests 40 self-written modern US-style messages after training on older UK SMS. We cannot invent future market outcomes, so our analogue uses **every eligible real later-date snapshot** and a separate **unseen-coin-market cohort**, not a 15/25 or 40-case quota. Their prices determine the labels.

We also inspect the top 12 word coefficients, as the sample does. These describe this fitted model, not economic laws or evidence that a word caused a price move.
""")
code("""market_scores,_=evaluate(models,train,unseen)
table(pd.DataFrame([{ "method":name,**{k:v for k,v in row.items() if not k.startswith("confusion")}} for name,row in market_scores.items()]),"Model 1: genuinely excluded coin cohort")
words=np.array(word_encoder.named_transformers_["text"].get_feature_names_out())
coef=model1.coef_[0][word_encoder.output_indices_["text"]]
table(pd.DataFrame({"toward_UP":words[np.argsort(coef)[-12:][::-1]],"toward_DOWN":words[np.argsort(coef)[:12]]}),"Model 1: top word coefficients")
""")
md("""## 8. Improvement experiment: balanced class weights
Make **exactly one change**, as ScamCheck does: class_weight=balanced. Keep the vocabulary, numeric features, training rows and target fixed. The purpose is to test class-imbalance behaviour, not assume that catching more UP days improves trading decisions. Report both tests again, whether results improve or worsen.
""")
code("""model2=LogisticRegression(max_iter=1000,class_weight="balanced")
model2.fit(X_train,train.actual_up,sample_weight=weight(train))
models["Model 2"]=(word_encoder,model2)
for label,part in [("Later dates",later),("Unseen coins",unseen)]:
    result,_=evaluate(models,train,part)
    table(pd.DataFrame([{ "method":name,**{k:v for k,v in row.items() if not k.startswith("confusion")}} for name,row in result.items() if name in ["Model 1","Model 2"]]),label+": before and after the one change")
    inspect_errors(part,word_encoder,model2,"Model 2: "+label+" errors")
""")
md("""### One alternative we tried: character n-grams
Reproduce the sample trial: char_wb, ngrams (2,5), sublinear TF, balanced logistic regression and C=10. Keep the same sentiment and coin features. Show all results. The word-balanced model is predeclared as the readable demo, **not selected because it wins on test accuracy**. Looking at these reused tests cannot establish an independent improvement.
""")
code("""cvec=TfidfVectorizer(analyzer="char_wb",ngram_range=(2,5),sublinear_tf=True)
char_encoder=ColumnTransformer([("text",cvec,"text"),("numeric",StandardScaler(),SOCIAL),("coin",OneHotEncoder(handle_unknown="ignore"),["coin"])])
cmodel=LogisticRegression(max_iter=1000,class_weight="balanced",C=10)
cmodel.fit(char_encoder.fit_transform(train),train.actual_up,sample_weight=weight(train))
models["Character trial"]=(char_encoder,cmodel)
def use_visible_models(data):
    return models if data.index.equals(train.index) else fit_models(data)
with contextlib.redirect_stdout(io.StringIO()):
    artifact,report=run_experiment(trainer=use_visible_models)
for name in ["later_time","unseen_coin_market"]:
    table(pd.DataFrame([{ "model":m,**{k:v for k,v in row.items() if not k.startswith("confusion")}} for m,row in report["tests"][name].items()]),name)
print("Random 80/20 is diagnostic only:",report["random_80_20_diagnostic"]["status"])
table(pd.DataFrame([{ "model":m,"diagnostic_accuracy":v["accuracy"]} for m,v in report["random_80_20_diagnostic"]["metrics"].items()]),"Do not confuse this with future forecasting")
""")
md("""## 9. The demo: explain a batch forecast
ScamCheck explains using TF-IDF × coefficient. We do the same for words in the supplied batch and keep sentiment/coin contributions separate. The first demonstration uses the **first eligible held-out historical snapshot**, transported as quoted CSV to preserve blank lines inside original posts. It is an illustration, not an independent test.

The three panels distinguish estimated language sentiment, a price-direction model estimate, and whether the evidence supports trusting it. No causal price-impact estimate or guaranteed return is shown.
""")
code("""from crowd_app import panels,build_app
example=later.iloc[0]
result=batch_check(historical_csv(example),example.coin,example.cutoff.isoformat(),artifact=artifact,input_format="csv")
for content in panels(result,report):display(HTML(content))
print("Historical reveal — actual next-day direction:","UP" if example.actual_up else "DOWN / FLAT")
table(pd.DataFrame(result["toward_up"]+result["toward_down"]),"Actual word × TF-IDF contributions (logit units)")
""")
md("""### Optional interactive app
Set RUN_APP=True and run the next cell. Paste public posts separated by blank lines, or CSV with a text column; choose a coin and click Analyse batch. Use CSV if an individual post itself contains blank lines. New text runs actual pinned FinTwitBERT inference, then the fitted forecast model—no phrase-list fallback.

Blank cutoff means actual submission time in UTC; exact next-day dates are displayed. A pasted old post is analysed at that submission time, not silently backtested. Supplied timestamps can be checked for window consistency, but their authenticity and complete platform coverage are not independently verified. No live X feed is connected.
""")
code("""#@title Open the real batch input app
RUN_APP=False #@param {type:"boolean"}
if RUN_APP:
    ui,app_state,controls=build_app(artifact)
    display(ui)
else:
    print("Optional app is off so Run all finishes. Set RUN_APP=True to enter your batch.")
""")
md("""## 10. Results summary
Compare the same rows before and after balanced weights, plus the character alternative. UP precision/recall expose the tradeoff that accuracy alone hides. Full confusion matrices and test support are saved in the report. The chart uses a full 0–100% axis.
""")
code("""table(pd.DataFrame([{ "test":test_name,"model":name,"accuracy":row["accuracy"],"UP_precision":row["up_precision"],"UP_recall":row["up_recall"],"Brier":row["brier"]} for test_name,test in report["tests"].items() for name,row in test.items()]),"Complete measured comparison")
accuracy_chart({"primary_model":"Model 2","metrics":report["tests"]["later_time"]},["Model 1","Model 2","Character trial","Coin only","Training majority"])
print("Paired date-block accuracy differences vs coin-only:",report["paired_accuracy_vs_coin_only"])
""")
md("""## 11. Limitations and next steps
- This is a sparse archive, not every public post about a coin. Estimated sample sentiment is not validated full-population sentiment.
- The later-time and unseen-coin experiments use actual prices but reused dates; they are exploratory, not independent confirmation. Small per-coin samples limit conclusions.
- Balanced weights may increase UP recall while worsening false alarms or accuracy. They do not create trading skill. The character trial can overfit and gives less readable fragments.
- Probabilities are not independently calibrated. No profitability, fees/slippage, causal effect or useful return-magnitude forecast is established. Do not invert a losing model after seeing test labels and call that a new validated strategy.
- Next: gather a genuinely new timestamped public sample, freeze this protocol and evaluate prospectively; test whether users understand the evidence warnings.

**Investor boundary:** this prototype does not justify trading because a batch sounds bullish or the model predicts UP. Its useful result is demonstrating how apparent text-based confidence can fail a fair future-market test.
""")
code("""for filename in ["test_crowd_snapshot.py","test_sample_pipeline.py","test_crowd_app.py"]:
    completed=subprocess.run([sys.executable,filename],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(completed.stdout);completed.check_returncode()
print("Saved safe JSON models and genuine outputs. Actual Google Colab runtime:",IN_COLAB)
""")
md("""## Sources
- [financial-tweets-crypto](https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto), pinned revision and source hash in pipeline.py.
- [FinTwitBERT sentiment](https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment), pinned safe weights; scores are model estimates, not human gold labels.
- Actual Binance spot daily bars, frozen and hash-checked. [Market-data documentation](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#klinecandlestick-data).
- scikit-learn, PyTorch, Transformers, safetensors, NumPy, pandas, PyArrow, IPython and ipywidgets.
- Veritas AI ScamCheck sample notebook and Finance brief: same experiment sequence with documented adaptations for temporal finance data. [Exact mapping](https://github.com/adeviwon/hypecheck/blob/main/docs/scamcheck-pipeline-map.md).
- Hermes Agent/OpenAI coding assistance. The team must understand and explain the work. Third-party data/model terms remain applicable.

*This project is a prototype and is not financial advice.*
""")
metadata={"colab":{"toc_visible":True},"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"},"hypecheck":{"universe":"crowd-snapshot-scamcheck-protocol","quota_sampling":False,"expected_code_cells":sum(c["cell_type"]=="code" for c in cells),"source_sha256":cfg["files"],"bootstrap_revision":bootstrap["revision"],"bootstrap_sha256":{"colab_setup.py":bootstrap["sha256"],"notebook_support.py":bootstrap["presentation_sha256"]}}}
nb={"nbformat":4,"nbformat_minor":5,"metadata":metadata,"cells":cells};path=ROOT/"HypeCheck.ipynb"
if path.exists():
 old=json.loads(path.read_text());a=[c for c in cells if c["cell_type"]=="code"];b=[c for c in old["cells"] if c["cell_type"]=="code"]
 if [c["source"] for c in a]==[c["source"] for c in b]:
  for x,y in zip(a,b):x["outputs"]=y.get("outputs",[]);x["execution_count"]=y.get("execution_count")
path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+chr(10))
r=json.loads((ROOT/"docs/crowd-report.json").read_text())
readme="# HypeCheck — test the crowd\n\nSupply a batch of public X posts about one memecoin. HypeCheck separates estimated sample sentiment, next-day direction and the evidence for trusting that prediction. No claim of all-X coverage or causal price impact.\n\n[Current Colab](https://colab.research.google.com/github/adeviwon/hypecheck/blob/main/HypeCheck.ipynb) · [Exact ScamCheck pipeline mapping](docs/scamcheck-pipeline-map.md)\n\n## What is actually trained\nDefault word TF-IDF + logistic regression; then the same features with balanced class weights; then the exact character-ngram trial from the sample. Aggregate pinned FinTwitBERT sentiment and coin identity stay fixed between comparisons. The classifiers train on actual subsequent market outcomes, not model guesses. No transformer fine-tuning is claimed.\n\n## Honest results\n\n| Model | Later-time accuracy | Unseen-coin accuracy |\n|---|---:|---:|\n"
for name in r["tests"]["later_time"]:
 readme+="| "+name+" | "+format(r["tests"]["later_time"][name]["accuracy"],".2%")+" | "+format(r["tests"]["unseen_coin_market"][name]["accuracy"],".2%")+" |\n"
readme+="\nAll accuracy is coin-day weighted. Training: "+str(r["training"]["snapshots"])+" snapshots / "+str(r["training"]["coin_days"])+" coin-days. Later-time test: "+str(r["tests"]["later_time"]["Model 2"]["snapshots"])+" snapshots / "+str(r["tests"]["later_time"]["Model 2"]["coin_days"])+" coin-days. Unseen-coin test: "+str(r["tests"]["unseen_coin_market"]["Model 2"]["snapshots"])+" snapshots / "+str(r["tests"]["unseen_coin_market"]["Model 2"]["coin_days"])+" coin-days. The word-balanced model fails the coin-only control in both tests. No demonstrated forecasting edge.\n\n## Use\nRuntime → Run all. Section 5 visibly fits Model 1; Section 8 fits balanced Model 2 and the character trial. Section 9 runs a genuine historical batch and offers the optional multi-post/CSV app. New text runs the actual pinned pretrained sentiment network. Missing/invalid models or inputs block forecasting rather than fabricate scores.\n\nThe primary target is the next complete UTC day after the snapshot cutoff, open to close, not the next 24 hours. Blank runtime cutoff means the actual submission time; old pasted text is not an automatic backtest.\n\n## Reproduction\nUse Python 3.12 and the pinned requirements. `python prepare_verification.py` restores exact legacy cache inputs; `python sample_pipeline.py` trains the current batch experiment; `python verify_submission.py` verifies the submission. Source/data downloads are version-pinned and hash-checked. Server notebook execution is not an actual Colab runtime test.\n\n## Sources and limits\n[Public post archive](https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto), [FinTwitBERT](https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment), actual Binance market bars, scikit-learn and dependencies; Veritas AI sample pipeline; Hermes Agent/OpenAI coding assistance. Sparse, biased archive and previously explored test dates mean exploratory evidence, not independent confirmation. No complete live feed, independently calibrated probabilities, causal effect or trading recommendation.\n\n**This project is a prototype and is not financial advice.**\n"
(ROOT/"README.md").write_text(readme)
(ROOT/"WRITEUP.md").write_text(readme.split("## Reproduction\n")[0]+"## Sources and limits\n"+readme.split("## Sources and limits\n")[1])
print("Generated ScamCheck-matched crowd notebook:",len(cells),"cells")
