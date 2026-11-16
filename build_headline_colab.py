"""Generate readable crowd-forecast methods, tables and plain-English findings."""
import json
from judge_tables import overall_findings,balance_findings,TEST_NAMES
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

**Try a tweet:** run the two setup cells below, paste your tweet in **Check a custom tweet**, choose its coin and run the cell. For the full study and the multi-post app in Section 9, choose **Runtime → Run all**.

**Live demo:** paste a tweet at https://adeviwon.github.io/hypecheck/live.html and the trained wording + coin model runs in the browser, with its own held-out scores shown beside the forecast. The complete model, including the frozen sentiment block, runs here.

**Our investigation:** inspect real posts → establish simple baselines → train a word-based model → examine its mistakes → test later dates and new coins → test balanced training and character features → explain the predictions and findings. All price outcomes are historical market observations.

*This project is a prototype and is not financial advice. No complete or live X feed, causal impact or trading skill is claimed. This reused archive is exploratory, not independent confirmation.*
""")
code("""#@title Automatic environment setup
import sys,subprocess,json,hashlib,urllib.request,contextlib,io
from pathlib import Path
IN_COLAB="google.colab" in sys.modules
if IN_COLAB:
    pins=["numpy==2.5.4","pandas==3.1.1","pyarrow==26.0.1","scikit-learn==1.9.1","requests==2.34.2","ipython==9.18.1","ipywidgets==8.1.9","transformers==5.23.0","safetensors==0.8.0"]
    subprocess.run([sys.executable,"-m","pip","install","-q",*pins],check=True)
    try:
        import torch
        from transformers import AutoTokenizer,AutoModelForSequenceClassification
    except (ImportError,OSError):
        subprocess.run([sys.executable,"-m","pip","install","-q","torch==2.15.0+cpu","--index-url","https://download.pytorch.org/whl/cpu"],check=True)
        import torch
print("Environment ready; actual Google Colab runtime:",IN_COLAB)
""",True)
setup="#@title Download and verify study files\nSETUP_REV="+repr(bootstrap["revision"])+"\nBOOTSTRAP="+repr({"colab_setup.py":bootstrap["sha256"],"notebook_support.py":bootstrap["presentation_sha256"]})+"\nSOURCE_REV="+repr(cfg["revision"])+"\nSOURCE_FILES="+repr(cfg["files"])+"\n"
setup+="""def verified_bytes(name,revision,expected):
    url=f"https://raw.githubusercontent.com/adeviwon/hypecheck/{revision}/{name}"
    with urllib.request.urlopen(url,timeout=60) as response:payload=response.read()
    if hashlib.sha256(payload).hexdigest()!=expected:raise ValueError("Changed verified file: "+name)
    return payload
for name,(revision,expected) in {**{n:(SETUP_REV,h) for n,h in BOOTSTRAP.items()},**{n:(SOURCE_REV,h) for n,h in SOURCE_FILES.items()}}.items():
    Path(name).write_bytes(verified_bytes(name,revision,expected))
from colab_setup import prepare
prepare()
import numpy as np,pandas as pd
from IPython.display import display,HTML
from notebook_support import table,accuracy_chart
from sample_pipeline import *
from judge_tables import *
print("Pinned sources and actual market inputs verified; no downloaded forecasts.")
"""
code(setup,True)
md("""## Try your own tweet
Run the two setup cells above, choose the coin your tweet is about, paste the tweet text and run **Check a custom tweet**. Edit the text and rerun the same cell to try another. Use Section 9 to analyse several posts together or enter CSV.

This cell uses the published Model 2 and FinTwitBERT. The first submitted tweet may trigger the sentiment model download. The output shows estimated language sentiment, the next full UTC day's direction estimate, and the coin-only baseline. The historical tests found no forecasting edge.
""")
model_pin = cfg["trained_model"]
coins = json.loads((ROOT/model_pin["path"]).read_text())["coins"]
custom = """#@title Check a custom tweet
tweet_text = "" #@param {type:"string", placeholder:"Paste one tweet as plain text"}
"""
custom += 'coin = "DOGE" #@param ' + json.dumps(coins) + '\n'
custom += f'published_model_pin = {tuple(model_pin[key] for key in ("path", "revision", "sha256"))!r}\n'
custom += """if tweet_text.strip():
    from crowd_app import panels
    if "published_model" not in globals():
        published_model = json.loads(verified_bytes(*published_model_pin))
    tweet_csv = pd.DataFrame({"text": [tweet_text]}).to_csv(index=False)
    custom_forecast = batch_check(tweet_csv, coin, artifact=published_model, input_format="csv")
    for content in panels(custom_forecast, published_model["report"]):
        display(HTML(content))
else:
    print("Paste a tweet, choose its coin and run this cell.")
"""
code(custom,True)
md("""## 1. Load the data
Real original public X posts and actual daily Binance prices. Frozen FinTwitBERT estimates language sentiment; next-day labels come from the **actual market**, not the sentiment model. The downstream forecast models are trained during this run. Transformers are **not fine-tuned**.
""")
code("""frame,train,later,unseen=dataset()
print(f"Training: {len(train):,} snapshots; later-time test: {len(later):,}; unseen-coin test: {len(unseen):,}.")
""")
md("""## 2. Look at the data first
We inspect real posts and their actual later market outcomes before fitting. Each row is a same-coin batch observed by a particular UTC cutoff, using the preceding 24 hours. Several snapshots can share one coin-day outcome, so each coin-day gets total weight one. Sentiment describes this supplied archive sample, not all of X. **Bullish language** and **bearish language** are average AI language scores, displayed as percentages—not the probability that the coin will rise or fall. **Actual next-day result** is measured from market prices, not predicted.
""")
code("""examples=examples_table(train)
table(examples.drop(columns=["Example wording"]),"Post batches, language scores and actual next-day results")
table(examples[["Forecast day (UTC)","Coin","Example wording"]],"Example public wording, kept separate so the data table stays readable")
from IPython.display import Markdown
display(Markdown("**What these examples show:** "+examples_findings(train.head(5))))
print("Established training coins:",BASE_COINS)
print("Excluded from supervised training:",NEW_COINS)
""")
md("""## 3. Split into training and test sets
We run a stratified random 80/20 split with random_state=42 as a **diagnostic** on historical training data, but do not use its score as forecasting evidence: related snapshots and market dates can appear on both sides.

The primary financial test instead uses earlier calendar dates for training and the final 20% of the price calendar for testing, with a purge. The later-time set uses established coins; the other set uses coins excluded from training. Multi-coin posts mentioning excluded assets are removed from the established training cohort. This is new-to-training data, **not independent confirmation**: the archive was explored before.
""")
code("""from sklearn.model_selection import train_test_split
random_train,random_test=train_test_split(train,test_size=.2,stratify=train.actual_up,random_state=42)
table(pd.DataFrame([{ "Purpose":name,"Post batches":len(d),"Distinct coin-days":len(d[["date","coin"]].drop_duplicates()),"Dates with data":d.date.nunique(),"First forecast day":d.date.min().strftime("%d %b %Y"),"Last forecast day":d.date.max().strftime("%d %b %Y")} for name,d in [("Training",train),("Later dates",later),("Unseen coins",unseen)]]))
assert train.date.max()+pd.Timedelta(days=1)<min(later.date.min(),unseen.date.min())
print("Random diagnostic shared dates:",len(set(random_train.date)&set(random_test.date)))
""")
md("""## 4. Baselines: simple rules before AI
We compare the AI models with always-up, training majority/prevalence, coin-only and the previous **completed** UTC-day direction at the snapshot cutoff. No unfinished-day close is allowed.

Report accuracy, precision and recall for UP, confusion matrices and **Brier error** (probability error; lower is better). **Correct directions** means how often the forecast matched the market. **UP calls that were right** is precision; **rising days caught** is recall. These answer different questions. Methods use identical rows and coin-day weights. A high accuracy alone does not establish forecasting value.
""")
code("""from sklearn.linear_model import LogisticRegression
coin_encoder=encoder(coin_only=True)
coin_model=LogisticRegression(max_iter=1000).fit(coin_encoder.fit_transform(train),train.actual_up,sample_weight=weight(train))
base_models={"Coin only":(coin_encoder,coin_model)}
for label,part in [("Later dates",later),("Unseen coins",unseen)]:
    scores,_=evaluate(base_models,train,part)
    table(scores_table(scores),label)
    display(Markdown("**Baseline finding:** "+label+" — the coin-only rule was correct "+format(scores["Coin only"]["accuracy"],".2%")+" of the time. The post-based model needs to improve on this, not just produce convincing-looking probabilities."))
""")
md("""## 5. Model 1: TF-IDF + logistic regression
We turn word usage into numerical features with TF-IDF, then fit logistic regression to actual price direction. Our fixed extra features represent estimated batch sentiment and coin identity; they remain unchanged between Model 1 and balanced Model 2. TF-IDF and scaling fit only training data. The label is whether the next complete UTC day closes above its open.
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
table(scores_table(score1),"Model 1 on later dates")
table(confusion_table(score1["Model 1"]),"Where Model 1 was right or wrong; weighted coin-day equivalents, not raw post counts")
display(Markdown("**What this result means:** "+model_findings(score1)))
""")
md("""## 6. Where Model 1 goes wrong
We inspect five actual held-out errors. **Missed a rise** means the model predicted no rise but the price rose. **Predicted a rise that did not happen** means it made a false UP call. A missed UP and a false-UP forecast are different mistakes. Neither has a universal financial cost: losses depend on returns, fees and execution. These are errors, not fabricated successful demonstrations.
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
    table(errors_table(shown),title)
    table(bad.groupby(["error_type","estimated_language"]).size().reset_index(name="raw_snapshot_errors").rename(columns={"error_type":"Mistake","estimated_language":"Post language","raw_snapshot_errors":"Wrong batch calls (raw count)"}),"Observed failure patterns; snapshots are dependent")
    print(title+":",len(bad),"wrong of",len(part),"raw snapshots")
inspect_errors(later,word_encoder,model1,"Model 1: later-time errors")
""")
md("""## 7. Our addition: later time and another coin market
We test whether the model transfers to **later dates** and to a separate **new-coin cohort excluded from training**. We retain every eligible batch rather than inventing cases or selecting a class quota. Actual subsequent prices determine the outcomes.

We inspect the top 12 learned word coefficients in each direction. These describe this fitted model, not economic laws or evidence that a word caused a price move.
""")
code("""market_scores,_=evaluate(models,train,unseen)
table(scores_table(market_scores),"Model 1 on coins excluded from training")
display(Markdown("**What happened on new coins:** "+model_findings(market_scores)))
words=np.array(word_encoder.named_transformers_["text"].get_feature_names_out())
coef=model1.coef_[0][word_encoder.output_indices_["text"]]
table(pd.DataFrame({"Words pushing the model toward a rise":words[np.argsort(coef)[-12:][::-1]],"Words pushing the model toward no rise":words[np.argsort(coef)[:12]]}),"Model 1: top word coefficients")
""")
md("""## 8. Improvement experiment: balanced class weights
We change **only the class weighting**: class_weight=balanced. Keep the vocabulary, numeric features, training rows and target fixed. The purpose is to test class-imbalance behaviour, not assume that catching more UP days improves trading decisions. Report both tests again, whether results improve or worsen.
""")
code("""model2=LogisticRegression(max_iter=1000,class_weight="balanced")
model2.fit(X_train,train.actual_up,sample_weight=weight(train))
models["Model 2"]=(word_encoder,model2)
for label,part in [("Later dates",later),("Unseen coins",unseen)]:
    result,_=evaluate(models,train,part)
    table(pd.DataFrame([{ "method":name,**{k:v for k,v in row.items() if not k.startswith("confusion")}} for name,row in result.items() if name in ["Model 1","Model 2"]]),label+": before and after the one change")
    inspect_errors(part,word_encoder,model2,"Model 2: "+label+" errors")
    display(Markdown("**What Model 2 achieved:** "+model_findings(result,"Model 2")))
""")
md("""### One alternative we tried: character n-grams
We test short character patterns instead of whole words: char_wb, ngrams (2,5), sublinear TF, balanced logistic regression and C=10. Keep the same sentiment and coin features. Show all results. The word-balanced model is predeclared as the readable demo, **not selected because it wins on test accuracy**. Looking at these reused tests cannot establish an independent improvement.
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
    table(scores_table(report["tests"][name]),TEST_NAMES[name])
display(Markdown("**What balancing changed:** "+balance_findings(report)))
print("Random 80/20 is diagnostic only:",report["random_80_20_diagnostic"]["status"])
table(pd.DataFrame([{ "Model":MODEL_NAMES.get(m,m),"Random-split accuracy (not a forecasting result)":format(v["accuracy"],".2%")} for m,v in report["random_80_20_diagnostic"]["metrics"].items()]),"Do not confuse this with future forecasting")
""")
character_report=json.loads((ROOT/"docs/crowd-report.json").read_text())
character_text=[]
for key,label in TEST_NAMES.items():
    values=character_report["tests"][key];char=values["Character trial"];word=values["Model 2"]
    character_text.append(label+": character features were correct "+format(char["accuracy"],".2%")+" of the time versus "+format(word["accuracy"],".2%")+" for the balanced word model. Their probability error was "+format(char["brier"],".4f")+" versus "+format(word["brier"],".4f")+" (lower is better).")
md("### What the character alternative tells us\n"+" ".join(character_text)+" A more complicated text representation did not solve the forecasting problem or establish trading value.")
md("""## 9. The demo: explain a batch forecast
For each word, we multiply its TF-IDF value by its learned coefficient to explain how it influenced this model. Sentiment and coin contributions are kept separate. The first demonstration uses the **first eligible held-out historical snapshot**, transported as quoted CSV to preserve blank lines inside original posts. It is an illustration, not an independent test.

The three panels distinguish estimated language sentiment, a price-direction model estimate, and whether the evidence supports trusting it. No causal price-impact estimate or guaranteed return is shown.
""")
code("""from crowd_app import panels,build_app
example=later.iloc[0]
result=batch_check(historical_csv(example),example.coin,example.cutoff.isoformat(),artifact=artifact,input_format="csv")
for content in panels(result,report):display(HTML(content))
print("Historical reveal — actual next-day direction:","UP" if example.actual_up else "DOWN / FLAT")
table(pd.DataFrame(result["toward_up"]+result["toward_down"]).rename(columns={"word":"Word","logit_contribution":"Contribution to this model score"}),"Actual word × TF-IDF contributions (logit units)")
""")
md("""### Try your own posts
The input app opens automatically during Run all. Scroll to this cell in Section 9. Paste public posts separated by blank lines, or CSV with a text column; choose a coin and click Analyse batch. Use CSV if an individual post itself contains blank lines. New text runs actual pinned FinTwitBERT inference, then the fitted forecast model—no phrase-list fallback.

Blank cutoff means actual submission time in UTC; exact next-day dates are displayed. A pasted old post is analysed at that submission time, not silently backtested. Supplied timestamps can be checked for window consistency, but their authenticity and complete platform coverage are not independently verified. No live X feed is connected.
""")
code("""#@title Open the real batch input app
RUN_APP=True #@param {type:"boolean"}
if RUN_APP:
    ui,app_state,controls=build_app(artifact)
    display(ui)
else:
    print("App disabled. Set RUN_APP=True and rerun this cell to enter your batch.")
""")
md("""## 10. Results summary
Compare the same rows before and after balanced weights, plus the character alternative. UP precision/recall expose the tradeoff that accuracy alone hides. Full confusion matrices and test support are saved in the report. The chart uses a full 0–100% axis.
""")
code("""table(results_table(report),"Complete measured comparison")
accuracy_chart({"primary_model":"Model 2","metrics":report["tests"]["later_time"]},["Model 1","Model 2","Character trial","Coin only","Training majority"])
print("Paired date-block accuracy differences vs coin-only:",report["paired_accuracy_vs_coin_only"])
""")
md("### Our findings, in plain English\n"+overall_findings(json.loads((ROOT/"docs/crowd-report.json").read_text())))
md("""## 11. Limitations and next steps
- This is a sparse archive, not every public post about a coin. Estimated sample sentiment is not validated full-population sentiment.
- The later-time and unseen-coin experiments use actual prices but reused dates; they are exploratory, not independent confirmation. Small per-coin samples limit conclusions.
- Balanced weights may increase UP recall while worsening false alarms or accuracy. They do not create trading skill. The character trial can overfit and gives less readable fragments.
- Probabilities are not independently calibrated. No profitability, fees/slippage, causal effect or useful return-magnitude forecast is established. Do not invert a losing model after seeing test labels and call that a new validated strategy.
- Next: gather a genuinely new timestamped public sample, freeze this protocol and evaluate prospectively; test whether users understand the evidence warnings.

**Investor boundary:** this prototype does not justify trading because a batch sounds bullish or the model predicts UP. Its useful result is demonstrating how apparent text-based confidence can fail a fair future-market test.
""")
code("""for filename in ["test_crowd_snapshot.py","test_sample_pipeline.py","test_crowd_app.py","test_judge_tables.py"]:
    completed=subprocess.run([sys.executable,filename],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(completed.stdout);completed.check_returncode()
print("Saved safe JSON models and genuine outputs. Actual Google Colab runtime:",IN_COLAB)
""")
md("""## Sources
- [financial-tweets-crypto](https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto), pinned revision and source hash in pipeline.py.
- [FinTwitBERT sentiment](https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment), pinned safe weights; scores are model estimates, not human gold labels.
- Actual Binance spot daily bars, frozen and hash-checked. [Market-data documentation](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#klinecandlestick-data).
- scikit-learn, PyTorch, Transformers, safetensors, NumPy, pandas, PyArrow, IPython and ipywidgets.
- Veritas AI participant reference notebook (ScamCheck) and Finance brief, credited as methodological sources.
- Hermes Agent/OpenAI coding assistance. The team must understand and explain the work. Third-party data/model terms remain applicable.

*This project is a prototype and is not financial advice.*
""")
metadata={"colab":{"toc_visible":True},"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"},"hypecheck":{"universe":"crowd-snapshot-forecast","quota_sampling":False,"expected_code_cells":sum(c["cell_type"]=="code" for c in cells),"source_sha256":cfg["files"],"bootstrap_revision":bootstrap["revision"],"bootstrap_sha256":{"colab_setup.py":bootstrap["sha256"],"notebook_support.py":bootstrap["presentation_sha256"]}}}
nb={"nbformat":4,"nbformat_minor":5,"metadata":metadata,"cells":cells};path=ROOT/"HypeCheck.ipynb"
if path.exists():
 old=json.loads(path.read_text());a=[c for c in cells if c["cell_type"]=="code"];b=[c for c in old["cells"] if c["cell_type"]=="code"]
 if [c["source"] for c in a]==[c["source"] for c in b]:
  for x,y in zip(a,b):x["outputs"]=y.get("outputs",[]);x["execution_count"]=y.get("execution_count")
path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+chr(10))
r=json.loads((ROOT/"docs/crowd-report.json").read_text())
readme="# HypeCheck — test the crowd\n\n**Live demo:** [paste a tweet and get the forecast](https://adeviwon.github.io/hypecheck/live.html) — the trained wording + coin model runs in your browser, no server, with its own held-out scores shown beside the forecast.\n\nSupply a batch of public X posts about one memecoin. HypeCheck separates estimated sample sentiment, next-day direction and the evidence for trusting that prediction. No claim of all-X coverage or causal price impact.\n\n[Current Colab](https://colab.research.google.com/github/adeviwon/hypecheck/blob/main/HypeCheck.ipynb)\n\n## What is actually trained\nDefault word TF-IDF + logistic regression; then the same features with balanced class weights; then a character-pattern alternative. Aggregate pinned FinTwitBERT sentiment and coin identity stay fixed between comparisons. The classifiers train on actual subsequent market outcomes, not model guesses. No transformer fine-tuning is claimed.\n\n## Honest results\n\n| Model | Later-time accuracy | Unseen-coin accuracy |\n|---|---:|---:|\n"
for name in r["tests"]["later_time"]:
 readme+="| "+name+" | "+format(r["tests"]["later_time"][name]["accuracy"],".2%")+" | "+format(r["tests"]["unseen_coin_market"][name]["accuracy"],".2%")+" |\n"
readme+="\nAll accuracy is coin-day weighted. Training: "+str(r["training"]["snapshots"])+" snapshots / "+str(r["training"]["coin_days"])+" coin-days. Later-time test: "+str(r["tests"]["later_time"]["Model 2"]["snapshots"])+" snapshots / "+str(r["tests"]["later_time"]["Model 2"]["coin_days"])+" coin-days. Unseen-coin test: "+str(r["tests"]["unseen_coin_market"]["Model 2"]["snapshots"])+" snapshots / "+str(r["tests"]["unseen_coin_market"]["Model 2"]["coin_days"])+" coin-days. The word-balanced model fails the coin-only control in both tests. No demonstrated forecasting edge.\n\n## Use\n**Try one tweet:** Open Colab, run the two setup cells, then choose a coin and paste the tweet in **Check a custom tweet**. Run that cell for the forecast. Edit the text and rerun it to try another tweet. This uses the published Model 2; the first tweet may trigger a FinTwitBERT download.\n\n**Run the full study or check a batch:** Runtime → Run all. Section 5 visibly fits Model 1; Section 8 fits balanced Model 2 and the character trial. Section 9 runs a genuine historical batch and opens the multi-post/CSV app automatically. In Section 9, choose a coin, paste your own public posts separated by blank lines (or select CSV text), leave UTC cutoff blank and click Analyse batch. The first new-text run downloads the pinned sentiment network; allow that download to finish. No API key or paid service is required. New text runs the actual pinned pretrained sentiment network. Missing/invalid models or inputs block forecasting rather than fabricate scores.\n\nThe primary target is the next complete UTC day after the snapshot cutoff, open to close, not the next 24 hours. Blank runtime cutoff means the actual submission time; old pasted text is not an automatic backtest.\n\n## Reproduction\nUse Python 3.12 and the pinned requirements. `python prepare_verification.py` restores exact legacy cache inputs; `python sample_pipeline.py` trains the current batch experiment; `python verify_submission.py` verifies the submission. Source/data downloads are version-pinned and hash-checked. Server notebook execution is not an actual Colab runtime test.\n\n## Verification\nCurrent evidence: [artifacts/crowd-verification.json](artifacts/crowd-verification.json); judge instructions and limitations: [JUDGE_READINESS.md](JUDGE_READINESS.md). The full gate, empty-directory notebook, simulated Colab dependency path and genuinely new-text inference were exercised. Corrupt-model rejection and default-open input have regression checks. Actual Google Colab infrastructure remains unverified. Older submission-verification.json is explicitly archived.\n\n## Sources and limits\n[Public post archive](https://huggingface.co/datasets/StephanAkkerman/financial-tweets-crypto), [FinTwitBERT](https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment), actual Binance market bars, scikit-learn and dependencies; Veritas AI participant reference notebook and Finance brief; Hermes Agent/OpenAI coding assistance. Sparse, biased archive and previously explored test dates mean exploratory evidence, not independent confirmation. No complete live feed, independently calibrated probabilities, causal effect or trading recommendation.\n\n**This project is a prototype and is not financial advice.**\n"
(ROOT/"README.md").write_text(readme)
writeup_values={}
for prefix,key in [("later","later_time"),("unseen","unseen_coin_market")]:
 for suffix,model in [("model","Model 2"),("coin","Coin only"),("up","Always up"),("previous","Previous completed day")]:
  writeup_values[prefix+"_"+suffix]=format(r["tests"][key][model]["accuracy"],".2%")
(ROOT/"WRITEUP.md").write_text((ROOT/"submission_writeup.md").read_text().format(**writeup_values))
print("Generated judge-readable crowd forecast notebook:",len(cells),"cells")
