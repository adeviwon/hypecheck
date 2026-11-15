"""Generate the current paste-tweet prediction submission, with visible actual training."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
cfg=json.loads((ROOT/"tweet-forecast-source.json").read_text())
bootstrap=json.loads((ROOT/"colab-loader.json").read_text())
cells=[]
def md(s):cells.append({"cell_type":"markdown","metadata":{},"source":s.strip().splitlines(keepends=True)})
def code(s,form=False):cells.append({"cell_type":"code","metadata":{"cellView":"form"} if form else {},"source":s.strip().splitlines(keepends=True),"execution_count":None,"outputs":[]})
md("""# HypeCheck — paste a post, test a forecast
**AI + Finance · Nathaniel Wong, Aaden Yick, Alastair Chu · Malvern College Hong Kong**

**Our user:** a trader reading a public X post about a memecoin. **Input:** its text, selected coin and optional earlier public discussion. **Output:** an estimated probability of next-day UP and a research-only return estimate.

The target is the **next complete UTC calendar day, open to close**, not the next 24 hours and not movement caused by this tweet. Wording is trained directly against actual market outcomes, not an AI sentiment label. This is a genuine predictive model, but prediction does not guarantee accuracy.

**Run:** Runtime → Run all. Change the text and coin in **Section 9** and rerun that cell. [Browser input demo](https://adeviwon.github.io/hypecheck/predict.html).

**Bottom line:** these models did not beat the training-majority direction rule. Recent-context and magnitude predictions are experiments, not validated trading signals. The reused archive makes this exploratory. No live X feed is connected.

*This project is a prototype and is not financial advice.*
""")
code("""#@title Automatic environment and verified source setup
import sys,subprocess,json,hashlib,urllib.request
from pathlib import Path
IN_COLAB="google.colab" in sys.modules
if IN_COLAB:
    pins=["numpy==2.5.4","pandas==3.1.1","pyarrow==26.0.1","scikit-learn==1.9.1","requests==2.34.2","ipython==9.18.1"]
    subprocess.run([sys.executable,"-m","pip","install","-q",*pins],check=True)
""",True)
code("\nSETUP_REV="+repr(bootstrap["revision"])+"\nSETUP_SHA="+repr(bootstrap["sha256"])+"\nSOURCE_REV="+repr(cfg["revision"])+"\nSOURCE_FILES="+repr(cfg["files"])+"\n"+"""
files={"colab_setup.py":(SETUP_REV,SETUP_SHA),"notebook_support.py":(SETUP_REV,'e5dd57e29724851b18e5bd2ebfb9b226b4bb5a72be36b4144a51503ab2e01e70'),**{n:(SOURCE_REV,h) for n,h in SOURCE_FILES.items()}}
for name,(rev,expected) in files.items():
    url=f"https://raw.githubusercontent.com/adeviwon/hypecheck/{rev}/{name}"
    with urllib.request.urlopen(url,timeout=60) as response:payload=response.read()
    if hashlib.sha256(payload).hexdigest()!=expected:raise ValueError("Changed source: "+name)
    Path(name).write_bytes(payload)
from colab_setup import prepare
prepare()
import numpy as np,pandas as pd
from notebook_support import table,accuracy_chart
import contextlib,io
from tweet_forecast import load,build_examples,split_data,weights,features,train_all,predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
print("Environment ready. Actual Colab runtime:",IN_COLAB)
""",True)
md("""## 1. Load the data
Original public posts and actual Binance daily market bars for 13 memecoins. No hand-picked bullish/bearish quota. The setup verifies the exact source and data before loading. Price labels are **actual market outcomes**, not model guesses.
""")
code("""raw,posts,prices,panel=load()
examples=build_examples(posts,prices)
print(f"{len(posts):,} original posts; {len(examples):,} eligible post/coin events; {len(prices)} coins.")
""")
md("""## 2. Inspect before training
A post can mention multiple coins. Repeated posts on one coin-day share its market outcome; they are not independent tests. Recent discussion includes at most 20 posts in the preceding 24 hours, strictly before the input post becomes available. No later likes, views or price moves are features.
""")
code("""table(examples[["available_at","date","coin","text","context_posts","actual_up"]].head(5),"Real posts and actual market labels")
assert (examples.available_at<examples.date).all()
assert (examples.context_latest.isna()|(examples.context_latest<examples.available_at)).all()
print("Input and context timestamps precede the prediction target.")
""")
md("""## 3. Split chronologically
Shared training/validation/test dates and a one-day purge. Fit vocabularies and coefficients on training only; choose regularisation using validation Brier error. Keep final dates out of fitting. Earlier research examined these dates, so this is **not independent confirmation**.
""")
code("""train,validation,test=[examples[examples.date.isin(p.date.unique())].copy() for p in split_data(panel)]
table(pd.DataFrame([{ "partition":name,"events":len(d),"coin_days":len(d[["date","coin"]].drop_duplicates()),"first":str(d.date.min().date()),"last":str(d.date.max().date())} for name,d in [("train",train),("validation",validation),("test",test)]]))
assert train.date.max()+pd.Timedelta(days=1)<validation.date.min()
assert validation.date.max()+pd.Timedelta(days=1)<test.date.min()
""")
md("""## 4. Baselines
Always up, training-majority, training prevalence and coin-only. Persistence uses the **last completed UTC day at the posting time**, not the still-unfinished posting day. All methods use exactly the same observed-post test cases. Each coin-day has total weight one divided over its posts.
""")
code("""from tweet_forecast import evaluate
prevalence=np.average(train.actual_up,weights=weights(train))
baselines={"Always up":np.ones(len(test)),"Training majority":np.full(len(test),float(prevalence>=.5)),"Previous completed day":test.previous_completed_day.to_numpy()}
table(pd.DataFrame([{ "method":name,**evaluate(test,p)} for name,p in baselines.items()]))
""")
md("""## 5. Model — actual visible training
Character TF-IDF learns wording patterns; logistic regression learns next-day direction from real price labels. We train separate tweet-only and tweet-plus-context models, with a coin-only control. Transformers are **not fine-tuned** here. Brier error measures probability error; lower is better. The function below is passed into the real training procedure: its `.fit(...)` calls produce the saved model used by the demo.
""")
code("""def train_direction(train,validation,context=False,coin_only=False):
    encoder=features(context=context,coin_only=coin_only)
    X_train=encoder.fit_transform(train)
    X_validation=encoder.transform(validation)
    candidates=[]
    for C in [0.01,0.1,1.0]:
        model=LogisticRegression(C=C,max_iter=1500,random_state=42)
        model.fit(X_train,train.actual_up,sample_weight=weights(train))
        p=model.predict_proba(X_validation)[:,1]
        loss=brier_score_loss(validation.actual_up,p,sample_weight=weights(validation))
        candidates.append((float(loss),C,model))
    loss,C,model=min(candidates,key=lambda row:(row[0],row[1]))
    return encoder,model,{"C":C,"validation_brier":loss}

with contextlib.redirect_stdout(io.StringIO()):
    artifact,report=train_all(trainer=train_direction)
table(pd.DataFrame(report["metrics"]).T.reset_index(),"Held-out direction predictions")
table(pd.DataFrame(report["selection"]).T.reset_index(),"Validation-only choices")
accuracy_chart({"metrics":report["metrics"],"primary_model":"tweet_only"},["tweet_only","tweet_and_context","coin_only","training_majority"])
""")
md("""## 6. Where the model goes wrong
Show confident errors, not just convincing examples. These mistakes are not used to tune the model. Probabilities are uncalibrated estimates, not guarantees.
""")
code("""predictions=pd.read_csv("artifacts/tweet-forecast-predictions.csv")
wrong=predictions[(predictions.tweet_only>=.5)!=predictions.actual_up].copy()
wrong["confidence"]=(wrong.tweet_only-.5).abs()
table(wrong.nlargest(5,"confidence")[["date","coin","actual_up","tweet_only","log_return"]])
""")
md("""## 7. Our addition — public discussion context
A separate model can use preceding public discussion as well as the input post. This tests whether broader wording context contributes, rather than assuming a tweet acts alone. No context supplied means the separately trained **tweet-only** model is used. Texts must already be public at prediction time; the app does not fetch X automatically.
""")
code("""table(pd.DataFrame(report["metrics"]).T.reset_index())
print("Paired date-block accuracy differences vs coin-only:",report["paired_accuracy_vs_coin_only"])
""")
md("""## 8. How much movement? A separate return experiment
A ridge regression learns actual next-day log returns, with training-only extreme-value clipping. Its output estimates the **whole coin return**, not the effect attributable to this tweet. Compare error with a constant trained without text; a larger error is worse. This head did not beat that baseline and must not be presented as a useful magnitude predictor.
""")
code("""table(pd.DataFrame(report["return_metrics"]).T.reset_index())
print("Units: percentage-point mean absolute error; lower is better.")
""")
md("""## 9. The demo — enter a public tweet
Edit the input fields and run this cell. The supplied sentence is an **illustrative editable input**, not a selected successful test. Inference uses the saved weights actually trained above. An estimated probability change relative to coin-only is a model comparison, **not causal impact**.
""")
code("""#@title Paste a public post and run the trained forecast
TWEET="DOGE is breaking resistance with increasing trading volume." #@param {type:"string"}
COIN="DOGE" #@param ["DOGE","SHIB","PEPE","BONK","FLOKI","WIF","TRUMP","NEIRO","PENGU","PNUT","TURBO","BOME","MEME"]
RECENT_PUBLIC_POSTS="" #@param {type:"string"}
result=predict(TWEET,COIN,RECENT_PUBLIC_POSTS,artifact=artifact)
table(pd.DataFrame([{"Output":k,"Value":str(v)} for k,v in result.items()]))
""")
md("""## 10. Results and what a trader can conclude
The model produces predictions but has not demonstrated an edge over simple baselines. Do **not** use a bullish model output as an independently justified reason to trade. An honest negative finding helps challenge misplaced confidence in social posts; it is not a profitable trading system.

The evaluation is sparse, retrospective and reused. Neither accuracy nor a return estimate accounts for fees, slippage or trade execution. Public post text can be misleading or sarcastic. We have not independently calibrated probabilities, collected a complete feed, established causation or tested a prospective deployment.
""")
code("""for filename in ["test_tweet_forecast.py"]:
    completed=subprocess.run([sys.executable,filename],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(completed.stdout);completed.check_returncode()
print("Training outputs saved as safe JSON; no pickled estimator. Actual Colab:",IN_COLAB)
""")
md("""## 11. Sources and responsibility
- Public original-post archive: Stephan Akkerman, financial-tweets-crypto on Hugging Face, pinned revision and hash in pipeline.py. Text is not independently human-verified sentiment ground truth.
- Binance daily spot market bars, frozen and hash-checked in expanded_inputs.py. Third-party terms remain applicable.
- scikit-learn, NumPy, pandas, PyArrow and Requests. Raw wording models here use TF-IDF, logistic regression and ridge regression; they do not fine-tune a pretrained transformer.
- Veritas AI Finance brief and sample submission structure. Hermes Agent/OpenAI coding assistance; the team must understand and explain all work.

This project is a prototype and is not financial advice.
""")
nb={"nbformat":4,"nbformat_minor":5,"metadata":{"colab":{"toc_visible":True},"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"},"hypecheck":{"universe":"tweet-to-next-day","source_sha256":cfg["files"],"bootstrap_revision":bootstrap["revision"],"bootstrap_sha256":{"colab_setup.py":bootstrap["sha256"],"notebook_support.py":bootstrap["presentation_sha256"]},"quota_sampling":False,"expected_code_cells":sum(c["cell_type"]=="code" for c in cells)}},"cells":cells}
path=ROOT/"HypeCheck.ipynb"
if path.exists():
 old=json.loads(path.read_text());a=[c for c in cells if c["cell_type"]=="code"];b=[c for c in old["cells"] if c["cell_type"]=="code"]
 if [c["source"] for c in a]==[c["source"] for c in b]:
  for x,y in zip(a,b):x["outputs"]=y.get("outputs",[]);x["execution_count"]=y.get("execution_count")
path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+chr(10))
report=json.loads((ROOT/"docs/tweet-forecast-report.json").read_text())
readme="# HypeCheck — public-post market forecast\n\nPaste a public tweet, choose its memecoin and get an estimated next-UTC-day direction. Optional earlier public posts supply recent context. No live X feed is connected.\n\n[Open the submission in Colab](https://colab.research.google.com/github/adeviwon/hypecheck/blob/main/HypeCheck.ipynb) · [Try the trained browser model](https://adeviwon.github.io/hypecheck/predict.html)\n\n## Actual training\nRaw wording → train-only character TF-IDF → trained logistic regression for direction. A separate ridge regression estimates whole-coin return, not a tweet-caused effect. The notebook visibly fits the models using actual market outcomes. This is not transformer fine-tuning.\n\n## Honest held-out results\n\n| Method | Coin-day-weighted accuracy |\n|---|---:|\n"
for name,row in report["metrics"].items():readme+="| "+name+" | "+format(100*row["accuracy"],".2f")+"% |\n"
readme+="\nTraining: 1,451 post/coin events across 859 coin-days. Validation: 71 events across 55 coin-days. Test: 271 events across 219 coin-days and 112 shared dates. Reused archive: exploratory, not independent confirmation. Neither wording model beats the training-majority baseline. Return estimates do not beat the training-constant error baseline.\n\n## Use\nIn Colab select Runtime → Run all, then edit Section 9. The static browser demo runs the same safe-JSON model locally, without uploading text. The target is the next complete UTC day, open to close, not the next 24 hours.\n\n## Sources and limits\nPublic financial-tweets-crypto archive, actual Binance daily market bars, scikit-learn and dependencies; pinned revisions and hashes are in source. Hermes Agent/OpenAI assisted coding. No causal impact identification, no independently calibrated probabilities, no validated trading edge, no complete live sentiment feed. Older files are retrospective research, not an alternative current submission.\n\n**This project is a prototype and is not financial advice.**\n"
(ROOT/"README.md").write_text(readme)
(ROOT/"WRITEUP.md").write_text(readme)
print("Generated current tweet-input notebook:",len(cells),"cells")
