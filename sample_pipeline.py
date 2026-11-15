"""ScamCheck experimental sequence applied to real, timed memecoin crowd snapshots."""
import json,math
from collections import Counter
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer,ENGLISH_STOP_WORDS
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,brier_score_loss
import pipeline as core
from expanded_inputs import load,COINS
from crowd_snapshot import build_snapshot,parse_pasted,parse_csv
from headline_study import paired_interval
ROOT=Path(__file__).resolve().parent
BASE_COINS=("DOGE","SHIB","PEPE","BONK","FLOKI")
NEW_COINS=tuple(c for c in COINS if c not in BASE_COINS)
SOCIAL=["sentiment","disagreement","bullish","bearish","neutral","log_posts"]

def weight(frame):return (1/frame.groupby(["date","coin"]).text.transform("size")).to_numpy()

def batch_features(texts,scores,coin):
 probs=np.array([[s[k] for k in ["bullish","bearish","neutral"]] for s in scores],float)
 if len(texts)!=len(probs) or not len(texts) or not np.isfinite(probs).all() or (probs<0).any() or (probs>1).any() or not np.allclose(probs.sum(axis=1),1,atol=1e-5):raise ValueError("Invalid real sentiment scores")
 sentiment=probs[:,0]-probs[:,1]
 return {"text":"\n".join(texts),"coin":coin,"sentiment":float(sentiment.mean()),"disagreement":float(sentiment.std()),"bullish":float(probs[:,0].mean()),"bearish":float(probs[:,1].mean()),"neutral":float(probs[:,2].mean()),"log_posts":float(np.log1p(len(texts)))}

@lru_cache(maxsize=1)
def dataset():
 _,posts,prices,panel=load();rows=[]
 for coin in COINS:
  history=posts[posts.tickers.apply(lambda tags:coin in tags)].copy()
  if coin in BASE_COINS:
   # No held-out-asset post may enter training through a multi-coin mention.
   history=history[~history.tickers.apply(lambda tags:bool(set(tags)&set(NEW_COINS)))]
  history=history.sort_values(["available_at","tweet_id"])
  bars=prices[coin].set_index("date")
  for cutoff in history.available_at.drop_duplicates():
   day=cutoff.floor("D")+pd.Timedelta(days=1);previous=cutoff.floor("D")-pd.Timedelta(days=1)
   if day not in bars.index or previous not in bars.index:continue
   batch=history[(history.available_at>=cutoff-pd.Timedelta(hours=24)) & (history.available_at<=cutoff)].tail(200)
   texts=batch.text.tolist();scores=batch[["bullish","bearish","neutral"]].to_dict("records")
   features=batch_features(texts,scores,coin);bar=bars.loc[day];prior=bars.loc[previous]
   features.update(date=day,cutoff=cutoff,actual_up=int(bar.close>bar.open),actual_return=float(bar.close/bar.open-1),
    previous_completed_day=int(prior.close>prior.open),post_count=len(batch),texts=texts,sentiment_scores=scores,
    source_ids=batch.tweet_id.astype(str).tolist(),group="established" if coin in BASE_COINS else "unseen_coins")
   rows.append(features)
 frame=pd.DataFrame(rows).sort_values(["date","coin","cutoff"]).reset_index(drop=True)
 days=sorted(panel.date.unique());boundary=pd.Timestamp(days[int(.8*len(days))])
 train=frame[(frame.group=="established") & (frame.date+pd.Timedelta(days=1)<boundary)].copy()
 later=frame[(frame.group=="established") & (frame.date>=boundary)].copy()
 unseen=frame[(frame.group=="unseen_coins") & (frame.date>=boundary)].copy()
 if any(x.empty or x.actual_up.nunique()!=2 for x in [train,later,unseen]):raise ValueError("Insufficient outcome support for an honest cohort test")
 assert train.date.max()+pd.Timedelta(days=1)<min(later.date.min(),unseen.date.min())
 assert not set(i for ids in train.source_ids for i in ids)&set(i for ids in unseen.source_ids for i in ids)
 return frame,train,later,unseen

def encoder(char=False,coin_only=False):
 blocks=[]
 if not coin_only:
  vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(2,5),sublinear_tf=True) if char else TfidfVectorizer()
  blocks.extend([("text",vec,"text"),("numeric",StandardScaler(),SOCIAL)])
 blocks.append(("coin",OneHotEncoder(handle_unknown="ignore"),["coin"]))
 return ColumnTransformer(blocks)

def fit_models(train):
 prep=encoder();x=prep.fit_transform(train);models={}
 for name,balanced in [("Model 1",None),("Model 2","balanced")]:
  m=LogisticRegression(max_iter=1000,class_weight=balanced)
  m.fit(x,train.actual_up,sample_weight=weight(train));models[name]=(prep,m)
 cprep=encoder(char=True);cx=cprep.fit_transform(train)
 cm=LogisticRegression(max_iter=1000,class_weight="balanced",C=10)
 cm.fit(cx,train.actual_up,sample_weight=weight(train));models["Character trial"]=(cprep,cm)
 cp=encoder(coin_only=True);cx=cp.fit_transform(train);m=LogisticRegression(max_iter=1000).fit(cx,train.actual_up,sample_weight=weight(train));models["Coin only"]=(cp,m)
 return models

def metrics(frame,p):
 y=frame.actual_up.to_numpy();w=weight(frame);hard=p>=.5
 return {"accuracy":float(accuracy_score(y,hard,sample_weight=w)),"up_precision":float(precision_score(y,hard,sample_weight=w,zero_division=0)),"up_recall":float(recall_score(y,hard,sample_weight=w,zero_division=0)),"brier":float(brier_score_loss(y,p,sample_weight=w)),"confusion_coin_day_weighted":confusion_matrix(y,hard,sample_weight=w,labels=[0,1]).tolist(),"snapshots":len(frame),"coin_days":len(frame[["date","coin"]].drop_duplicates()),"dates":int(frame.date.nunique())}

def evaluate(models,train,test):
 ps={name:model.predict_proba(prep.transform(test))[:,1] for name,(prep,model) in models.items()}
 prevalence=float(np.average(train.actual_up,weights=weight(train)))
 ps.update({"Always up":np.ones(len(test)),"Training majority":np.full(len(test),float(prevalence>=.5)),"Training prevalence":np.full(len(test),prevalence),"Previous completed day":test.previous_completed_day.to_numpy()})
 return {name:metrics(test,p) for name,p in ps.items()},ps

def export(prep,model):
 out={"intercept":float(model.intercept_[0]),"blocks":[],"estimator":{"C":model.C,"class_weight":model.class_weight,"max_iter":model.max_iter}};coef=model.coef_[0]
 for name,transformer,_ in prep.transformers_:
  if name=="remainder":continue
  b={"name":name,"coefficient":coef[prep.output_indices_[name]].tolist()}
  if name=="text":b.update(vocabulary={k:int(v) for k,v in transformer.vocabulary_.items()},idf=transformer.idf_.tolist(),analyzer=transformer.analyzer,ngram_range=list(transformer.ngram_range),sublinear_tf=transformer.sublinear_tf)
  elif name=="numeric":b.update(features=SOCIAL,mean=transformer.mean_.tolist(),scale=transformer.scale_.tolist())
  else:b.update(categories=transformer.categories_[0].tolist())
  out["blocks"].append(b)
 return out

def infer(saved,row,explain=False):
 z=saved["intercept"];reasons=[];other=[]
 for b in saved["blocks"]:
  if b["name"]=="coin":
   v=b["coefficient"][b["categories"].index(row["coin"])] if row["coin"] in b["categories"] else 0
   z+=v;other.append({"feature":"coin identity","logit_contribution":float(v)});continue
  if b["name"]=="numeric":
   for i,name in enumerate(b["features"]):
    v=(row[name]-b["mean"][i])/b["scale"][i]*b["coefficient"][i];z+=v;other.append({"feature":name,"logit_contribution":float(v)})
   continue
  analyzer=TfidfVectorizer(analyzer=b["analyzer"],ngram_range=tuple(b["ngram_range"])).build_analyzer()
  counts=Counter(analyzer(row["text"]));values={b["vocabulary"][term]:((1+math.log(n)) if b["sublinear_tf"] else n)*b["idf"][b["vocabulary"][term]] for term,n in counts.items() if term in b["vocabulary"]}
  norm=math.sqrt(sum(v*v for v in values.values()));reverse={i:t for t,i in b["vocabulary"].items()}
  if norm:
   for i,value in values.items():
    contribution=value/norm*b["coefficient"][i];z+=contribution
    if reverse[i] not in ENGLISH_STOP_WORDS:reasons.append({"word":reverse[i],"logit_contribution":float(contribution)})
 p=1/(1+math.exp(-max(-700,min(700,z))))
 return {"probability_up":p,"direction":"UP" if p>=.5 else "DOWN / FLAT","toward_up":sorted([r for r in reasons if r["logit_contribution"]>0],key=lambda r:-r["logit_contribution"])[:5],"toward_down":sorted([r for r in reasons if r["logit_contribution"]<0],key=lambda r:r["logit_contribution"])[:5],"other_feature_contributions":other}

def top_words(models):
 out={}
 for name in ["Model 1","Model 2","Character trial"]:
  prep,m=models[name];v=prep.named_transformers_["text"];words=np.array(v.get_feature_names_out());coef=m.coef_[0][prep.output_indices_["text"]]
  out[name]={"up":words[np.argsort(coef)[-12:][::-1]].tolist(),"down":words[np.argsort(coef)[:12]].tolist()}
 return out

def run_experiment(trainer=fit_models):
 frame,train,later,unseen=dataset();models=trainer(train)
 i,j=train_test_split(np.arange(len(train)),test_size=.2,stratify=train.actual_up,random_state=42)
 diagnostic_train,diagnostic_test=train.iloc[i],train.iloc[j]
 diagnostic_models=trainer(diagnostic_train)
 diag,_=evaluate(diagnostic_models,diagnostic_train,diagnostic_test)
 reports={};ledger=[];replay=[];contrasts={};test_coverage={}
 for name,part in [("later_time",later),("unseen_coin_market",unseen)]:
  reports[name],ps=evaluate(models,train,part)
  test_coverage[name]={"first_target_day":str(part.date.min().date()),"last_target_day":str(part.date.max().date()),"snapshots_per_coin":{k:int(v) for k,v in part.coin.value_counts().items()}}
  df=part[["date","coin","actual_up"]].copy();df["delta"]=((ps["Model 2"]>=.5)==part.actual_up).astype(float)-((ps["Coin only"]>=.5)==part.actual_up).astype(float)
  grouped=df.groupby(["date","coin"],as_index=False).delta.mean();contrasts[name]=paired_interval(grouped.date,grouped.delta.to_numpy())
  for k,(_,row) in enumerate(part.iterrows()):
   record={c:str(row[c]) if c in ["date","cutoff"] else row[c] for c in ["coin","date","cutoff","actual_up","actual_return","post_count"]}
   record.update(test=name,**{m:float(p[k]) for m,p in ps.items()});ledger.append(record)
   replay.append({**record,"texts":row.texts,"sentiment":{"bullish":row.bullish,"bearish":row.bearish,"neutral":row.neutral,"disagreement":row.disagreement}})
 # Serialization parity is asserted against every actual held-out row, not a dummy fixture.
 saved={name:export(*pair) for name,pair in models.items()}
 for part in [later,unseen]:
  for name,(prep,m) in models.items():
   expected=m.predict_proba(prep.transform(part))[:,1]
   actual=np.array([infer(saved[name],row.to_dict())["probability_up"] for _,row in part.iterrows()])
   assert np.allclose(expected,actual,atol=1e-10),name
 common=set(diagnostic_train.date)&set(diagnostic_test.date)
 report={"reference":"ScamCheck uploaded sample: default word TF-IDF; balanced class-weight change; char_wb(2,5), sublinear TF, balanced C=10 trial; coefficient explanations.","primary_model":"Model 2","target":"Next full UTC day after snapshot cutoff, close > open; flat is not up","input":"A collection of same-coin posts available in the preceding 24 hours, including cutoff; frozen FinTwitBERT sample sentiment and train-only wording features.","base_coins":list(BASE_COINS),"excluded_training_coins":list(NEW_COINS),"training":{"snapshots":len(train),"coin_days":len(train[["date","coin"]].drop_duplicates()),"first":str(train.date.min().date()),"last":str(train.date.max().date())},"tests":reports,"test_coverage":test_coverage,"training_post_count_range":[int(train.post_count.min()),int(train.post_count.max())],"random_80_20_diagnostic":{"metrics":diag,"shared_train_test_dates":len(common),"status":"Exact sample split reproduced on historical training subset only; NOT a valid forecasting headline because shared dates/outcomes can leak."},"paired_accuracy_vs_coin_only":contrasts,"top_words":top_words(models),"test_reuse":"Exploratory: this archive and final dates have been examined before. Not independent confirmation.","adaptations":["Actual market outcomes replace invented labels; no 40-case quota.","Primary earlier/later calendar split with purge replaces random-row forecasting claims.","Unseen coins and later times test the sample geographic/time-shift analogue.","Class weights may improve UP recall but can harm precision or accuracy. A balanced model is not assumed better.","Character trial is reported; word Model 2 is predeclared for understandable explanations, not chosen because it wins test accuracy.","Each coin/target-day has total weight one across its snapshots.","Sentiment means estimated language in this archive sample, not full X sentiment or causal impact."]}
 artifact={"version":2,"primary_model":"Model 2","models":saved,"coins":list(COINS),"report":report}
 for d in ["docs","artifacts"]:(ROOT/d).mkdir(exist_ok=True)
 (ROOT/"artifacts/crowd-model.json").write_text(json.dumps(artifact,separators=(",",":")))
 (ROOT/"docs/crowd-report.json").write_text(json.dumps(report,indent=2))
 (ROOT/"docs/crowd-replay.json").write_text(json.dumps(replay,separators=(",",":")))
 pd.DataFrame(ledger).to_csv(ROOT/"artifacts/crowd-predictions.csv",index=False)
 print(json.dumps({"training":report["training"],"tests":reports},indent=2));return artifact,report

@lru_cache(maxsize=1)
def neural():
 import torch
 from transformers import AutoTokenizer,AutoModelForSequenceClassification
 torch.set_num_threads(4)
 tokenizer=AutoTokenizer.from_pretrained(core.MODEL,revision=core.MODEL_REV,trust_remote_code=False,token=False)
 model=AutoModelForSequenceClassification.from_pretrained(core.MODEL,revision=core.MODEL_REV,trust_remote_code=False,use_safetensors=True,token=False).eval()
 return tokenizer,model

def score_batch(texts):
 # Exact verified historical scores may be reused; genuinely new text runs the actual frozen network.
 _,posts,_,_=load();known={r.text:{k:getattr(r,k) for k in ["bullish","bearish","neutral"]} for r in posts.itertuples()}
 missing=[t for t in texts if t not in known]
 if missing:
  import torch
  tokenizer,model=neural();labels={str(v).lower():int(k) for k,v in model.config.id2label.items()}
  if set(labels)!={"bullish","bearish","neutral"}:raise ValueError("Unexpected pretrained label map")
  for start in range(0,len(missing),16):
   batch=missing[start:start+16];encoded=tokenizer(batch,padding=True,truncation=True,max_length=128,return_tensors="pt")
   with torch.inference_mode():probs=model(**encoded).logits.softmax(-1).cpu().numpy()
   for text,values in zip(batch,probs):known[text]={k:float(values[i]) for k,i in labels.items()}
 return [known[t] for t in texts]

def batch_check(text,coin,cutoff=None,artifact=None,input_format="posts"):
 if input_format not in ["posts","csv"]:raise ValueError("Choose posts or csv")
 snapshot=(parse_csv if input_format=="csv" else parse_pasted)(text,coin,cutoff);texts=[p["text"] for p in snapshot.posts]
 scores=score_batch(texts);row=batch_features(texts,scores,coin)
 a=artifact or json.loads((ROOT/"artifacts/crowd-model.json").read_text())
 result=infer(a["models"]["Model 2"],row)
 result.update(coin=coin,cutoff_utc=snapshot.cutoff_utc,target_start_utc=snapshot.target_start_utc,target_end_utc=snapshot.target_end_utc,
  sample_sentiment={k:row[k] for k in ["bullish","bearish","neutral","disagreement"]},data_quality={**snapshot.audit,"training_post_count_range":a["report"]["training_post_count_range"],"batch_size_outside_training_range":not(a["report"]["training_post_count_range"][0]<=len(texts)<=a["report"]["training_post_count_range"][1])},
  baseline_probability_up=infer(a["models"]["Coin only"],row)["probability_up"],
  warning="Experimental uncalibrated forecast, not causal price impact or financial advice. Input is a supplied sample, not all X posts. No demonstrated trading edge may be claimed from this reused archive.")
 return result
if __name__=="__main__":run_experiment()
