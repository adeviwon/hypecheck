"""Train user-entered tweet forecasts against real next-day price labels."""
import json, math
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score,brier_score_loss,mean_absolute_error
from expanded_inputs import select_posts,load_prices,split_data,COINS
import pipeline as core
ROOT=Path(__file__).resolve().parent

def load():
 raw,_=core.load_posts();posts=select_posts(raw);prices=load_prices();parts=[]
 for coin in COINS:
  f=core.market_features(prices[coin]);f=f[(f.date>=core.START) & (f.date<core.END)].dropna(subset=core.MARKET).copy();f["coin"]=coin;parts.append(f)
 return raw,posts,prices,pd.concat(parts).sort_values(["date","coin"]).reset_index(drop=True)

def build_examples(posts,prices):
 rows=[]
 for coin in COINS:
  market=prices[coin].set_index('date')
  history=posts[posts.tickers.apply(lambda tags:coin in tags)].sort_values(['available_at','tweet_id']).to_dict('records')
  for item in history:
   t=pd.Timestamp(item['available_at']);day=t.floor('D')+pd.Timedelta(days=1);previous=t.floor('D')-pd.Timedelta(days=1)
   if day not in market.index or previous not in market.index:continue
   bar=market.loc[day];prior=market.loc[previous]
   earlier=[p for p in history if t-pd.Timedelta(hours=24)<=pd.Timestamp(p['available_at'])<t][-20:]
   rows.append({'tweet_id':str(item['tweet_id']),'coin':coin,'available_at':t,'date':day,'text':item['text'],
    'context':'\n'.join(p['text'] for p in earlier),'context_posts':len(earlier),
    'context_latest':max((p['available_at'] for p in earlier),default=pd.NaT),
    'actual_up':int(bar.close>bar.open),'log_return':float(np.log(bar.close/bar.open)),
    'previous_completed_day':int(prior.close>prior.open)})
 out=pd.DataFrame(rows).sort_values(['date','coin','available_at']).reset_index(drop=True)
 if out.empty:raise ValueError('No eligible tweet/outcome pairs')
 assert (out.available_at<out.date).all()
 assert (out.context_latest.isna() | (out.context_latest<out.available_at)).all()
 return out

def weights(frame):return (1/frame.groupby(['date','coin']).tweet_id.transform('size')).to_numpy()

def features(context=False,coin_only=False):
 blocks=[]
 if not coin_only:
  blocks.append(('tweet',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,max_features=4000,sublinear_tf=True),'text'))
  if context:blocks.append(('context',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,max_features=2000,sublinear_tf=True),'context'))
 blocks.append(('coin',OneHotEncoder(handle_unknown='error'),['coin']))
 return ColumnTransformer(blocks)

def fit_direction(train,validation,context=False,coin_only=False):
 prep=features(context,coin_only);x=prep.fit_transform(train);v=prep.transform(validation);candidates=[]
 for c in [.01,.1,1.]:
  model=LogisticRegression(C=c,max_iter=1500,random_state=42)
  model.fit(x,train.actual_up,sample_weight=weights(train))
  loss=brier_score_loss(validation.actual_up,model.predict_proba(v)[:,1],sample_weight=weights(validation))
  candidates.append((float(loss),c,model))
 loss,c,model=min(candidates,key=lambda z:(z[0],z[1]))
 return prep,model,{'C':c,'validation_brier':loss}

def export_model(prep,model,kind):
 coef=np.asarray(model.coef_).reshape(-1)
 saved={'kind':kind,'intercept':float(np.asarray(model.intercept_).reshape(-1)[0]),'blocks':[]}
 for name,transformer,columns in prep.transformers_:
  if name=='remainder':continue
  coefficients=coef[prep.output_indices_[name]].tolist()
  if name=='coin':saved['blocks'].append({'name':'coin','categories':transformer.categories_[0].tolist(),'coefficient':coefficients})
  else:saved['blocks'].append({'name':name,'vocabulary':{k:int(v) for k,v in transformer.vocabulary_.items()},'idf':transformer.idf_.tolist(),'coefficient':coefficients})
 return saved

def linear(saved,text,coin,context=''):
 result=saved['intercept'];analyzer=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5)).build_analyzer()
 for block in saved['blocks']:
  if block['name']=='coin':result+=block['coefficient'][block['categories'].index(coin)];continue
  counts=Counter(analyzer(text if block['name']=='tweet' else context))
  values={block['vocabulary'][term]:(1+math.log(count))*block['idf'][block['vocabulary'][term]] for term,count in counts.items() if term in block['vocabulary']}
  norm=math.sqrt(sum(v*v for v in values.values()))
  if norm:result+=sum(v*block['coefficient'][i]/norm for i,v in values.items())
 return result

def predict(text,coin,context='',artifact=None):
 if not isinstance(text,str) or not 8<=len(text.strip())<=4000:raise ValueError('Enter a tweet of 8–4000 characters')
 if coin not in COINS:raise ValueError('Choose a supported coin')
 if not isinstance(context,str) or len(context)>20000:raise ValueError('Context must be text, at most 20,000 characters')
 a=artifact or json.loads((ROOT/'artifacts/tweet-forecast-model.json').read_text())
 mode='tweet_and_context' if context.strip() else 'tweet_only';m=a['models'][mode]
 z=linear(m['direction'],text,coin,context);prob=1/(1+math.exp(-max(-700,min(700,z))))
 base=1/(1+math.exp(-linear(a['coin_baseline'],text,coin)))
 return {'coin':coin,'forecast_window':'Next complete UTC calendar day, open to close; not next 24 hours',
  'probability_up':prob,'direction':'UP' if prob>=.5 else 'DOWN / FLAT',
  'estimated_return_percent':100*math.expm1(linear(m['return'],text,coin,context)),
  'change_vs_coin_baseline_percentage_points':100*(prob-base),'context_mode':mode,
  'historical_direction_accuracy':a['metrics'][mode]['accuracy'],'historical_coin_baseline_accuracy':a['metrics']['coin_only']['accuracy'],
  'warning':'Exploratory estimate, not proven trading skill, calibrated certainty, or this tweet’s causal price impact. Context must already be public before the forecast. No live X feed is connected.'}

def evaluate(frame,prob):
 return {'accuracy':float(accuracy_score(frame.actual_up,prob>=.5,sample_weight=weights(frame))),
  'brier':float(brier_score_loss(frame.actual_up,prob,sample_weight=weights(frame))),
  'post_coin_events':len(frame),'distinct_coin_days':len(frame[['date','coin']].drop_duplicates()),'dates':frame.date.nunique()}

def train_all(trainer=fit_direction):
 _,posts,prices,panel=load();examples=build_examples(posts,prices)
 groups=[examples[examples.date.isin(p.date.unique())].copy() for p in split_data(panel)]
 train,validation,test=groups
 assert train.date.max()+pd.Timedelta(days=1)<validation.date.min()
 assert validation.date.max()+pd.Timedelta(days=1)<test.date.min()
 for part in groups:
  if part.actual_up.nunique()!=2:raise ValueError('Both actual market classes required')
 models={};predictions={};metrics={};selection={};returns={}
 low,high=np.quantile(train.log_return,[.01,.99]);target=train.log_return.clip(low,high)
 for name,use_context,coin_only in [('tweet_only',False,False),('tweet_and_context',True,False),('coin_only',False,True)]:
  prep,model,choice=trainer(train,validation,use_context,coin_only)
  x=prep.transform(train);xt=prep.transform(test);prob=model.predict_proba(xt)[:,1]
  predictions[name]=prob;metrics[name]=evaluate(test,prob);selection[name]=choice
  reg=Ridge(alpha=10,solver='lsqr').fit(x,target,sample_weight=weights(train))
  returns[name]={'mae_percentage_points':float(mean_absolute_error(100*np.expm1(test.log_return),100*np.expm1(reg.predict(xt)),sample_weight=weights(test)))}
  models[name]={'direction':export_model(prep,model,'logistic'),'return':export_model(prep,reg,'ridge')}
  for index in range(min(12,len(test))):
   row=test.iloc[index];z=linear(models[name]['direction'],row.text,row.coin,row.context)
   assert abs(1/(1+math.exp(-z))-prob[index])<1e-9
   assert abs(linear(models[name]["return"],row.text,row.coin,row.context)-reg.predict(xt)[index])<1e-9
 prevalence=float(np.average(train.actual_up,weights=weights(train)))
 for name,p in [('always_up',np.ones(len(test))),('training_majority',np.full(len(test),float(prevalence>=.5))),('training_prevalence',np.full(len(test),prevalence)),('previous_completed_day',test.previous_completed_day.to_numpy())]:
  predictions[name]=p;metrics[name]=evaluate(test,p)
 from headline_study import paired_interval
 comparisons={}
 for name in ['tweet_only','tweet_and_context']:
  df=test[['date','coin','actual_up']].copy()
  df['delta']=((predictions[name]>=.5)==test.actual_up).astype(float)-((predictions['coin_only']>=.5)==test.actual_up).astype(float)
  daily=df.groupby(['date','coin'],as_index=False).delta.mean()
  comparisons[name]=paired_interval(daily.date,daily.delta.to_numpy())
 constant=100*math.expm1(float(np.average(target,weights=weights(train))))
 returns['training_constant']={'mae_percentage_points':float(mean_absolute_error(100*np.expm1(test.log_return),np.full(len(test),constant),sample_weight=weights(test)))}
 report={'task':'Paste a public tweet and choose its memecoin; forecast next UTC-day open-to-close direction.',
  'training_labels':'Actual Binance next-day close > open. Return head uses actual next-day log(close/open), clipped at training-only 1st/99th percentiles.',
  'text_learning':'Train-only character TF-IDF with logistic regression. Raw wording learns directly from actual market outcomes, not model-generated sentiment labels.',
  'context':'Optional preceding 24 hours of same-coin archived posts, strictly earlier availability; maximum 20 posts. Separate tweet-only model if no context supplied.',
  'partitions':{n:{'events':len(d),'coin_days':len(d[['date','coin']].drop_duplicates()),'first':str(d.date.min().date()),'last':str(d.date.max().date())} for n,d in zip(['train','validation','test'],groups)},
  'metrics':metrics,'selection':selection,'return_metrics':returns,'paired_accuracy_vs_coin_only':comparisons,
  'weighting':'Each observed coin-day has weight one, divided over its posts. Date-block uncertainty preserves dependent rows.',
  'limitations':['Reused archive/test dates: exploratory, not independent confirmation.','No causal tweet-impact label exists. Return estimate is whole-coin outcome, not movement caused by this post.','No live X feed; user optionally supplies already-public recent context.','Probabilities are not independently calibrated.','Return magnitude can fail its baseline; accuracy is not profitability.']}
 artifact={'version':1,'coins':list(COINS),'models':{k:v for k,v in models.items() if k!='coin_only'},'coin_baseline':models['coin_only']['direction'],'metrics':metrics,'return_metrics':returns,'task':report['task']}
 for name in ['artifacts','docs']:(ROOT/name).mkdir(exist_ok=True)
 (ROOT/'artifacts/tweet-forecast-model.json').write_text(json.dumps(artifact,separators=(',',':')))
 (ROOT/'docs/tweet-forecast-model.json').write_text(json.dumps(artifact,separators=(',',':')))
 (ROOT/'docs/tweet-forecast-report.json').write_text(json.dumps(report,indent=2))
 output=test.drop(columns=['text','context']).copy()
 for name,p in predictions.items():output[name]=p
 output.to_csv(ROOT/'artifacts/tweet-forecast-predictions.csv',index=False)
 print(json.dumps(report,indent=2));return artifact,report
if __name__=='__main__':train_all()
