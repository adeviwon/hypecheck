"""Does prior-day X-post sentiment predict next-day memecoin direction?
Frozen dates, prior-only features, and text-kind sensitivity; no causal claim.
"""
import argparse,json,re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
import pipeline as core
KINDS=('events','promotion','technical','other')
PATTERNS={
 'events':r'\b(?:etf|approv(?:al|ed)|list(?:ing|ed)|delist|partnership|launch|upgrade|lawsuit|sec|regulat\w*|hack(?:ed)?|exploit|burn(?:ed|ing)?|airdrop|adoption)\b',
 'promotion':r'\b(?:buy now|last chance|don.t miss|guaranteed|moon|fomo|ape in|send it|gem|100x|1000x|lambo)\b|🚀',
 'technical':r'\b(?:support|resistance|breakout|breakdown|trend(?:line)?|rsi|moving average|sma|ema|chart|consolidat\w*|candle|volume|bullish|bearish|longs?|shorts?)\b',
}
def headline_kind(text):
 for kind in KINDS[:-1]:
  if re.search(PATTERNS[kind],str(text),re.I):return kind
 return 'other'
def previous_day(prices):
 x=prices.sort_values('date')
 result=(x.close>x.open).astype(float).shift(1)
 # The next row must actually be the next UTC day, not simply next available bar.
 result=result.where(x.date.diff().eq(pd.Timedelta(days=1)))
 return pd.Series(result.to_numpy(),index=x.date,name='previous_day')
def aggregate_kinds(posts):
 records=[]
 for row in posts.to_dict('records'):
  target=pd.Timestamp(row['available_at']).floor('D')+pd.Timedelta(days=1)
  for coin in set(row['tickers'])&set(core.COINS):
   records.append({'coin':coin,'date':target,'kind':headline_kind(row['text']),'sentiment':float(row['sentiment'])})
 r=pd.DataFrame(records);index=pd.MultiIndex.from_frame(r[['coin','date']].drop_duplicates())
 out=pd.DataFrame(index=index)
 for k in KINDS:
  g=r[r.kind.eq(k)].groupby(['coin','date']).sentiment.agg(['size','mean'])
  out[k+'_count']=g['size'];out[k+'_sentiment']=g['mean']
 return out.fillna(0)
def paired_interval(dates,delta,reps=1000):
 z=pd.DataFrame({'date':np.asarray(dates),'delta':np.asarray(delta,float)})
 groups=[g.delta.to_numpy() for _,g in z.groupby('date',sort=True)]
 rng=np.random.default_rng(42);draws=[];n=len(groups)
 for _ in range(reps):
  starts=rng.integers(0,n,size=int(np.ceil(n/7)))
  ids=np.concatenate([(s+np.arange(7))%n for s in starts])[:n]
  draws.append(float(np.concatenate([groups[i] for i in ids]).mean()))
 return {'mean':float(z.delta.mean()),'interval95':np.quantile(draws,[.025,.975]).tolist(),
         'method':'1000 paired circular seven-observed-date block resamples; coins kept together'}
def holm(values):
 a=np.asarray(values,float);order=np.argsort(a);out=np.zeros(len(a));prev=0.
 for rank,i in enumerate(order):
  prev=max(prev,min(1.,(len(a)-rank)*a[i]));out[i]=prev
 return out.tolist()
def block_sign_p(dates,delta,reps=2000):
 z=pd.DataFrame({'date':np.asarray(dates),'delta':np.asarray(delta,float)}).groupby('date').delta.agg(['sum','size'])
 sums=z['sum'].to_numpy();blocks=np.array([sums[i:i+7].sum() for i in range(0,len(sums),7)])
 rng=np.random.default_rng(42);obs=abs(blocks.sum());null=abs((rng.choice([-1,1],size=(reps,len(blocks)))*blocks).sum(axis=1))
 return float((1+sum(null>=obs))/(reps+1))
def choose(train,val,features):
 candidates=[]
 for c in (.01,.1,1.):
  m=core.classifier(features,c).fit(train,train.actual_up)
  candidates.append((float(brier_score_loss(val.actual_up,m.predict_proba(val)[:,1])),c,m))
 loss,c,m=min(candidates,key=lambda x:(x[0],x[1]))
 return m,{'C':c,'validation_brier':loss}
def safe_metrics(y,p):return core.scores(y,p)
def main():
 raw,posts=core.load_posts();posts=core.score_posts(posts)
 prices={c:core.load_prices(c) for c in core.COINS};hourly={c:core.load_hourly(c) for c in core.COINS};btc=core.load_btc()
 frame,_=core.build_dataset(posts,prices,hourly,btc)
 frame=frame.merge(aggregate_kinds(posts).reset_index(),on=['coin','date'],how='left',validate='one_to_one')
 kindcols=[k+s for k in KINDS for s in ('_count','_sentiment')];frame[kindcols]=frame[kindcols].fillna(0.)
 frame['previous_day']=np.nan
 for c in core.COINS:
  mask=frame.coin.eq(c);frame.loc[mask,'previous_day']=frame.loc[mask,'date'].map(previous_day(prices[c])).to_numpy()
 if frame.previous_day.isna().any():raise ValueError('Missing previous UTC-day direction.')
 train,val,test=core.split_dates(frame)
 assert train.date.max()+pd.Timedelta(days=1)<val.date.min()
 assert val.date.max()+pd.Timedelta(days=1)<test.date.min()
 assert not set(train.date)&set(test.date)
 y=test.actual_up.to_numpy()
 configs={'Sentiment + coin':core.SOCIAL,'Coin only':[],'Market only':core.MARKET,'Market + sentiment':core.MARKET+core.SOCIAL}
 predictions={'Always up':np.ones(len(test)),'Previous-day direction':test.previous_day.to_numpy(),
              'Training majority':np.full(len(test),float(train.actual_up.mean()>=.5)),
              'Training prevalence':np.full(len(test),float(train.actual_up.mean()))}
 models={};selection={}
 for name,features in configs.items():
  models[name],selection[name]=choose(train,val,features);predictions[name]=models[name].predict_proba(test)[:,1]
 primary='Sentiment + coin';metrics={k:core.metric(k,test,p) for k,p in predictions.items()}
 paired={}
 for base in ('Always up','Previous-day direction','Coin only','Market only','Training majority'):
  delta=((predictions[primary]>=.5)==y).astype(float)-((predictions[base]>=.5)==y).astype(float)
  paired[base]=paired_interval(test.date,delta)
 paired['sentiment_increment_brier']=paired_interval(test.date,(predictions['Market + sentiment']-y)**2-(predictions['Market only']-y)**2)
 # Type comparisons and definitions are fixed in code before this run. Exploratory, since this corpus has been reused.
 kinds=[];basep=predictions['Market only']
 for k in KINDS:
  features=core.MARKET+[k+'_count',k+'_sentiment'];m,sel=choose(train,val,features);p=m.predict_proba(test)[:,1]
  mask=test[k+'_count'].to_numpy()>0;delta=(p-y)**2-(basep-y)**2
  row={'kind':k,'selection':sel,'train_posts':int(train[k+'_count'].sum()),'test_posts':int(test[k+'_count'].sum()),
       'test_coin_days_with_kind':int(mask.sum()),'all_test_rows':safe_metrics(y,p),
       'kind_present_rows':{name:safe_metrics(y[mask],pr[mask]) for name,pr in {'kind_model':p,'market_only':basep,'always_up':np.ones(len(test)),'previous_day':test.previous_day.to_numpy()}.items()},
       'paired_brier_vs_market':paired_interval(test.date,delta),'block_sign_p':block_sign_p(test.date,delta),
       'scope':'Exploratory fixed-rule buckets; overlapping coin-day slices; not manually validated semantic labels.'}
  predictions['Kind: '+k]=p;kinds.append(row)
 for row,p in zip(kinds,holm([r['block_sign_p'] for r in kinds])):row['holm_adjusted_p']=p
 # Both controls refit on training data; do not merely shuffle labels when scoring existing predictions.
 features=core.SOCIAL;real=predictions[primary];null=[];placebo=[];rng=np.random.default_rng(42)
 for _ in range(20):
  labels=rng.permutation(train.actual_up.to_numpy())
  m=core.classifier(features,selection[primary]['C']).fit(train,labels)
  null.append(safe_metrics(y,m.predict_proba(test)[:,1])['accuracy'])
  a,b=train.copy(),test.copy()
  a[features]=a[features].to_numpy()[rng.permutation(len(a))];b[features]=b[features].to_numpy()[rng.permutation(len(b))]
  m=core.classifier(features,selection[primary]['C']).fit(a,a.actual_up)
  placebo.append(safe_metrics(y,m.predict_proba(b)[:,1])['accuracy'])
 # Four expanding windows over the same final period; C frozen using earlier validation only.
 wf={name:np.full(len(test),np.nan) for name in (primary,'Market only','Market + sentiment')};windows=[]
 testdays=sorted(test.date.unique())
 for chunk in np.array_split(np.array(testdays,dtype=object),4):
  start=pd.Timestamp(chunk[0]);tr=frame[frame.date+pd.Timedelta(days=1)<start]
  mask=test.date.isin(chunk).to_numpy();part=test[mask];win={'start':str(start.date()),'end':str(pd.Timestamp(chunk[-1]).date()),'n':len(part),'metrics':{}}
  for name in wf:
   m=core.classifier(configs[name],selection[name]['C']).fit(tr,tr.actual_up);p=m.predict_proba(part)[:,1];wf[name][mask]=p
   win['metrics'][name]=safe_metrics(part.actual_up,p)
  win['metrics']['Always up']=safe_metrics(part.actual_up,np.ones(len(part)))
  win['metrics']['Previous-day direction']=safe_metrics(part.actual_up,part.previous_day.to_numpy());windows.append(win)
 walk={'note':'Secondary expanding-window re-estimation on the same final dates, not an independent replication.',
       'windows':windows,'pooled':{name:core.metric(name,test,p) for name,p in wf.items()}}
 cache=json.loads((core.ROOT/'data/sentiment.json').read_text())
 report={'question':'Do prior-day X-post headline-style texts add information about next-UTC-day memecoin direction?',
  'primary_model':primary,'target':'close > open on target UTC day; flat counts as not up',
  'source_description':'Original X posts, not newspaper headlines. Archive subset, not complete coverage of X.',
  'protocol':{'coins':list(core.COINS),'splits':{name:{'start':str(d.date.min().date()),'end':str(d.date.max().date()),'rows':len(d),'dates':d.date.nunique()} for name,d in [('train',train),('validation',val),('test',test)]},
     'C_candidates':[.01,.1,1.],'selection':'validation Brier only; threshold 0.5; no refit on validation for frozen model',
     'availability':'max(created_at, first_posted_at); prior UTC-day posts only',
     'previous_day_baseline':'Previous calendar UTC-day close > open, not previous close-to-close return.',
     'test_reuse':'Exploratory extension: this test period was inspected in earlier research. Not a fresh confirmatory test.'},
  'model_source':{'model':core.MODEL,'revision':core.MODEL_REV,'dataset':core.DATASET,'dataset_revision':core.DATASET_REV,'score_signature':cache['signature']},
  'leakage_checks':{'shared_train_test_dates':len(set(train.date)&set(test.date)),'train_target_end_before_validation':True,'validation_target_end_before_test':True,
     'prior_only_post_windows':bool((posts.available_at<posts.date).all()),'all_models_same_test_rows':True},
  'coverage':{'eligible_original_posts':len(posts),'test_coin_days_with_posts':int((test.post_count>0).sum()),'test_coin_days_total':len(test)},
  'selection':selection,'metrics':metrics,'paired_accuracy_differences':paired,'kinds':kinds,'taxonomy':PATTERNS,
  'controls':{'runs':20,'real_accuracy':metrics[primary]['accuracy'],'label_permuted_accuracy':null,'feature_placebo_accuracy':placebo,
       'label_permuted_max':max(null),'feature_placebo_max':max(placebo)},'walk_forward':walk,
  'investor_interpretation':{'justifies':['Treat bullish posts as opinions, not validated forecasts.','Use this study to question claims of reliable sentiment-based direction prediction.'],
     'does_not_justify':['Buying, selling, shorting or leverage from these predictions.','Claiming posts cause returns, or that all memecoin text prediction is impossible.','Expected profits, live trading efficacy or universal conclusions about all X posts.']},
  'limitations':['Frozen pretrained sentiment encoder plus trained logistic regression; encoder is not fine-tuned in this experiment.',
      'Rule-based text kinds are approximate and not manually validated; group slices may overlap.',
      'Sparse archive coverage and source concentration limit generalisation.',
      'Pretraining-corpus overlap and collection/edited-text history are not fully auditable.',
      'No fees, slippage, trade execution or profitability evaluation.'],
  'disclaimer':'This project is a prototype and is not financial advice.'}
 core.atomic_json(core.ROOT/'docs/headline-report.json',report)
 pred=test[['date','coin','actual_up','actual_return','post_count','previous_day']].copy()
 for name,p in predictions.items():pred[name]=p
 pred.to_csv(core.ROOT/'artifacts/headline-predictions.csv',index=False)
 m=models[primary];prep=m.named_steps['prep'];coef=m.named_steps['model']
 # Safe JSON parameters, no pickle. The notebook trains afresh and has full inference object.
 core.atomic_json(core.ROOT/'artifacts/headline-model.json',{'features':core.SOCIAL,'selection':selection[primary],
     'coefficient':coef.coef_.tolist(),'intercept':coef.intercept_.tolist(),'feature_names':prep.get_feature_names_out().tolist(),
     'numeric_mean':prep.named_transformers_['numeric'].mean_.tolist(),'numeric_scale':prep.named_transformers_['numeric'].scale_.tolist()})
 print('TEST PERIOD',report['protocol']['splits']['test'])
 for name,row in metrics.items():print(f"{name:25s} accuracy {row['accuracy']:.4%} Brier {row['brier']:.5f} n={row['n']}")
 print('KIND COMPARISONS (exploratory, full-panel Brier change; negative favours text kind)')
 for r in kinds:print(r['kind'],'observed coin-days',r['test_coin_days_with_kind'],'delta',r['paired_brier_vs_market'],'Holm p',r['holm_adjusted_p'])
 print('WALK FORWARD',json.dumps(walk['pooled'],default=str))
 print('WROTE docs/headline-report.json, artifacts/headline-predictions.csv, artifacts/headline-model.json')
if __name__=='__main__':main()
