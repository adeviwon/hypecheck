"""Expanded, naturally sampled memecoin inputs. No sentiment quotas or synthetic data."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pipeline as core
ROOT=Path(__file__).resolve().parent
COINS=('DOGE','SHIB','PEPE','BONK','FLOKI','WIF','TRUMP','NEIRO','PENGU','PNUT','TURBO','BOME','MEME')
PANEL_SHA='4ea2b737ee19daff97414a444ec3ea5313b797b78705e1b56dbf055bd5d47a56'

def select_posts(raw):
 x=raw.copy();x['created_at']=pd.to_datetime(x.created_at,utc=True,errors='coerce');x['first_posted_at']=pd.to_datetime(x.first_posted_at,utc=True,errors='coerce')
 x['available_at']=x[['created_at','first_posted_at']].max(axis=1)
 x=x[x.created_at.notna()&x.available_at.notna()&x.text.notna()&x.tweet_type.eq('tweet')].copy()
 x['text']=x.text.astype(str).str.strip();x=x[x.text.str.len().between(8,4000)]
 x=x.sort_values('available_at').drop_duplicates('text')
 x=x[(x.available_at>=core.START-pd.Timedelta(days=1))&(x.available_at<core.END)].copy()
 x['tickers']=x.tickers.apply(lambda a:list(a) if isinstance(a,(list,np.ndarray)) else [])
 x=x[x.tickers.apply(lambda a:bool(set(a)&set(COINS)))].copy()
 x['date']=x.available_at.dt.floor('D')+pd.Timedelta(days=1)
 return x

def inspect_posts(raw,price_path='data/expanded-memecoin-daily.csv'):
 selected=select_posts(raw)
 print('Eligible original posts:',len(selected))
 print('Coins:',', '.join(COINS))
 print(selected[['available_at','text']].head(5).to_string(index=False))
 print('Market bars:',pd.read_csv(price_path).shape[0])
 print('Natural sentiment counts are reported after scoring in the next section.')
 return selected

def load_prices(path=None):
 p=Path(path) if path is not None else ROOT/'data/expanded-memecoin-daily.csv'
 if hashlib.sha256(p.read_bytes()).hexdigest()!=PANEL_SHA:raise ValueError('Expanded price panel SHA-256 mismatch')
 d=pd.read_csv(p);d['date']=pd.to_datetime(d.date,utc=True)
 if set(d.coin)!=set(COINS):raise ValueError('Coin panel changed')
 out={}
 for c in COINS:
  x=d[d.coin.eq(c)].drop(columns='coin').sort_values('date').reset_index(drop=True)
  if x.date.duplicated().any() or not x.date.diff().iloc[1:].eq(pd.Timedelta(days=1)).all():raise ValueError('Missing/duplicate calendar bar for '+c)
  a=x[['open','high','low','close','volume']].to_numpy(float)
  if not np.isfinite(a).all() or (a[:,:4]<=0).any() or (a[:,4]<0).any():raise ValueError('Invalid market values')
  if (x.high<x[['open','close','low']].max(axis=1)).any() or (x.low>x[['open','close','high']].min(axis=1)).any():raise ValueError('Invalid OHLC bounds')
  out[c]=x
 return out

def build_frame(posts,prices):
 parts=[]
 for c in COINS:
  sub=posts[posts.tickers.apply(lambda a:c in a)]
  daily=sub.groupby('date').agg(post_count=('tweet_id','size'),sentiment=('sentiment','mean'),disagreement=('sentiment','std'),bullish=('bullish','mean'),bearish=('bearish','mean'))
  f=core.market_features(prices[c]).set_index('date').join(daily).reset_index()
  f['post_count']=f.post_count.fillna(0).astype(int);f['has_posts']=(f.post_count>0).astype(int);f[core.SOCIAL]=f[core.SOCIAL].fillna(0)
  f['coin']=c;f=f[(f.date>=core.START)&(f.date<core.END)].dropna(subset=core.MARKET);parts.append(f)
 frame=pd.concat(parts).sort_values(['date','coin']).reset_index(drop=True)
 if frame.duplicated(['date','coin']).any() or not np.isfinite(frame[core.MARKET+core.SOCIAL].to_numpy(float)).all():raise ValueError('Invalid expanded feature frame')
 return frame

def split_data(frame):
 days=sorted(frame.date.unique());a=pd.Timestamp(days[int(.6*len(days))]);b=pd.Timestamp(days[int(.8*len(days))])
 train=frame[frame.date+pd.Timedelta(days=1)<a].copy();val=frame[(frame.date>=a)&(frame.date+pd.Timedelta(days=1)<b)].copy();test=frame[frame.date>=b].copy()
 for x in [train,val,test]:
  if set(x.coin)!=set(COINS) or x.actual_up.nunique()!=2:raise ValueError('Incomplete coin or outcome support')
 return train,val,test

def load():
 raw,original=core.load_posts();selected=select_posts(raw)
 temp=ROOT/'.cache/expanded-headlines';(temp/'data').mkdir(parents=True,exist_ok=True)
 path=temp/'data/sentiment.json';canonical=ROOT/'data/expanded-sentiment.json'
 if canonical.exists():shutil.copy2(canonical,path)
 elif (ROOT/'data/sentiment.json').exists():
  old=json.loads((ROOT/'data/sentiment.json').read_text())
  expected=hashlib.sha256((core.MODEL_REV+'|max_length=128|'+'|'.join(original.tweet_id.astype(str)+original.text)).encode()).hexdigest()
  if old.get('signature')==expected:
   # Same pinned original dataset, model revision and text. Reuse overlapping frozen scores only.
   ids=set(selected.tweet_id.astype(str));scores={k:v for k,v in old['scores'].items() if k in ids}
   for v in scores.values():core.validate_sentiment_scores(v)
   signature=hashlib.sha256((core.MODEL_REV+'|max_length=128|'+'|'.join(selected.tweet_id.astype(str)+selected.text)).encode()).hexdigest()
   path.write_text(json.dumps({'signature':signature,'scores':scores}))
 oldroot=core.ROOT
 try:
  core.ROOT=temp;scored=core.score_posts(selected)
 finally:core.ROOT=oldroot
 shutil.copy2(path,canonical)
 prices=load_prices();frame=build_frame(scored,prices)
 return raw,scored,prices,frame
