import unittest,tempfile
from pathlib import Path
import pandas as pd
from expanded_inputs import select_posts,load_prices,split_data,COINS
class ExpandedTests(unittest.TestCase):
 def test_supported_original_posts_only(self):
  r=pd.DataFrame({'tweet_id':['1','2','3'],'text':['$WIF resistance test','$WIF resistance test','$NOT supported example'],
   'tweet_type':['tweet','retweet','tweet'],'created_at':['2025-01-01']*3,'first_posted_at':['2025-01-01']*3,'tickers':[['WIF'],['WIF'],['NOT']]})
  x=select_posts(r);self.assertEqual(len(x),1);self.assertEqual(x.iloc[0].tweet_id,'1')
 def test_collection_delay_and_next_day(self):
  r=pd.DataFrame({'tweet_id':['1'],'text':['$DOGE example post'],'tweet_type':['tweet'],'created_at':['2025-01-01T23:00Z'],'first_posted_at':['2025-01-02T01:00Z'],'tickers':[['DOGE']]})
  x=select_posts(r).iloc[0];self.assertEqual(x.date,pd.Timestamp('2025-01-03',tz='UTC'));self.assertLess(x.available_at,x.date)
 def test_price_panel_tamper_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'bad.csv';p.write_text('bad data')
   with self.assertRaises(ValueError):load_prices(p)
 def test_temporal_purges_and_support(self):
  d=pd.DataFrame([{'date':x,'coin':c,'actual_up':i%2} for i,x in enumerate(pd.date_range('2024-03-01',periods=200,tz='UTC')) for c in COINS])
  tr,v,te=split_data(d)
  self.assertLess(tr.date.max()+pd.Timedelta(days=1),v.date.min());self.assertLess(v.date.max()+pd.Timedelta(days=1),te.date.min());self.assertEqual(set(te.coin),set(COINS))
 def test_inspection_does_not_require_sentiment_cache(self):
  import os,io,contextlib
  from expanded_inputs import inspect_posts
  raw=pd.DataFrame({'tweet_id':['1'],'text':['DOGE example post'],'tweet_type':['tweet'],'created_at':['2025-01-01'],'first_posted_at':['2025-01-01'],'tickers':[['DOGE']]})
  cwd=Path.cwd()
  with tempfile.TemporaryDirectory() as t:
   try:
    os.chdir(t);Path('data').mkdir();pd.DataFrame({'close':[1]}).to_csv('data/expanded-memecoin-daily.csv',index=False)
    with contextlib.redirect_stdout(io.StringIO()):result=inspect_posts(raw)
    self.assertEqual(len(result),1)
    self.assertFalse(Path('data/expanded-sentiment.json').exists())
   finally:os.chdir(cwd)
if __name__=='__main__':unittest.main()
