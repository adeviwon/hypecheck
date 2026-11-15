import json,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from tweet_forecast import *
class ForecastTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.a=json.loads((ROOT/'artifacts/tweet-forecast-model.json').read_text());cls.r=json.loads((ROOT/'docs/tweet-forecast-report.json').read_text())
 def test_json_models_are_real_fitted_weights(self):
  for m in self.a['models'].values():
   for head in m.values():
    self.assertGreater(sum(len(b['coefficient']) for b in head['blocks']),100)
    self.assertTrue(all(np.isfinite(b['coefficient']).all() for b in head['blocks']))
 def test_inference_returns_bounded_probability(self):
  for coin in COINS:
   for context in ['', 'Earlier public discussion about price and volume']:
    p=predict('Example bullish public post about the coin',coin,context,self.a)
    self.assertTrue(0<p['probability_up']<1);self.assertTrue(np.isfinite(p['estimated_return_percent']))
    self.assertEqual(p['context_mode'],'tweet_and_context' if context else 'tweet_only')
 def test_invalid_inputs(self):
  for text,coin,context in [('', 'DOGE',''),('x'*4001,'DOGE',''),('Long enough post','BTC',''),('Long enough post','DOGE','x'*20001)]:
   with self.assertRaises(ValueError):predict(text,coin,context,self.a)
 def test_text_changes_forecast(self):
  a=predict('Strong bullish breakout buy the dip and moon','DOGE',artifact=self.a)
  b=predict('Bearish collapse sell everything support broken','DOGE',artifact=self.a)
  self.assertNotEqual(a['probability_up'],b['probability_up'])
 def test_training_boundaries_and_test_counts(self):
  parts=self.r['partitions'];self.assertLess(parts['train']['last'],parts['validation']['first']);self.assertLess(parts['validation']['last'],parts['test']['first'])
  p=pd.read_csv(ROOT/'artifacts/tweet-forecast-predictions.csv');self.assertEqual(len(p),parts['test']['events'])
  for name,metric in self.r['metrics'].items():
   prob=p[name].to_numpy();w=weights(p);self.assertAlmostEqual(metric['accuracy'],float(np.average((prob>=.5)==p.actual_up,weights=w)))
 def test_target_and_context_are_time_correct(self):
  _,posts,prices,_=load();e=build_examples(posts,prices)
  self.assertTrue((e.date==e.available_at.dt.floor('D')+pd.Timedelta(days=1)).all())
  self.assertTrue((e.context_latest.isna()|(e.context_latest<e.available_at)).all())
  for row in e.iloc[::100].itertuples():
   bar=prices[row.coin].set_index('date').loc[row.date]
   self.assertEqual(row.actual_up,int(bar.close>bar.open));self.assertAlmostEqual(row.log_return,np.log(bar.close/bar.open))
 def test_coin_day_total_weight_is_one(self):
  p=pd.read_csv(ROOT/'artifacts/tweet-forecast-predictions.csv');p['weight']=weights(p)
  self.assertTrue(np.allclose(p.groupby(['date','coin']).weight.sum(),1))
 def test_context_unavailable_is_not_claimed_live(self):
  p=predict('Public example post with sufficient length','PEPE',artifact=self.a)
  self.assertIn('No live X feed',p['warning']);self.assertIn('causal',p['warning'])
if __name__=='__main__':unittest.main()
