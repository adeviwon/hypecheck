"""Verify the real measured deliverables and safe model parameters."""
import json,unittest
from pathlib import Path
import numpy as np
import pandas as pd
import pipeline as core
from headline_study import previous_day
ROOT=Path(__file__).resolve().parent
class ArtifactTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.r=json.loads((ROOT/'docs/headline-report.json').read_text());cls.p=pd.read_csv(ROOT/'artifacts/headline-predictions.csv')
 def test_same_rows_and_exact_metrics(self):
  y=self.p.actual_up.to_numpy()
  for name,row in self.r['metrics'].items():
   p=self.p[name].to_numpy();self.assertEqual(row['n'],len(y))
   self.assertAlmostEqual(row['accuracy'],float(((p>=.5)==y).mean()))
   self.assertAlmostEqual(row['brier'],float(np.mean((p-y)**2)))
 def test_actual_baseline_from_raw_bars(self):
  p=self.p.copy();p['date']=pd.to_datetime(p.date,utc=True)
  for coin in core.COINS:
   rows=p[p.coin.eq(coin)];values=rows.date.map(previous_day(core.load_prices(coin))).to_numpy()
   self.assertTrue(np.array_equal(rows.previous_day.to_numpy(),values))
 def test_kind_sample_counts(self):
  for row in self.r['kinds']:
   self.assertEqual(row['all_test_rows']['n'],len(self.p))
   for group in row['kind_present_rows'].values():self.assertEqual(group['n'],row['test_coin_days_with_kind'])
 def test_notebook_has_real_outputs(self):
  nb=json.loads((ROOT/'HypeCheck.ipynb').read_text());cells=[c for c in nb['cells'] if c['cell_type']=='code' and ''.join(c['source']).strip()]
  self.assertEqual(len(cells),10)
  self.assertTrue(all(c.get('outputs') and c.get('execution_count') for c in cells))
  for c in cells:self.assertFalse(any(x['output_type']=='error' for x in c['outputs']))
 def test_published_prediction_json_matches_csv(self):
  records=json.loads((ROOT/'docs/headline-predictions.json').read_text())
  self.assertEqual(len(records),len(self.p))
  for a,b in zip(records,self.p.to_dict('records')):
   self.assertEqual(a['date'],b['date']);self.assertEqual(a['coin'],b['coin'])
   self.assertAlmostEqual(a['Sentiment + coin'],b['Sentiment + coin'],places=9)
 def test_saved_safe_model_matches_fresh_features(self):
  _,posts=core.load_posts();posts=core.score_posts(posts)
  prices={c:core.load_prices(c) for c in core.COINS};hourly={c:core.load_hourly(c) for c in core.COINS}
  frame,_=core.build_dataset(posts,prices,hourly,core.load_btc());_,_,test=core.split_dates(frame)
  a=json.loads((ROOT/'artifacts/headline-model.json').read_text())
  numeric=(test[a['features']].to_numpy()-np.array(a['numeric_mean']))/np.array(a['numeric_scale'])
  keys=a['feature_names'][len(a['features']):]
  cats=np.column_stack([(test.coin==key.removeprefix('coin__coin_')).astype(float).to_numpy() for key in keys])
  x=np.column_stack([numeric,cats]);logit=x@np.array(a['coefficient'])[0]+a['intercept'][0]
  p=1/(1+np.exp(-logit));self.assertTrue(np.allclose(p,self.p['Sentiment + coin'],atol=1e-12))
if __name__=='__main__':unittest.main()
