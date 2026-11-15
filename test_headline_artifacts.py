"""Verify measured deliverables and safe parameters on the actual expanded panel."""
import json,unittest
from pathlib import Path
import numpy as np
import pandas as pd
import pipeline as core
from headline_study import previous_day
from expanded_inputs import load,split_data
ROOT=Path(__file__).resolve().parent
class ArtifactTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.r=json.loads((ROOT/'docs/headline-report.json').read_text());cls.p=pd.read_csv(ROOT/'artifacts/headline-predictions.csv')
  _,cls.posts,cls.prices,cls.frame=load();_,_,cls.test=split_data(cls.frame)
 def test_same_rows_and_exact_metrics(self):
  y=self.p.actual_up.to_numpy()
  for name,row in self.r['metrics'].items():
   p=self.p[name].to_numpy();self.assertEqual(row['n'],len(y));self.assertAlmostEqual(row['accuracy'],float(((p>=.5)==y).mean()));self.assertAlmostEqual(row['brier'],float(np.mean((p-y)**2)))
 def test_actual_baseline_from_raw_bars(self):
  p=self.p.copy();p['date']=pd.to_datetime(p.date,utc=True)
  for coin in self.r['protocol']['coins']:
   rows=p[p.coin.eq(coin)];values=rows.date.map(previous_day(self.prices[coin])).to_numpy();self.assertTrue(np.array_equal(rows.previous_day.to_numpy(),values))
 def test_kind_sample_counts(self):
  for row in self.r['kinds']:
   self.assertEqual(row['all_test_rows']['n'],len(self.p))
   for group in row['kind_present_rows'].values():self.assertEqual(group['n'],row['test_coin_days_with_kind'])
 def test_notebook_has_real_outputs_and_sample_structure(self):
  nb=json.loads((ROOT/'HypeCheck.ipynb').read_text());cells=[c for c in nb['cells'] if c['cell_type']=='code' and ''.join(c['source']).strip()]
  self.assertEqual(len(cells),nb['metadata']['hypecheck']['expected_code_cells']);self.assertFalse(nb['metadata']['hypecheck']['quota_sampling'])
  self.assertTrue(all(c.get('outputs') and c.get('execution_count') for c in cells))
  for c in cells:self.assertFalse(any(x['output_type']=='error' for x in c['outputs']))
  text=''.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='markdown')
  for s in ['## 1. Load','## 3. Split','## 4. Baselines','## 5. Model','## 6. Where','## 7. Our addition','## 9. The demo','## 10. Results']:self.assertIn(s,text)
 def test_published_prediction_json_matches_csv(self):
  records=json.loads((ROOT/'docs/headline-predictions.json').read_text());self.assertEqual(len(records),len(self.p))
  for a,b in zip(records,self.p.to_dict('records')):
   self.assertEqual(a['date'],b['date']);self.assertEqual(a['coin'],b['coin']);self.assertAlmostEqual(a['Sentiment + coin'],b['Sentiment + coin'],places=9)
 def test_saved_safe_model_matches_fresh_features(self):
  test=self.test;a=json.loads((ROOT/'artifacts/headline-model.json').read_text())
  numeric=(test[a['features']].to_numpy()-np.array(a['numeric_mean']))/np.array(a['numeric_scale'])
  keys=a['feature_names'][len(a['features']):];cats=np.column_stack([(test.coin==key.removeprefix('coin__coin_')).astype(float).to_numpy() for key in keys])
  x=np.column_stack([numeric,cats]);logit=x@np.array(a['coefficient'])[0]+a['intercept'][0];p=1/(1+np.exp(-logit))
  self.assertTrue(np.allclose(p,self.p['Sentiment + coin'],atol=1e-12))
 def test_natural_class_counts_are_real_post_counts(self):
  actual=self.posts[['bullish','bearish','neutral']].idxmax(axis=1).value_counts().to_dict()
  self.assertEqual(self.r['tweet_sentiment_counts'],actual);self.assertEqual(sum(actual.values()),len(self.posts));self.assertGreater(len(self.p),965)
if __name__=='__main__':unittest.main()
