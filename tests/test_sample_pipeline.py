import json,unittest
import numpy as np
import pandas as pd
from sample_pipeline import *
class SampleTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.frame,cls.train,cls.later,cls.unseen=dataset();cls.a=json.loads((ROOT/"artifacts/crowd-model.json").read_text())
 def test_split_really_holds_out_assets_and_dates(self):
  self.assertTrue(set(self.train.coin)<=set(BASE_COINS));self.assertTrue(set(self.unseen.coin)<=set(NEW_COINS))
  self.assertLess(self.train.date.max()+pd.Timedelta(days=1),self.later.date.min())
  self.assertFalse(set(i for ids in self.train.source_ids for i in ids)&set(i for ids in self.unseen.source_ids for i in ids))
 def test_real_labels_and_snapshot_times(self):
  _,posts,prices,_=load();times=posts.set_index(posts.tweet_id.astype(str)).available_at
  for row in self.frame.itertuples():
   available=times.loc[row.source_ids]
   self.assertTrue((available<=row.cutoff).all());self.assertTrue((available>=row.cutoff-pd.Timedelta(hours=24)).all())
   self.assertEqual(row.date,row.cutoff.floor("D")+pd.Timedelta(days=1))
   bar=prices[row.coin].set_index("date").loc[row.date]
   self.assertEqual(row.actual_up,int(bar.close>bar.open))
 def test_coin_day_weights_do_not_multiply_evidence(self):
  x=self.later.copy();x["w"]=weight(x);self.assertTrue(np.allclose(x.groupby(["date","coin"]).w.sum(),1))
 def test_safe_artifact_matches_actual_prediction_ledger(self):
  p=pd.read_csv(ROOT/"artifacts/crowd-predictions.csv")
  for cohort,frame in [("later_time",self.later),("unseen_coin_market",self.unseen)]:
   reference=p[p.test.eq(cohort)]
   for name,saved in self.a["models"].items():
    actual=np.array([infer(saved,row.to_dict())["probability_up"] for _,row in frame.iterrows()])
    self.assertTrue(np.allclose(actual,reference[name],atol=1e-10))
 def test_sample_hyperparameters_and_same_word_vocabulary(self):
  one=self.a["models"]["Model 1"];two=self.a["models"]["Model 2"];char=self.a["models"]["Character trial"]
  self.assertEqual(one["blocks"][0]["vocabulary"],two["blocks"][0]["vocabulary"])
  self.assertEqual(one["estimator"],{"C":1.0,"class_weight":None,"max_iter":1000})
  self.assertEqual(two["estimator"],{"C":1.0,"class_weight":"balanced","max_iter":1000})
  self.assertEqual(char["estimator"],{"C":10,"class_weight":"balanced","max_iter":1000})
  self.assertEqual(char["blocks"][0]["analyzer"],"char_wb");self.assertEqual(char["blocks"][0]["ngram_range"],[2,5]);self.assertTrue(char["blocks"][0]["sublinear_tf"])
 def test_contributions_reconstruct_logit(self):
  row=self.later.iloc[0].to_dict();r=infer(self.a["models"]["Model 2"],row)
  self.assertTrue(all(x["word"] in row["text"].lower() for x in r["toward_up"]+r["toward_down"]))
  self.assertTrue(0<r["probability_up"]<1)
 def test_corrupt_model_or_features_block_forecasting(self):
  import copy
  row=self.later.iloc[0].to_dict()
  for value in [float('nan'),float('inf'),-float('inf')]:
   for field in ['intercept','coefficient','idf','mean','scale','feature']:
    with self.subTest(field=field,value=value):
     saved=copy.deepcopy(self.a['models']['Model 2']);features=dict(row)
     if field=='intercept':saved[field]=value
     elif field=='feature':features[SOCIAL[0]]=value
     else:
      block=next(b for b in saved['blocks'] if field in b);block[field][0]=value
     with self.assertRaises(ValueError):infer(saved,features)
  saved=copy.deepcopy(self.a['models']['Model 2'])
  next(b for b in saved['blocks'] if b['name']=='numeric')['scale'][0]=0
  with self.assertRaises(ValueError):infer(saved,row)
  saved['blocks']=[]
  with self.assertRaises(ValueError):infer(saved,row)
 def test_invalid_sentiment_is_not_silently_faked(self):
  with self.assertRaises(ValueError):batch_features(["Example public text"],[{"bullish":2,"bearish":0,"neutral":0}],"DOGE")
 def test_metrics_are_reproduced_from_real_ledger(self):
  ledger=pd.read_csv(ROOT/"artifacts/crowd-predictions.csv")
  for cohort,frame in [("later_time",self.later),("unseen_coin_market",self.unseen)]:
   for name,m in self.a["report"]["tests"][cohort].items():
    actual=metrics(frame,ledger[ledger.test.eq(cohort)][name].to_numpy())
    self.assertAlmostEqual(actual["accuracy"],m["accuracy"]);self.assertAlmostEqual(actual["up_recall"],m["up_recall"])
 def test_random_split_is_flagged_diagnostic(self):
  d=self.a["report"]["random_80_20_diagnostic"]
  self.assertGreater(d["shared_train_test_dates"],0);self.assertIn("NOT a valid",d["status"])
 def test_historical_csv_input_matches_training_features(self):
  for _,row in self.later.head(5).iterrows():
   csv=historical_csv(row)
   snapshot=parse_csv(csv,row.coin,row.cutoff.isoformat())
   texts=[p["text"] for p in snapshot.posts]
   self.assertEqual(texts,row.texts)
   self.assertEqual(snapshot.audit["unknown_timestamps"],0)
   actual=batch_features(texts,row.sentiment_scores,row.coin)
   for key in SOCIAL:self.assertAlmostEqual(actual[key],row[key])
if __name__=="__main__":unittest.main()
