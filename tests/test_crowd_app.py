import copy,json,unittest
from sample_pipeline import ROOT,dataset,historical_csv,infer
from crowd_app import build_app,panels
class CrowdAppTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.artifact=json.loads((ROOT/"artifacts/crowd-model.json").read_text());cls.row=dataset()[2].iloc[0]
 def test_real_widget_uses_same_batch_and_model(self):
  ui,state,c=build_app(self.artifact);c["format"].value="csv";c["coin"].value=self.row.coin;c["cutoff"].value=self.row.cutoff.isoformat();c["posts"].value=historical_csv(self.row);c["button"].click()
  self.assertIn("result",state,c["error"].value);r=state["result"]
  self.assertAlmostEqual(r["probability_up"],infer(self.artifact["models"]["Model 2"],self.row.to_dict())["probability_up"])
  self.assertEqual(r["data_quality"]["unknown_timestamps"],0);self.assertIn("No demonstrated",c["evidence"].value);self.assertIn("Target UTC",c["forecast"].value)
  c["posts"].value="";c["button"].click();self.assertNotIn("result",state);self.assertIn("blocked",c["error"].value)
 def test_corrupt_model_clears_previous_forecast(self):
  artifact=copy.deepcopy(self.artifact);ui,state,c=build_app(artifact)
  c['format'].value='csv';c['coin'].value=self.row.coin;c['cutoff'].value=self.row.cutoff.isoformat();c['posts'].value=historical_csv(self.row)
  c['button'].click();self.assertIn('result',state)
  artifact['models']['Model 2']['intercept']=float('nan');c['button'].click()
  self.assertNotIn('result',state);self.assertEqual(c['forecast'].value,'');self.assertIn('blocked',c['error'].value)
 def test_public_word_reasons_are_html_escaped(self):
  ui,state,c=build_app(self.artifact);c["format"].value="csv";c["coin"].value=self.row.coin;c["cutoff"].value=self.row.cutoff.isoformat();c["posts"].value=historical_csv(self.row);c["button"].click()
  r=copy.deepcopy(state["result"]);r["toward_up"]=[{"word":"<script>unsafe</script>","logit_contribution":1}]
  text="".join(panels(r,self.artifact["report"]));self.assertIn("&lt;script&gt;",text);self.assertNotIn("<script>",text)
if __name__=="__main__":unittest.main()
