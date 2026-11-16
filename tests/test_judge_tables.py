import copy,json,unittest
from pathlib import Path
from judge_tables import *
ROOT=Path(__file__).resolve().parents[1]
class JudgeTableTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.r=json.loads((ROOT/"docs/crowd-report.json").read_text())
 def test_results_use_actual_percentages(self):
  t=results_table(self.r)
  for index,(name,row) in enumerate(self.r["tests"]["later_time"].items()):
   self.assertEqual(t.iloc[index]["Later dates — correct"],format(row["accuracy"],".2%"))
 def test_report_is_not_modified(self):
  before=copy.deepcopy(self.r);results_table(self.r);scores_table(self.r["tests"]["later_time"]);overall_findings(self.r);self.assertEqual(before,self.r)
 def test_example_data_is_plain_language(self):
  frame=pd.DataFrame([{"cutoff":"2026-01-01T12:00:00Z","date":"2026-01-02T00:00:00Z","coin":"DOGE","post_count":2,"bullish":.95,"bearish":.02,"actual_up":0,"actual_return":-.01,"text":"Line one\n\nLine two"}])
  out=examples_table(frame);self.assertEqual(out.iloc[0]["Bullish language"],"95.0%");self.assertEqual(out.iloc[0]["Actual next-day result"],"Fell (-1.00%)");self.assertNotIn("actual_up",out.columns);self.assertNotIn("\n",out.iloc[0]["Example wording"])
  self.assertIn("NOT the probability",examples_findings(frame));self.assertIn("not independent",examples_findings(frame))
 def test_false_precision_is_not_assigned_when_no_up_calls(self):
  t=scores_table({"Training majority":self.r["tests"]["later_time"]["Training majority"]});self.assertEqual(t.iloc[0]["UP calls that were right"],"No UP calls")
 def test_findings_preserve_negative_result(self):
  text=overall_findings(self.r);self.assertIn(format(self.r["tests"]["later_time"]["Model 2"]["accuracy"],".2%"),text);self.assertIn("did not add useful",text);self.assertIn("does not prove",text)
 def test_confusion_table_has_labels_not_class_codes(self):
  t=confusion_table(self.r["tests"]["later_time"]["Model 1"]);self.assertEqual(t.iloc[0,0],"Did not rise");self.assertEqual(t.iloc[1,0],"Rose")
if __name__=="__main__":unittest.main()
