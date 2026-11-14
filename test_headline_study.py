import unittest
import numpy as np
import pandas as pd
from headline_study import headline_kind, previous_day, aggregate_kinds, paired_interval, holm
class HeadlineTests(unittest.TestCase):
 def test_taxonomy(self):
  self.assertEqual(headline_kind('ETF approved for exchange listing'),'events')
  self.assertEqual(headline_kind('Buy now 100x to the moon'),'promotion')
  self.assertEqual(headline_kind('Rejected at resistance, support below'),'technical')
  self.assertEqual(headline_kind('Thinking about the weekend'),'other')
 def test_exclusive_and_case(self):
  self.assertEqual(headline_kind('BREAKOUT after ETF approval'),'events')
 def test_previous_direction_is_open_close_not_close_close(self):
  prices=pd.DataFrame({'date':pd.date_range('2025-01-01',periods=3,tz='UTC'),'open':[100,90,110],'close':[95,100,105]})
  out=previous_day(prices)
  self.assertTrue(pd.isna(out.iloc[0]));self.assertEqual(out.iloc[1],0);self.assertEqual(out.iloc[2],1)
 def test_aggregation_uses_availability_day(self):
  p=pd.DataFrame({'available_at':pd.to_datetime(['2025-01-01T23:59Z','2025-01-02T00:00Z']),
    'text':['breakout','ETF approval'],'tickers':[['DOGE'],['DOGE']],'sentiment':[.5,-.5]})
  g=aggregate_kinds(p)
  self.assertEqual(g.loc[('DOGE',pd.Timestamp('2025-01-02',tz='UTC')),'technical_count'],1)
  self.assertEqual(g.loc[('DOGE',pd.Timestamp('2025-01-03',tz='UTC')),'events_count'],1)
 def test_paired_identical(self):
  d=pd.date_range('2025-01-01',periods=20,tz='UTC');x=np.ones(20)
  r=paired_interval(d,x-x,reps=100);self.assertEqual(r['interval95'],[0.,0.])
 def test_holm(self):
  q=holm([.01,.04,.03]);self.assertTrue(np.allclose(q,[.03,.06,.06]))
if __name__=='__main__':unittest.main()
