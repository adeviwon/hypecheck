import unittest
from datetime import datetime,timezone
from crowd_snapshot import build_snapshot,parse_pasted,parse_csv
CUT="2026-11-15T12:00:00Z"
class SnapshotTests(unittest.TestCase):
 def test_paste_separates_posts_not_single_line_breaks(self):
  s=parse_pasted("A public post\nwith a second line\n\nAnother public post", "DOGE", CUT)
  self.assertEqual(len(s.posts),2);self.assertEqual(s.audit["timing_status"],"unverified_user_supplied")
  self.assertIsNone(s.posts[0]["available_at_utc"])
 def test_window_and_exact_forecast_dates(self):
  s=build_snapshot([{"text":"A public post","published_at_utc":"2026-11-14T12:00:00Z"},{"text":"Another public post","published_at_utc":CUT},{"text":"An older excluded post","published_at_utc":"2026-11-14T11:59:59Z"}],"DOGE",CUT)
  self.assertEqual(len(s.posts),2);self.assertEqual(s.audit["excluded_outside_window"],1)
  self.assertEqual(s.target_start_utc,"2026-11-16T00:00:00Z");self.assertEqual(s.target_end_utc,"2026-11-17T00:00:00Z")
 def test_future_record_rejected(self):
  with self.assertRaisesRegex(ValueError,"future"):build_snapshot([{"text":"A public post","published_at_utc":"2026-11-15T12:00:01Z"}],"DOGE",CUT)
 def test_later_archive_availability_is_used(self):
  with self.assertRaisesRegex(ValueError,"future"):build_snapshot([{"text":"A public post","published_at_utc":"2026-11-14T12:00:00Z","available_at_utc":"2026-11-15T13:00:00Z"}],"DOGE",CUT)
 def test_duplicates_do_not_inflate_count(self):
  s=parse_pasted("A public post\n\nA public post","PEPE",CUT)
  self.assertEqual(len(s.posts),1);self.assertEqual(s.audit["duplicates_removed"],1)
 def test_timezone_offsets_are_converted(self):
  s=build_snapshot([{"text":"A public post","published_at_utc":"2026-11-15T20:00:00+08:00"}],"DOGE",CUT)
  self.assertEqual(s.posts[0]["available_at_utc"],CUT)
 def test_naive_timestamp_rejected(self):
  with self.assertRaisesRegex(ValueError,"timezone"):parse_pasted("A public post","DOGE","2026-11-15T12:00:00")
 def test_no_global_coverage_or_authenticity_claim(self):
  s=parse_csv("text,published_at_utc\nA public post,2026-11-15T11:00:00Z\n","DOGE",CUT)
  self.assertEqual(s.audit["timing_status"],"provided_timestamps_checked")
  self.assertFalse(s.audit["source_authenticity_verified"]);self.assertEqual(s.audit["platform_coverage"],"unknown")
 def test_empty_unknown_coin_and_limits(self):
  for text,coin in [("","DOGE"),("tiny","DOGE"),("A public post","BTC"),("x"*4001,"DOGE")]:
   with self.assertRaises(ValueError):parse_pasted(text,coin,CUT)
  with self.assertRaises(ValueError):build_snapshot([{"text":"A public post"}]*201,"DOGE",CUT)
 def test_csv_requires_text(self):
  with self.assertRaises(ValueError):parse_csv("sentiment\nbullish\n","DOGE",CUT)
 def test_no_usable_posts_after_window_filter(self):
  with self.assertRaises(ValueError):build_snapshot([{"text":"An older public post","published_at_utc":"2026-11-13T12:00:00Z"}],"DOGE",CUT)
 def test_invalid_falsey_timestamps_are_rejected(self):
  for field in ["published_at_utc","available_at_utc"]:
   for value in [0,False,[],{}]:
    with self.subTest(field=field,value=value):
     with self.assertRaises(ValueError):build_snapshot([{"text":"A public post",field:value}],"DOGE",CUT)
 def test_optional_blank_timestamps_stay_missing(self):
  for value in [None,"","  "]:
   with self.subTest(value=value):
    s=build_snapshot([{"text":"A public post","published_at_utc":value}],"DOGE",CUT)
    self.assertIsNone(s.posts[0]["available_at_utc"])
    self.assertEqual(s.audit["timing_status"],"unverified_user_supplied")
if __name__=="__main__":unittest.main()
