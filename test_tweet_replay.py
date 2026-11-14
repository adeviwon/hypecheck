"""Verify the saved real-tweet diagnostic against the frozen report and input bytes."""
import hashlib
import json
from pathlib import Path
import unittest
from replay_bearish_tweets import predict
ROOT = Path(__file__).resolve().parent

class TweetReplayChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ledger = json.loads((ROOT/'tweet-tests/bearish-replay.json').read_text())
        cls.report = json.loads((ROOT/'docs/report.json').read_text())

    def test_provenance(self):
        for name, expected in self.ledger['summary']['input_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), expected, name)

    def test_summary_is_derived_from_all_records(self):
        rows = self.ledger['records']; s = self.ledger['summary']
        self.assertEqual(s['sampled'], len(rows))
        self.assertEqual(s['single_post_correct'], sum(r['direction_result']=='correct' for r in rows))
        self.assertEqual(s['single_post_incorrect'], sum(r['direction_result']=='incorrect' for r in rows))
        self.assertEqual(s['unique_target_dates'], len(set(r['cutoff'] for r in rows)))
        self.assertEqual(len(rows), len(set((r['coin'],r['cutoff']) for r in rows)))
        self.assertFalse(s['new_independent_holdout'])

    def test_outcomes_and_full_context_match_original_frozen_report(self):
        source = {(r['coin'], r['cutoff']):r for r in self.report['predictions']}
        for row in self.ledger['records']:
            actual = source[(row['coin'],row['cutoff'])]
            self.assertEqual(row['actual_return'],actual['actual_return'])
            self.assertAlmostEqual(row['daily_full_context_up_probability'],actual['social_prob'],places=12)
            self.assertLess(row['available_at'],row['cutoff'])
            expected = (row['single_post_up_probability']>=0.5)==bool(actual['actual_up'])
            self.assertEqual(row['direction_result']=='correct',expected)

    def test_predict_uses_declared_features_only(self):
        export = {'intercept':[0.0], 'feature_names':['numeric__x','coin__coin_DOGE'],
                  'coefficients':[1.0,1.0], 'numeric_mean':[0.0], 'numeric_scale':[1.0]}
        # Outcome columns must not influence inference.
        self.assertEqual(predict(export, {'x':0,'coin':'PEPE','actual_up':1}),0.5)
        self.assertEqual(predict(export, {'x':0,'coin':'PEPE','actual_up':0}),0.5)
        self.assertGreater(predict(export, {'x':0,'coin':'DOGE'}),0.5)

if __name__=='__main__':unittest.main()
