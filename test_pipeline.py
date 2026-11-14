"""Small runnable gates: .venv/bin/python -m unittest -v test_pipeline.py."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import pipeline as p


class PipelineChecks(unittest.TestCase):
    def test_market_features_cannot_see_today_or_future(self):
        prices = p.load_prices("DOGE")
        original = p.market_features(prices)
        changed = prices.copy()
        index = 80
        changed.loc[index:, ["open", "high", "low", "close", "volume"]] *= 3
        altered = p.market_features(changed)
        np.testing.assert_allclose(original.loc[index, p.MARKET].to_numpy(float), altered.loc[index, p.MARKET].to_numpy(float), rtol=0, atol=0)
        self.assertFalse(np.allclose(original.loc[index + 1, p.MARKET].to_numpy(float), altered.loc[index + 1, p.MARKET].to_numpy(float)))
        np.testing.assert_allclose(original.actual_return, prices.close / prices.open - 1)

    def test_source_availability_is_conservative(self):
        raw, selected = p.load_posts()
        self.assertFalse(raw.tweet_id.duplicated().any())
        self.assertTrue(selected.tweet_type.eq("tweet").all())
        self.assertTrue((selected.available_at >= selected.created_at).all())
        observed = selected.first_posted_at.notna()
        self.assertTrue((selected.loc[observed, "available_at"] >= selected.loc[observed, "first_posted_at"]).all())
        self.assertTrue((selected.available_at < selected.date).all())
        self.assertTrue((selected.available_at >= selected.date - pd.Timedelta(days=1)).all())
        self.assertFalse(selected.text.duplicated().any())
        self.assertTrue((selected.available_at >= p.START - pd.Timedelta(days=1)).all())

    def test_pinned_dataset_tampering_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "data").mkdir()
            (root / "data/posts.parquet").write_bytes(b"PAR1not the pinned dataset")
            with patch.object(p, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    p.load_posts()

    def test_missing_price_day_is_not_forward_filled(self):
        prices = p.load_prices("DOGE").drop(index=5)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "data").mkdir()
            prices.to_csv(root / "data/DOGE-prices.csv", index=False)
            with patch.object(p, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "Missing daily prices"):
                    p.load_prices("DOGE")
        with self.assertRaisesRegex(ValueError, "Unknown coin"):
            p.load_prices("../../secret")

    def test_split_boundaries_and_complete_coin_support(self):
        frame = pd.read_csv(p.ROOT / "data/features.csv")
        frame["date"] = pd.to_datetime(frame.date, utc=True)
        train, val, test = p.split_dates(frame)
        self.assertLess(train.date.max() + pd.Timedelta(days=1), val.date.min())
        self.assertLess(val.date.max() + pd.Timedelta(days=1), test.date.min())
        for a, b in [(train, val), (val, test), (train, test)]:
            self.assertFalse(set(a.date) & set(b.date))
        for part in (train, val, test):
            self.assertEqual(set(part.coin), set(p.COINS))
            self.assertTrue((part.groupby("date").coin.nunique() == 5).all())
            self.assertEqual(part.actual_up.nunique(), 2)
        self.assertTrue((frame.loc[frame.post_count.eq(0), p.SOCIAL] == 0).all().all())

    def test_report_metrics_match_saved_predictions(self):
        report = json.loads((p.ROOT / "docs/report.json").read_text())
        rows = pd.DataFrame(report["predictions"])
        self.assertEqual(len(rows), len(rows.drop_duplicates(["coin", "date"])))
        self.assertEqual(set(rows.coin), set(p.COINS))
        for metric in report["metrics"]:
            self.assertEqual(metric["n"], len(rows))
            self.assertEqual(metric["n_dates"], rows.date.nunique())
        for name, column in [("Market only", "market_prob"), ("Market + attention", "attention_prob"), ("Market + X", "social_prob")]:
            measured = p.scores(rows.actual_up, rows[column])
            saved = next(x for x in report["metrics"] if x["name"] == name)
            for key in measured:
                self.assertAlmostEqual(measured[key], saved[key], places=12)
        for row in report["predictions"]:
            cutoff = pd.Timestamp(row["cutoff"])
            self.assertEqual(pd.Timestamp(row["target_end"]) - cutoff, pd.Timedelta(days=1))
            self.assertEqual(row["actual_up"], int(row["actual_return"] > 0))
            self.assertEqual(row["sentiment"] is None, row["post_count"] == 0)
            for post in row["examples"]:
                available = pd.Timestamp(post["available_at"])
                self.assertLess(available, cutoff)
                self.assertGreaterEqual(available, cutoff - pd.Timedelta(days=1))

    def test_exported_json_coefficients_reproduce_predictions(self):
        exports = json.loads((p.ROOT / "artifacts/coefficients.json").read_text())
        report = json.loads((p.ROOT / "docs/report.json").read_text())
        frame = pd.read_csv(p.ROOT / "data/features.csv")
        frame["date"] = pd.to_datetime(frame.date, utc=True)
        _, _, test = p.split_dates(frame)
        for name, column in [("Market only", "market_prob"), ("Market + attention", "attention_prob"), ("Market + X", "social_prob")]:
            artifact = exports[name]
            names = artifact["feature_names"]
            numeric = [key.removeprefix("numeric__") for key in names if key.startswith("numeric__")]
            matrix = (test[numeric].to_numpy() - artifact["numeric_mean"]) / artifact["numeric_scale"]
            categories = [key.removeprefix("coin__coin_") for key in names if key.startswith("coin__")]
            onehot = np.column_stack([test.coin.eq(coin).to_numpy(float) for coin in categories])
            logits = np.column_stack([matrix, onehot]) @ artifact["coefficients"] + artifact["intercept"][0]
            prob = 1 / (1 + np.exp(-logits))
            np.testing.assert_allclose(prob, [r[column] for r in report["predictions"]], atol=1e-12, rtol=1e-12)

    def test_manifest_binds_code_inputs_and_published_outputs(self):
        manifest = json.loads((p.ROOT / "artifacts/manifest.json").read_text())
        self.assertEqual(manifest["code_sha256"], p.digest(p.ROOT / "pipeline.py"))
        for key in ["docs/report.json", "artifacts/coefficients.json"]:
            self.assertIn(key, manifest["files"])
        for filename, checksum in manifest["files"].items():
            self.assertEqual(checksum, p.digest(p.ROOT / filename))

    def test_empty_subgroup_and_invalid_probabilities(self):
        self.assertEqual(p.scores([], [])["n"], 0)
        self.assertIsNone(p.scores([], [])["accuracy"])
        for values in ([np.nan], [-0.1], [1.1]):
            with self.assertRaisesRegex(ValueError, "Invalid forecast"):
                p.scores([1], values)


if __name__ == "__main__":
    unittest.main()
