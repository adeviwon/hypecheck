"""Focused tests for the notebook's 'enter tweets here' block."""
import json
import unittest
from pathlib import Path

import tweet_demo as td

ARTIFACT = json.loads((Path(__file__).resolve().parent / "artifacts" / "crowd-model.json").read_text())
COINS = ARTIFACT["report"]["base_coins"] + ARTIFACT["report"]["excluded_training_coins"]


class TweetDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.known = td.examples()
        cls.entries = [f"{row['coin']} | {row['text']}" for row in cls.known]

    def test_forty_real_held_out_posts(self):
        self.assertEqual(len(self.known), 40)
        self.assertEqual(sum(row["group"] == "later date" for row in self.known), 20)
        self.assertEqual(sum(row["group"] == "unseen coin" for row in self.known), 20)
        self.assertGreaterEqual(len({row["coin"] for row in self.known}), 10)
        for row in self.known:
            self.assertGreaterEqual(len(row["text"]), 30)
            self.assertIn(row["actual_up"], (0, 1))
            if row["group"] == "unseen coin":
                self.assertIn(row["coin"], ARTIFACT["report"]["excluded_training_coins"])
            else:
                self.assertIn(row["coin"], ARTIFACT["report"]["base_coins"])

    def test_held_out_posts_come_from_after_training(self):
        _, _, later, unseen = td.sp.dataset()
        training_last = td.sp.dataset()[1].date.max()
        for part in (later, unseen):
            self.assertTrue((part.date > training_last).all(), "a held-out post is not after training")

    def test_list_entry_is_one_post_and_prefix_is_stripped_exactly(self):
        items = [td._item(entry, "DOGE", COINS, number) for number, entry in enumerate(self.entries, 1)]
        self.assertEqual(len(items), 40)
        for item, row in zip(items, self.known):
            self.assertEqual(item["text"], row["text"])
            self.assertEqual(item["coin"], row["coin"])

    def test_coin_falls_back_to_cashtag_then_default(self):
        self.assertEqual(td._item("$PEPE about to send it again", "DOGE", COINS, 1)["coin"], "PEPE")
        self.assertEqual(td._item("no ticker at all in this post", "DOGE", COINS, 1)["coin"], "DOGE")
        self.assertEqual(td._item("TRUMP | $TRUMP has the volume today", "DOGE", COINS, 1)["coin"], "TRUMP")

    def test_explicit_default_coin_overrides_conflicting_cashtag(self):
        text = "Traders are discussing $PEPE today."
        entry = f"DOGE | {text}"
        expected = {"coin": "DOGE", "text": text}
        self.assertEqual(td._item(entry, "DOGE", COINS, 1), expected)
        self.assertEqual(td.parse_lines(entry, "DOGE", COINS), [expected])

    def test_scoring_matches_the_real_outcome(self):
        table, summary, rated = td.run_block(self.entries, "DOGE", ARTIFACT, known=self.known)
        self.assertEqual(summary["scored"], 40)
        self.assertEqual(summary["total"], 40)
        for row in rated:
            self.assertTrue(row["known"])
            self.assertEqual(row["model_right"], int((row["probability_up"] >= 0.5) == bool(row["actual_up"])))
            self.assertEqual(row["baseline_right"], int((row["baseline_probability_up"] >= 0.5) == bool(row["actual_up"])))
        self.assertEqual(summary["model_correct"], sum(row["model_right"] for row in rated))
        self.assertEqual(summary["model_accuracy"], summary["model_correct"] / 40)
        self.assertEqual(len(table), 40)
        self.assertIn("Model right?", table.columns)

    def test_user_posts_are_rated_but_never_scored(self):
        table, summary, rated = td.run_block(["$PEPE looking strong today, loading up before the next leg"], "DOGE", ARTIFACT, known=self.known)
        self.assertEqual(summary["rated"], 1)
        self.assertEqual(summary["scored"], 0)
        self.assertFalse(rated[0]["known"])
        self.assertNotIn("model_accuracy", summary)
        self.assertEqual(table.iloc[0]["Market actually did"], "unknown (your post)")

    def test_known_text_for_another_coin_has_no_outcome(self):
        row = next(example for example in self.known if example["coin"] == "BONK")
        table, summary, rated = td.run_block([f"SHIB | {row['text']}"], "DOGE", ARTIFACT, known=self.known)
        self.assertEqual(summary, {"rated": 1, "scored": 0})
        self.assertEqual(rated[0]["coin"], "SHIB")
        self.assertIsNone(rated[0]["actual_up"])
        self.assertEqual(table.iloc[0]["Market actually did"], "unknown (your post)")

    def test_interval_is_a_real_interval(self):
        low, high = td.wilson(14, 40)
        self.assertLess(low, 14 / 40)
        self.assertGreater(high, 14 / 40)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)

    def test_empty_and_too_short_input_are_rejected(self):
        for bad in ([], ["   "], ["tiny"]):
            with self.assertRaises(ValueError):
                td.run_block(bad, "DOGE", ARTIFACT, known=self.known)

    def test_no_causal_or_trading_claim_in_the_block(self):
        text = Path("tweet_demo.py").read_text().lower()
        for phrase in ["guaranteed", "profit", "will rise", "caused the price"]:
            self.assertNotIn(phrase, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
