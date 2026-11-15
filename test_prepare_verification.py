"""Preparation must preserve the original scientific manifest, never relax it."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
import prepare_verification as setup


class PreparationTests(unittest.TestCase):
    def fixture(self, folder):
        root = Path(folder)
        (root / "data").mkdir()
        (root / "artifacts").mkdir()
        scores = {"1": {"bullish": 0.2, "bearish": 0.3, "neutral": 0.5}}
        posts = pd.DataFrame({"tweet_id": [1], "text": ["real fixture text"]})
        signature = hashlib.sha256((setup.core.MODEL_REV + "|max_length=128|1real fixture text").encode()).hexdigest()
        payload = json.dumps({"signature": signature, "scores": scores}, ensure_ascii=False, allow_nan=False, indent=2).encode()
        (root / "artifacts/manifest.json").write_text(json.dumps({"files": {"data/sentiment.json": hashlib.sha256(payload).hexdigest()}}))
        source = json.dumps({"scores": scores}).encode()
        (root / "data/expanded-sentiment.json").write_bytes(source)
        return root, posts, source, payload

    def test_exact_reconstruction_then_offline_idempotence(self):
        with tempfile.TemporaryDirectory() as folder:
            root, posts, source, payload = self.fixture(folder)
            with patch.object(setup, "ROOT", root), patch.dict(setup.FILE_DIGESTS, {"data/expanded-sentiment.json": hashlib.sha256(source).hexdigest()}), patch.object(setup.core, "load_posts", return_value=(None, posts)) as load:
                target = setup.prepare()
                self.assertEqual(target.read_bytes(), payload)
                setup.prepare()
                self.assertEqual(load.call_count, 1)

    def test_tampered_source_fails_without_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            root, _, _, _ = self.fixture(folder)
            with patch.object(setup, "ROOT", root), self.assertRaisesRegex(ValueError, "Expanded.*SHA-256"):
                setup.prepare()
            self.assertFalse((root / "data/sentiment.json").exists())

    def test_tampered_existing_cache_is_not_silently_repaired(self):
        with tempfile.TemporaryDirectory() as folder:
            root, _, _, _ = self.fixture(folder)
            target = root / "data/sentiment.json"
            target.write_text("tampered")
            with patch.object(setup, "ROOT", root), self.assertRaisesRegex(ValueError, "Legacy.*SHA-256"):
                setup.prepare()
            self.assertEqual(target.read_text(), "tampered")

    def test_wrong_reconstruction_fails_original_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root, posts, source, _ = self.fixture(folder)
            posts.loc[0, "text"] = "changed source text"
            with patch.object(setup, "ROOT", root), patch.dict(setup.FILE_DIGESTS, {"data/expanded-sentiment.json": hashlib.sha256(source).hexdigest()}), patch.object(setup.core, "load_posts", return_value=(None, posts)), self.assertRaisesRegex(ValueError, "original manifest"):
                setup.prepare()
            self.assertFalse((root / "data/sentiment.json").exists())


if __name__ == "__main__":
    unittest.main()
