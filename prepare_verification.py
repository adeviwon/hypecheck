"""Restore the archived cache from verified existing scores; never invent/reinfer scores."""
import hashlib
import json
from pathlib import Path
import pipeline as core
from colab_setup import FILE_DIGESTS

ROOT = Path(__file__).resolve().parent


def prepare():
    target = ROOT / "data/sentiment.json"
    expected = json.loads((ROOT / "artifacts/manifest.json").read_text())["files"]["data/sentiment.json"]
    if target.exists():
        if core.digest(target) != expected:
            raise ValueError("Legacy sentiment cache SHA-256 mismatch; refusing to overwrite")
        return target
    source = ROOT / "data/expanded-sentiment.json"
    if core.digest(source) != FILE_DIGESTS["data/expanded-sentiment.json"]:
        raise ValueError("Expanded sentiment cache SHA-256 mismatch")
    _, posts = core.load_posts()  # Validates the pinned raw input before selecting posts.
    scores = json.loads(source.read_text())["scores"]
    signature = hashlib.sha256((core.MODEL_REV + "|max_length=128|" +
                                "|".join(posts.tweet_id.astype(str) + posts.text)).encode()).hexdigest()
    selected = {str(tid): scores[str(tid)] for tid in posts.tweet_id}
    for values in selected.values():
        core.validate_sentiment_scores(values)
    payload = json.dumps({"signature": signature, "scores": selected},
                         ensure_ascii=False, allow_nan=False, indent=2).encode()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("Reconstructed legacy cache does not match original manifest")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".part")
    temporary.write_bytes(payload)
    temporary.replace(target)
    return target


if __name__ == "__main__":
    print("Verified original legacy sentiment cache:", prepare())
