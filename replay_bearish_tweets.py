"""Seeded historical bearish-post replay with frozen HypeCheck coefficients.
Not a new independent test or a trading recommendation. Run after notebook inputs exist.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pipeline as p

ROOT = Path(__file__).resolve().parent


def predict(export, row):
    total = float(export["intercept"][0])
    numeric = 0
    for name, weight in zip(export["feature_names"], export["coefficients"]):
        if name.startswith("numeric__"):
            value = (float(row[name.removeprefix("numeric__")]) - export["numeric_mean"][numeric]) / export["numeric_scale"][numeric]
            numeric += 1
        elif name.startswith("coin__coin_"):
            value = float(row["coin"] == name.removeprefix("coin__coin_"))
        else:
            raise ValueError("Unexpected exported feature: " + name)
        total += float(weight) * value
    return 1 / (1 + math.exp(-total))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.count <= 100:
        parser.error("count must be 1..100")
    _, posts = p.load_posts()
    cache = json.loads((ROOT / "data/sentiment.json").read_text())["scores"]
    report = json.loads((ROOT / "docs/report.json").read_text())
    exports = json.loads((ROOT / "artifacts/coefficients.json").read_text())
    frozen = {(r["coin"], r["date"]): r for r in report["predictions"]}
    frame = pd.read_csv(ROOT / "data/features.csv")
    frame["date"] = pd.to_datetime(frame.date, utc=True).dt.strftime("%Y-%m-%d")
    features = {(r["coin"], r["date"]): r for r in frame.to_dict("records")}
    groups = {}
    # Outcome values are never used in eligibility or seeded sampling.
    for row in posts.to_dict("records"):
        tid = str(row["tweet_id"])
        score = cache.get(tid)
        if score is None:
            continue
        p.validate_sentiment_scores(score)
        if score["bearish"] < 0.5:
            continue
        for coin in p.COINS:
            key = (coin, row["date"].strftime("%Y-%m-%d"))
            if key not in frozen or coin not in row["tickers"]:
                continue
            # Explicit ticker in text avoids merely retrospectively assigned tags.
            if not re.search(r"(?<![A-Za-z0-9])[$#]?" + coin + r"(?![A-Za-z0-9])", row["text"], re.I):
                continue
            groups.setdefault(key, []).append(row)
    rng = random.Random(args.seed)
    keys = sorted(groups)
    if not keys:
        raise ValueError("No eligible held-out bearish posts")
    selected = rng.sample(keys, min(args.count, len(keys)))
    chosen = [(key, rng.choice(sorted(groups[key], key=lambda r: str(r["tweet_id"])))) for key in selected]
    # Fresh neural inference on chosen post texts, not just a cached-score lookup.
    torch.set_num_threads(4)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(p.MODEL, revision=p.MODEL_REV, trust_remote_code=False, token=False)
    model = AutoModelForSequenceClassification.from_pretrained(p.MODEL, revision=p.MODEL_REV, trust_remote_code=False, use_safetensors=True, token=False).eval()
    labels = {str(v).lower(): int(k) for k, v in model.config.id2label.items()}
    assert set(labels) == {"bullish", "bearish", "neutral"}
    inputs = tokenizer([row["text"] for _, row in chosen], padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.inference_mode():
        values = model(**inputs).logits.softmax(-1).cpu().numpy()
    records = []
    for (key, tweet), probs in zip(chosen, values):
        coin, date = key
        row = features[key]
        actual = frozen[key]
        daily_prob = predict(exports["Market + X"], row)
        market_prob = predict(exports["Market only"], row)
        assert abs(daily_prob - actual["social_prob"]) < 1e-12
        assert abs(market_prob - actual["market_prob"]) < 1e-12
        sent = {name: float(probs[idx]) for name, idx in labels.items()}
        p.validate_sentiment_scores(sent)
        single = dict(row, post_count=1, has_posts=1, disagreement=0,
                      sentiment=sent["bullish"]-sent["bearish"], bullish=sent["bullish"], bearish=sent["bearish"])
        single_prob = predict(exports["Market + X"], single)
        assert tweet["available_at"] < pd.Timestamp(actual["cutoff"])
        record = dict(coin=coin, tweet_id=str(tweet["tweet_id"]), text=tweet["text"],
                      source_url=tweet["url"] if isinstance(tweet.get("url"), str) else "",
                      created_at=tweet["created_at"].isoformat(), available_at=tweet["available_at"].isoformat(),
                      cutoff=actual["cutoff"], target_end=actual["target_end"],
                      sentiment_probabilities=sent, fresh_inference_bearish=sent["bearish"]>=0.5,
                      cached_bearish_probability=cache[str(tweet["tweet_id"])]["bearish"],
                      single_post_up_probability=single_prob,
                      hypothetical_signal="BUY/up" if single_prob>=0.5 else "SELL/down",
                      daily_full_context_up_probability=daily_prob,
                      market_only_up_probability=market_prob,
                      actual_return=actual["actual_return"], actual_direction="up" if actual["actual_up"] else "down_or_flat",
                      daily_post_count=int(row["post_count"]),
                      human_intent_review="unreviewed",
                      execution_status="completed",
                      direction_result="correct" if (single_prob>=0.5)==bool(actual["actual_up"]) else "incorrect",
                      daily_context_correct=(daily_prob>=0.5)==bool(actual["actual_up"]),
                      market_only_correct=(market_prob>=0.5)==bool(actual["actual_up"]))
        records.append(record)
    correct=sum(r["direction_result"]=="correct" for r in records)
    summary=dict(seed=args.seed, requested=args.count, sampled=len(records), eligible_coin_days=len(keys),
                 independent_sample_size_warning="Distinct coin-days, but coins can share a market date; not independent trials.",
                 single_post_correct=correct, single_post_incorrect=len(records)-correct,
                 full_daily_correct=sum(r["daily_context_correct"] for r in records),
                 market_only_correct=sum(r["market_only_correct"] for r in records),
                 always_down_correct=sum(r["actual_direction"]=="down_or_flat" for r in records),
                 new_independent_holdout=False,
                 unique_target_dates=len(set(r["cutoff"] for r in records)),
                 unique_tweet_ids=len(set(r["tweet_id"] for r in records)),
                 all_calls_down=all(r["hypothetical_signal"]=="SELL/down" for r in records),
                 model_revision=p.MODEL_REV,
                 input_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ["pipeline.py", "replay_bearish_tweets.py", "data/posts.parquet", "data/features.csv", "data/sentiment.json", "artifacts/coefficients.json", "docs/report.json"]},
                 protocol="Seeded sample of existing held-out coin-days with explicit ticker and cached bearish probability >=0.5; one random post per selected coin-day. Fresh sentiment inference, frozen JSON coefficient prediction. No retraining or outcome-based selection.",
                 target="Next UTC day's open-to-close, not 24 hours after the tweet.",
                 single_post_warning="Single-post diagnostic replaces daily social aggregates with one post while keeping historical market features fixed. It is not the original full-context forecast or a model trained specifically for single tweets.",
                 disclaimer="Historical hypothetical direction signals only. No fees, execution, position sizing or profitability validation. This project is a prototype and is not financial advice.")
    out=ROOT/"tweet-tests"
    out.mkdir(exist_ok=True)
    (out/"bearish-replay.json").write_text(json.dumps(dict(summary=summary,records=records),ensure_ascii=False,indent=2)+"\n")
    flat=[]
    for r in records:
        item={k:v for k,v in r.items() if k!="sentiment_probabilities"}
        item.update({"text_"+k:v for k,v in r["sentiment_probabilities"].items()})
        flat.append(item)
    pd.DataFrame(flat).to_csv(out/"bearish-replay.csv",index=False)
    lines=["# Real archived bearish-tweet replay", "", summary["protocol"], "", summary["single_post_warning"], "", summary["target"], "", summary["disclaimer"], "", f"Single-post diagnostic: {correct}/{len(records)} correct; full daily model: {summary['full_daily_correct']}/{len(records)}; market-only: {summary['market_only_correct']}/{len(records)}; always down: {summary['always_down_correct']}/{len(records)}.", "", "This is an illustrative subset of already examined data, not a fresh accuracy claim. Archive URLs are recorded, not independently authenticated against live X.", "", "| Coin | Target date UTC | Text bearish | Single-post P(up) | Hypothetical signal | Actual return | Result |", "|---|---|---:|---:|---|---:|---|"]
    for r in records:
        lines.append(f"| {r['coin']} | {r['cutoff'][:10]} | {r['sentiment_probabilities']['bearish']:.1%} | {r['single_post_up_probability']:.1%} | {r['hypothetical_signal']} | {r['actual_return']:+.2%} | {r['direction_result']} |")
    for i,r in enumerate(records,1):
        lines.extend(["", f"## Example {i}: {r['coin']} / {r['cutoff'][:10]}", "", r["text"], "", "Archived source: "+r["source_url"], "", "Available: "+r["available_at"]])
    (out/"README.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(summary,indent=2))
    print("FIRST RANDOM EXAMPLE",json.dumps(records[0],ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
