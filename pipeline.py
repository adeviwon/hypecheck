"""HypeCheck: reproducible retrospective experiment, never a trading service."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATASET = "StephanAkkerman/financial-tweets-crypto"
DATASET_REV = "c4b17a94a667af3da1c9ed5144f998e3a4f29966"
DATASET_SHA256 = "81b2f7563a44c1a103e9468a83d52319dfeae0fc616422b4a4df90d8d684b960"
MODEL = "StephanAkkerman/FinTwitBERT-sentiment"
MODEL_REV = "da059da3b3bbcb43f9ed1aeb5ae61644010c7e1e"
# Declared secondary inputs, from the independent HypeCheck data package.
# Both files are pinned by SHA-256 so a swapped or edited copy fails closed.
HOURLY_FILE = "market-hourly.csv"
HOURLY_SHA256 = "d6819bf4ff46c31c88d62130904e858deb0191125b2603910755feeba8cce003"
EXPANDED_FILE = "expanded-daily.csv"
EXPANDED_SHA256 = "2aab9bbe2c0d6782e04ec7902e3414ad564bc09eb35dccb802c75d45b2324935"
INDEPENDENT_FILE = "independent-post-scores.csv"
INDEPENDENT_SHA256 = "bb35778b88edeb9e0b11bc2ac9fba6068bf472614a48445eff447c79965e17a2"
INDEPENDENT_MODEL_REV = "c079da7618e551d182ffdf68ef7aa25cb9a79667"
COINS = ("DOGE", "SHIB", "PEPE", "BONK", "FLOKI")
START = pd.Timestamp("2024-03-01", tz="UTC")  # after this checkpoint was public
END = pd.Timestamp("2026-10-22", tz="UTC")  # exclusive, fixed historical horizon
MARKET = ["ret1", "ret3", "ret7", "vol7", "range1", "volume_change"]
ATTENTION = ["post_count", "has_posts"]
SOCIAL = ATTENTION + ["sentiment", "disagreement", "bullish", "bearish"]
# Every extended value is computed only from bars that closed BEFORE cutoff t.
BTC = ["btc_ret1", "btc_ret7"]
INTRADAY = ["hourly_vol1", "hourly_range1", "last_hour_ret", "taker_ratio1"]
MARKET_EXT = MARKET + BTC + INTRADAY


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def digest(path: Path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def request(url, **kwargs):
    """Bounded public GETs only; never accept an input URL from the web UI."""
    for attempt in range(3):
        r = requests.get(url, timeout=(10, 60), **kwargs)
        if r.status_code == 429 or r.status_code >= 500:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
        r.raise_for_status()
        return r
    raise RuntimeError("Public data request failed")


def load_posts():
    path = ROOT / "data" / "posts.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{DATASET_REV}/data/train.parquet"
        r = request(url)
        if len(r.content) > 50_000_000 or not r.content.startswith(b"PAR1"):
            raise ValueError("Unexpected dataset payload; refusing to parse")
        temp = path.with_suffix(".part")
        temp.write_bytes(r.content)
        temp.replace(path)
    if digest(path) != DATASET_SHA256:
        raise ValueError("Dataset SHA-256 does not match pinned Hugging Face LFS object")
    raw = pd.read_parquet(path)
    required = {"tweet_id", "created_at", "text", "tickers", "tweet_type", "first_posted_at"}
    if not required <= set(raw):
        raise ValueError(f"Missing columns: {required - set(raw)}")
    if raw.tweet_id.isna().any() or raw.tweet_id.duplicated().any():
        raise ValueError("Source has missing or duplicate tweet IDs")
    raw["created_at"] = pd.to_datetime(raw.created_at, utc=True, errors="coerce")
    raw["first_posted_at"] = pd.to_datetime(raw.first_posted_at, utc=True, errors="coerce")
    raw["available_at"] = raw[["created_at", "first_posted_at"]].max(axis=1)
    # A source collection time is conservative availability, not proof of live delivery.
    valid = raw.created_at.notna() & raw.available_at.notna() & raw.text.notna()
    original = raw[valid & raw.tweet_type.eq("tweet")].copy()
    original["text"] = original.text.astype(str).str.strip()
    original = original[original.text.str.len().between(8, 4000)]
    original = original.sort_values("available_at").drop_duplicates("text", keep="first")
    original = original[(original.available_at >= START - pd.Timedelta(days=1)) & (original.available_at < END)]
    original["tickers"] = original.tickers.apply(lambda x: list(x) if isinstance(x, (list, np.ndarray)) else [])
    original = original[original.tickers.apply(lambda xs: bool(set(xs) & set(COINS)))].copy()
    original["date"] = original.available_at.dt.floor("D") + pd.Timedelta(days=1)
    return raw, original


def load_prices(coin):
    if coin not in COINS:
        raise ValueError("Unknown coin")
    path = ROOT / "data" / f"{coin}-prices.csv"
    if path.exists():
        prices = pd.read_csv(path)
        prices["date"] = pd.to_datetime(prices.date, utc=True)
    else:
        rows = []
        cursor = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)
        stop = int(END.timestamp() * 1000)
        for _ in range(10):
            batch = request("https://api.binance.com/api/v3/klines", params={
                "symbol": coin + "USDT", "interval": "1d", "startTime": cursor,
                "endTime": stop - 1, "timeZone": "0", "limit": 1000,
            }).json()
            if not isinstance(batch, list):
                raise ValueError("Invalid price response")
            if not batch:
                break
            rows.extend(batch)
            next_cursor = int(batch[-1][0]) + 86_400_000
            if next_cursor <= cursor:
                raise ValueError("Price pagination did not advance")
            cursor = next_cursor
            if cursor >= stop:
                break
            time.sleep(0.15)
        else:
            raise RuntimeError("Price pagination exceeded bounded range")
        if not rows:
            raise ValueError(f"No prices for {coin}")
        prices = pd.DataFrame({
            "date": pd.to_datetime([r[0] for r in rows], unit="ms", utc=True),
            "open": [float(r[1]) for r in rows], "high": [float(r[2]) for r in rows],
            "low": [float(r[3]) for r in rows], "close": [float(r[4]) for r in rows],
            "volume": [float(r[7]) for r in rows],
        })
        temp = path.with_suffix(".part")
        prices.to_csv(temp, index=False)
        temp.replace(path)
    prices = prices.sort_values("date").reset_index(drop=True)
    if prices.date.duplicated().any() or prices.empty:
        raise ValueError("Invalid price date index")
    if not prices.date.eq(prices.date.dt.floor("D")).all() or prices.date.max() >= END:
        raise ValueError("Price bars must open at UTC midnight within the fixed horizon")
    if not (prices.date.diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError(f"Missing daily prices for {coin}; no silent forward filling")
    vals = prices[["open", "high", "low", "close", "volume"]].to_numpy(float)
    if not np.isfinite(vals).all() or (vals[:, :4] <= 0).any() or (vals[:, 4] < 0).any():
        raise ValueError("Non-finite or invalid market data")
    if (prices.high < prices[["open", "close", "low"]].max(axis=1)).any() or (prices.low > prices[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Invalid OHLC bounds")
    return prices


def validate_sentiment_scores(values):
    """Treat the local JSON cache as untrusted numeric data, not model authority."""
    if not isinstance(values, dict) or set(values) != {"bullish", "bearish", "neutral"}:
        raise ValueError("Invalid cached sentiment labels; remove data/sentiment.json and rerun")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values.values()):
        raise ValueError("Invalid cached sentiment type")
    probs = np.array(list(values.values()), dtype=float)
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any() or not np.isclose(probs.sum(), 1.0, atol=1e-5):
        raise ValueError("Invalid cached sentiment probabilities; remove data/sentiment.json and rerun")


def score_posts(posts, batch_size=16):
    """Frozen safe-weights inference; checkpoints are data, never Python pickle."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    path = ROOT / "data" / "sentiment.json"
    signature = hashlib.sha256((MODEL_REV + "|max_length=128|" + "|".join(posts.tweet_id.astype(str) + posts.text)).encode()).hexdigest()
    cache = json.loads(path.read_text()) if path.exists() else {"signature": signature, "scores": {}}
    if cache.get("signature") != signature:
        cache = {"signature": signature, "scores": {}}
    remaining = posts[~posts.tweet_id.astype(str).isin(cache["scores"])]
    if not remaining.empty:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        torch.set_num_threads(4)
        torch.manual_seed(42)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV, trust_remote_code=False, token=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL, revision=MODEL_REV, trust_remote_code=False, use_safetensors=True, token=False,
        ).eval()
        labels = {str(v).upper(): int(k) for k, v in model.config.id2label.items()}
        if set(labels) != {"NEUTRAL", "BULLISH", "BEARISH"}:
            raise ValueError(f"Unexpected sentiment label mapping: {labels}")
        for start in range(0, len(remaining), batch_size):
            batch = remaining.iloc[start:start + batch_size]
            inputs = tokenizer(batch.text.tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt")
            with torch.inference_mode():
                probs = model(**inputs).logits.softmax(-1).cpu().numpy()
            if not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=1), 1, atol=1e-5):
                raise ValueError("Invalid sentiment probabilities")
            for tid, values in zip(batch.tweet_id.astype(str), probs):
                cache["scores"][tid] = {key.lower(): float(values[idx]) for key, idx in labels.items()}
            atomic_json(path, cache)  # resumable after every batch
            print(f"Sentiment: {min(start + batch_size, len(remaining))}/{len(remaining)} uncached posts", flush=True)
    records = []
    for row in posts.to_dict("records"):
        values = cache["scores"][str(row["tweet_id"])]
        validate_sentiment_scores(values)
        row.update(values)
        row["sentiment"] = values["bullish"] - values["bearish"]
        records.append(row)
    if not records:
        raise ValueError("No eligible posts; cannot claim a sentiment experiment")
    return pd.DataFrame(records)


def _read_pinned(name, sha256):
    """Local pinned copies only. A missing or edited file fails closed."""
    path = ROOT / "data" / name
    if not path.exists():
        return None
    if digest(path) != sha256:
        raise ValueError(f"{name} SHA-256 does not match the pinned copy")
    return path


def load_hourly(coin):
    """Hourly Binance spot bars, aggregated to the completed day before each cutoff."""
    if coin not in COINS:
        raise ValueError("Unknown coin")
    path = _read_pinned(HOURLY_FILE, HOURLY_SHA256)
    if path is None:
        raise ValueError(f"data/{HOURLY_FILE} is required; see README for the pinned source")
    columns = ["coin", "open_time", "open_usdt", "high_usdt", "low_usdt", "close_usdt",
               "volume_usdt", "taker_buy_volume_usdt"]
    bars = pd.read_csv(path, usecols=columns)
    bars = bars[bars.coin.eq(coin)].copy()
    bars["open_time"] = pd.to_datetime(bars.open_time, utc=True, errors="coerce")
    bars = bars.sort_values("open_time").reset_index(drop=True)
    if bars.empty or bars.open_time.isna().any() or bars.open_time.duplicated().any():
        raise ValueError(f"Invalid hourly index for {coin}")
    if not bars.open_time.dt.minute.eq(0).all():
        raise ValueError("Hourly bars must open on the hour")
    if not (bars.open_time.diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError(f"Hourly bars are not contiguous for {coin}; no silent interpolation")
    values = bars[["open_usdt", "high_usdt", "low_usdt", "close_usdt", "volume_usdt",
                   "taker_buy_volume_usdt"]].to_numpy(float)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4:] < 0).any():
        raise ValueError("Non-finite or invalid hourly market data")
    if (bars.high_usdt < bars[["open_usdt", "close_usdt", "low_usdt"]].max(axis=1)).any():
        raise ValueError("Invalid hourly OHLC bounds")
    if (bars.low_usdt > bars[["open_usdt", "close_usdt", "high_usdt"]].min(axis=1)).any():
        raise ValueError("Invalid hourly OHLC bounds")
    bars["day"] = bars.open_time.dt.floor("D")
    bars["hour_ret"] = np.log(bars.close_usdt / bars.open_usdt)
    grouped = bars.groupby("day")
    daily = pd.DataFrame({
        "hourly_vol1": grouped.hour_ret.std(ddof=1),
        "hourly_range1": (grouped.high_usdt.max() - grouped.low_usdt.min()) / grouped.close_usdt.last(),
        "last_hour_ret": grouped.hour_ret.last(),
        "taker_ratio1": grouped.taker_buy_volume_usdt.sum() / grouped.volume_usdt.sum(),
        "hours": grouped.size(),
    })
    daily = daily[daily.hours.eq(24)].drop(columns="hours")
    if daily.empty:
        raise ValueError(f"No complete UTC days of hourly bars for {coin}")
    # A day's 23:00 bar closes at its own midnight, so it informs the NEXT cutoff.
    daily.index = daily.index + pd.Timedelta(days=1)
    return daily.reset_index(names="date")


def load_btc():
    """BTC/USDT daily context from the pinned independent market file."""
    path = _read_pinned(EXPANDED_FILE, EXPANDED_SHA256)
    if path is None:
        raise ValueError(f"data/{EXPANDED_FILE} is required; see README for the pinned source")
    frame = pd.read_csv(path, usecols=["utc_day", "coin", "open_usdt", "close_usdt", "btc_close_usdt"])
    btc = frame[["utc_day", "btc_close_usdt"]].dropna().drop_duplicates("utc_day").copy()
    btc["date"] = pd.to_datetime(btc.utc_day, utc=True, errors="coerce")
    btc = btc.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if btc.empty or btc.date.duplicated().any():
        raise ValueError("Invalid BTC date index")
    if not (btc.date.diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError("BTC daily series has gaps; no silent forward filling")
    closes = btc.btc_close_usdt.to_numpy(float)
    if not np.isfinite(closes).all() or (closes <= 0).any():
        raise ValueError("Invalid BTC closes")
    btc["btc_ret1"] = btc.btc_close_usdt.pct_change().shift(1)
    btc["btc_ret7"] = btc.btc_close_usdt.pct_change(7).shift(1)
    # Same-provider cross-check: the independent daily file must reproduce our own bars exactly.
    for coin in COINS:
        local = ROOT / "data" / f"{coin}-prices.csv"
        if not local.exists():
            continue
        mine = pd.read_csv(local)
        mine["date"] = pd.to_datetime(mine.date, utc=True)
        theirs = frame[frame.coin.eq(coin)].copy()
        theirs["date"] = pd.to_datetime(theirs.utc_day, utc=True, errors="coerce")
        merged = mine.merge(theirs[["date", "open_usdt", "close_usdt"]], on="date", how="inner")
        if merged.empty:
            raise ValueError(f"No overlapping daily bars to cross-check for {coin}")
        if not np.array_equal(merged.open.to_numpy(float), merged.open_usdt.to_numpy(float)):
            raise ValueError(f"Independent daily opens disagree with our Binance bars for {coin}")
        if not np.array_equal(merged.close.to_numpy(float), merged.close_usdt.to_numpy(float)):
            raise ValueError(f"Independent daily closes disagree with our Binance bars for {coin}")
    return btc[["date", "btc_ret1", "btc_ret7"]]


def independent_check(scored):
    """Agreement with a separately produced FinTwitBERT pass on a different pinned revision."""
    path = _read_pinned(INDEPENDENT_FILE, INDEPENDENT_SHA256)
    if path is None:
        return {"available": False, "reason": f"data/{INDEPENDENT_FILE} is absent"}
    other = pd.read_csv(path, usecols=["post_id", "coin", "p_bearish", "p_neutral", "p_bullish"])
    other["post_id"] = other.post_id.astype(str)
    other = other.drop_duplicates("post_id")
    mine = scored[["tweet_id", "text", "bullish", "neutral", "bearish"]].copy()
    mine["post_id"] = mine.tweet_id.astype(str)
    merged = mine.merge(other, on="post_id", how="inner")
    if merged.empty:
        return {"available": False, "reason": "no shared post IDs"}
    classes = ["bearish", "neutral", "bullish"]
    ours = merged[classes].to_numpy(float).argmax(axis=1)
    theirs = merged[["p_" + name for name in classes]].to_numpy(float).argmax(axis=1)
    ours_net = merged.bullish.to_numpy(float) - merged.bearish.to_numpy(float)
    theirs_net = merged.p_bullish.to_numpy(float) - merged.p_bearish.to_numpy(float)
    delta = np.abs(merged.bullish.to_numpy(float) - merged.p_bullish.to_numpy(float))
    length = merged.text.astype(str).str.len().to_numpy(float)
    short, long = length <= 400, length > 400
    return {"available": True, "our_model_revision": MODEL_REV,
            "independent_model_revision": INDEPENDENT_MODEL_REV,
            "shared_posts": int(len(merged)),
            "class_agreement": float((ours == theirs).mean()),
            "net_sentiment_sign_agreement": float((np.sign(ours_net) == np.sign(theirs_net)).mean()),
            "mean_abs_p_bullish_difference": float(delta.mean()),
            "max_abs_p_bullish_difference": float(delta.max()),
            "share_p_bullish_differing_over_0_25": float((delta > 0.25).mean()),
            "length_diagnostic": {
                "mean_abs_difference_short_text": float(delta[short].mean()) if short.any() else None,
                "mean_abs_difference_long_text": float(delta[long].mean()) if long.any() else None,
                "note": "Disagreement is flat across text-length buckets, so truncation does not explain it.",
            },
            "note": "Independent scoring pass from a different pinned checkpoint revision. This is a reproducibility check, not a ground-truth accuracy measure."}


def market_features(prices):
    out = prices.copy()
    returns = out.close.pct_change()
    out["ret1"] = returns.shift(1)
    out["ret3"] = out.close.pct_change(3).shift(1)
    out["ret7"] = out.close.pct_change(7).shift(1)
    out["vol7"] = returns.rolling(7).std().shift(1)
    out["range1"] = ((out.high - out.low) / out.close).shift(1)
    out["volume_change"] = np.log1p(out.volume).diff().shift(1)
    out["actual_return"] = out.close / out.open - 1
    out["actual_up"] = (out.actual_return > 0).astype(int)
    return out


def build_dataset(posts, prices, hourly, btc):
    frames = []
    examples = {}
    for coin in COINS:
        subset = posts[posts.tickers.apply(lambda xs: coin in xs)]
        group = subset.groupby("date")
        daily = group.agg(post_count=("tweet_id", "size"), sentiment=("sentiment", "mean"),
                          disagreement=("sentiment", "std"), bullish=("bullish", "mean"), bearish=("bearish", "mean"))
        daily["disagreement"] = daily.disagreement.fillna(0)
        frame = (market_features(prices[coin]).set_index("date").join(daily)
                 .join(hourly[coin].set_index("date")).join(btc.set_index("date")).reset_index())
        frame["post_count"] = frame.post_count.fillna(0).astype(int)
        frame["has_posts"] = (frame.post_count > 0).astype(int)
        # Zero is an imputation only; has_posts distinguishes unobserved text from neutral.
        frame[SOCIAL] = frame[SOCIAL].fillna(0)
        frame["coin"] = coin
        frame = frame[(frame.date >= START) & (frame.date < END)].dropna(subset=MARKET)
        # The primary row set stays exactly as published; extended inputs must be complete.
        if not np.isfinite(frame[MARKET_EXT].to_numpy(float)).all():
            raise ValueError(f"Extended market features are incomplete for {coin}; refusing to impute")
        for date, samples in group:
            examples[(coin, date.strftime("%Y-%m-%d"))] = [{
                "text": r["text"][:700], "url": r["url"] if isinstance(r.get("url"), str) and r["url"].startswith(("https://twitter.com/", "https://x.com/")) else "",
                "created_at": r["created_at"].isoformat(), "available_at": r["available_at"].isoformat(),
            } for r in samples.sort_values("available_at").tail(3).to_dict("records")]
        frames.append(frame)
    dataset = pd.concat(frames).sort_values(["date", "coin"]).reset_index(drop=True)
    if dataset.duplicated(["date", "coin"]).any() or not np.isfinite(dataset[MARKET_EXT + SOCIAL].to_numpy()).all():
        raise ValueError("Invalid feature table")
    return dataset, examples


def split_dates(frame):
    dates = sorted(frame.date.unique())
    if len(dates) < 150:
        raise ValueError("Too few unique dates for the planned evaluation")
    val_start = pd.Timestamp(dates[int(len(dates) * 0.6)])
    test_start = pd.Timestamp(dates[int(len(dates) * 0.8)])
    # Exclude the adjacent day so a full target window ends BEFORE next partition.
    train = frame[frame.date + pd.Timedelta(days=1) < val_start].copy()
    val = frame[(frame.date >= val_start) & (frame.date + pd.Timedelta(days=1) < test_start)].copy()
    test = frame[frame.date >= test_start].copy()
    for part in (train, val, test):
        if part.actual_up.nunique() != 2 or set(part.coin) != set(COINS):
            raise ValueError("Each partition must cover five coins and both outcomes")
    return train, val, test


def classifier(features, c):
    prep = ColumnTransformer([
        ("numeric", StandardScaler(), features),
        ("coin", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["coin"]),
    ])
    return Pipeline([("prep", prep), ("model", LogisticRegression(C=c, max_iter=1000, random_state=42))])


def scores(y, prob):
    prob = np.asarray(prob)
    if len(y) == 0:
        return {"accuracy": None, "balanced_accuracy": None, "brier": None, "n": 0}
    if not np.isfinite(prob).all() or (prob < 0).any() or (prob > 1).any():
        raise ValueError("Invalid forecast probabilities")
    return {"accuracy": float(accuracy_score(y, prob >= 0.5)),
            "balanced_accuracy": float(balanced_accuracy_score(y, prob >= 0.5)),
            "brier": float(brier_score_loss(y, prob)), "n": int(len(y))}


def metric(name, frame, prob):
    out = {"name": name, **scores(frame.actual_up, prob), "n_dates": int(frame.date.nunique()), "per_coin": {}}
    for coin in COINS:
        mask = frame.coin.eq(coin).to_numpy()
        out["per_coin"][coin] = scores(frame.loc[mask, "actual_up"], np.asarray(prob)[mask])
    return out


def bootstrap_delta(test, market, social):
    # ponytail: 7-day moving blocks, exploratory interval; not a formal edge test.
    tmp = pd.DataFrame({"date": test.date, "delta": (social - test.actual_up.to_numpy()) ** 2 - (market - test.actual_up.to_numpy()) ** 2})
    series = tmp.groupby("date").delta.mean().to_numpy()
    rng = np.random.default_rng(42)
    draws = []
    for _ in range(1000):
        starts = rng.integers(0, len(series), size=int(np.ceil(len(series) / 7)))
        sample = np.concatenate([series[(start + np.arange(7)) % len(series)] for start in starts])[:len(series)]
        draws.append(float(sample.mean()))
    return {"mean_social_minus_market_brier": float(series.mean()), "interval95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
            "method": "1000 circular moving-block resamples of UTC dates, 7-day blocks; coins grouped by day; exploratory interval"}


def fit_and_report(raw, posts, prices, hourly, btc):
    frame, examples = build_dataset(posts, prices, hourly, btc)
    train, val, test = split_dates(frame)
    configurations = {"Market only": MARKET, "Market + attention": MARKET + ATTENTION, "Market + X": MARKET + SOCIAL}
    models, validation = {}, []
    # Freeze feature definitions, candidates and threshold before viewing final test.
    for name, features in configurations.items():
        candidates = []
        for c in (0.01, 0.1, 1.0):
            fitted = classifier(features, c).fit(train, train.actual_up)
            brier = brier_score_loss(val.actual_up, fitted.predict_proba(val)[:, 1])
            candidates.append((float(brier), c, fitted))
        best_brier, best_c, selected = min(candidates, key=lambda x: (x[0], x[1]))
        models[name] = selected  # no refit on validation; final models see train only
        validation.append({"name": name, "selected_C": best_c, "brier": best_brier, "candidates": [{"C": c, "brier": score} for score, c, _ in candidates]})
    probabilities = {name: fitted.predict_proba(test)[:, 1] for name, fitted in models.items()}
    # Declared secondary comparison: identical rows, identical protocol, richer market inputs.
    extended_configurations = {"Market + BTC + intraday": MARKET_EXT,
                               "Market + BTC + intraday + X": MARKET_EXT + SOCIAL}
    ext_models, ext_validation = {}, []
    for name, features in extended_configurations.items():
        candidates = []
        for c in (0.01, 0.1, 1.0):
            fitted = classifier(features, c).fit(train, train.actual_up)
            brier = brier_score_loss(val.actual_up, fitted.predict_proba(val)[:, 1])
            candidates.append((float(brier), c, fitted))
        best_brier, best_c, selected = min(candidates, key=lambda x: (x[0], x[1]))
        ext_models[name] = selected
        ext_validation.append({"name": name, "selected_C": best_c, "brier": best_brier,
                               "candidates": [{"C": c, "brier": score} for score, c, _ in candidates]})
    ext_probabilities = {name: fitted.predict_proba(test)[:, 1] for name, fitted in ext_models.items()}
    EXT_BASE = "Market + BTC + intraday"
    EXT_X = "Market + BTC + intraday + X"
    majority = int(train.actual_up.mean() >= 0.5)
    metric_rows = [metric("Always up", test, np.ones(len(test))),
                   metric("Training majority", test, np.full(len(test), majority, dtype=float)),
                   metric("Previous direction", test, (test.ret1.to_numpy() > 0).astype(float)),
                   metric("Training prevalence", test, np.full(len(test), float(train.actual_up.mean())))]
    metric_rows += [metric(name, test, p) for name, p in probabilities.items()]
    predictions = []
    for idx, row in enumerate(test.to_dict("records")):
        date = row["date"].strftime("%Y-%m-%d")
        prediction = {key: row[key] for key in ["coin", "actual_return", "actual_up", "post_count"]}
        prediction.update(date=date, cutoff=row["date"].isoformat(), target_end=(row["date"] + pd.Timedelta(days=1)).isoformat(),
                          market_prob=float(probabilities["Market only"][idx]), social_prob=float(probabilities["Market + X"][idx]),
                          attention_prob=float(probabilities["Market + attention"][idx]),
                          extended_market_prob=float(ext_probabilities[EXT_BASE][idx]),
                          extended_social_prob=float(ext_probabilities[EXT_X][idx]), persistence=int(row["ret1"] > 0),
                          examples=examples.get((row["coin"], date), []))
        prediction.update({key: float(row[key]) if row["post_count"] else None for key in ["sentiment", "bullish", "bearish"]})
        predictions.append(prediction)
    coverage = []
    for coin in COINS:
        source_mask = raw.tickers.apply(lambda x: coin in x if isinstance(x, (list, np.ndarray)) else False)
        selected = posts[posts.tickers.apply(lambda xs: coin in xs)]
        coverage.append({"coin": coin, "total_posts": int(source_mask.sum()), "original_posts": int((source_mask & raw.tweet_type.eq("tweet")).sum()),
                         "eligible_posts": int(len(selected)), "days_with_posts": int(selected.date.nunique()),
                         "price_days": int(len(prices[coin])), "first": str(prices[coin].date.min().date()), "last": str(prices[coin].date.max().date())})
    learning_curve = []
    for fraction in (0.25, 0.5, 1.0):
        dates = sorted(train.date.unique())
        small = train[train.date.isin(dates[-max(30, int(len(dates) * fraction)):])]
        c = next(x["selected_C"] for x in validation if x["name"] == "Market + X")
        fitted = classifier(MARKET + SOCIAL, c).fit(small, small.actual_up)
        learning_curve.append({"history_fraction": fraction, "train_dates": int(small.date.nunique()), "validation_brier": float(brier_score_loss(val.actual_up, fitted.predict_proba(val)[:, 1]))})
    mask = test.post_count.to_numpy() > 0
    subgroup = {"with_observed_posts": {"n": int(mask.sum()), "metrics": [metric(name, test[mask], p[mask]) for name, p in probabilities.items()] if mask.any() else []},
                "without_observed_posts": {"n": int((~mask).sum())}}
    extended = {
        "note": "Secondary comparison on the same held-out rows: does richer market context change the answer? Not used for selection.",
        "features": {"BTC": BTC, "intraday": INTRADAY},
        "validation": ext_validation,
        "metrics": [metric("Market only (primary)", test, probabilities["Market only"]),
                    metric("Market + X (primary)", test, probabilities["Market + X"])]
                   + [metric(name, test, p) for name, p in ext_probabilities.items()],
        "uncertainty": bootstrap_delta(test, ext_probabilities[EXT_BASE], ext_probabilities[EXT_X]),
        "primary_uncertainty": bootstrap_delta(test, probabilities["Market only"], probabilities["Market + X"]),
    }
    report = {
        "meta": {"project": "HypeCheck", "generated_at": datetime.now(timezone.utc).isoformat(),
                 "dataset": DATASET, "dataset_revision": DATASET_REV, "sentiment_model": MODEL, "sentiment_revision": MODEL_REV,
                 "disclaimer": "This project is a prototype and is not financial advice.",
                 "splits": {name: [part.date.min().strftime("%Y-%m-%d"), part.date.max().strftime("%Y-%m-%d")] for name, part in [("train", train), ("validation", val), ("test", test)]},
                 "evaluation_scope": "Retrospective frozen-model historical replay. No live forecasts, orders, or promised returns.",
                 "retrospective_pretrained_cutoff": START.strftime("%Y-%m-%d"),
                 "eligible_posts_missing_collection_timestamp": int(posts.first_posted_at.isna().sum()),
                 "availability_policy": "Original posts only; max(created_at, first_posted_at) where available. Prior-day posts only.",
                 "target": "Positive Binance USDT open-to-close return during the next UTC day; flat counts as not up.",
                 "hourly_market_source": "Binance Vision 1h spot klines from the pinned independent data package, aggregated only over UTC hours that closed before the cutoff.",
                 "btc_context_source": "BTC/USDT daily closes from the pinned independent market file, cross-checked bar-for-bar against our own Binance daily bars.",
                 "independent_scoring_revision": INDEPENDENT_MODEL_REV,
                 "limitations": [
                     "Selected influencers are not representative of all X. Missing observations do not mean no discussion occurred.",
                     "Sparse and changing coverage; author/source changes can mimic sentiment effects. No causal claim.",
                     "Pretrained sentiment checkpoint was public before the experiment period, but training-corpus overlap is not fully auditable.",
                     "English sentiment model; source language is not independently verified. Multi-coin posts share one sentiment score.",
                     "Sentiment is truncated to 128 tokens. Probabilities are not independently calibrated; sentiment certainty is not return certainty.",
                     "Historical USDT exchange bars exclude fees, slippage, latency and execution constraints; no profitability claim.",
                     "Only one held-out period. Date-block intervals are exploratory, and the final period must not be reused to tune models.",
                     "Two pinned revisions of the same sentiment checkpoint disagree on the direction of about 13% of shared posts, so these scores are model-revision dependent rather than a stable ground truth.",
                 ]},
        "metrics": metric_rows, "coverage": coverage, "predictions": predictions,
        "validation": validation, "learning_curve": learning_curve, "subgroups": subgroup,
        "label_counts": {name: {coin: {"n": int(len(part[part.coin.eq(coin)])), "up": int(part.loc[part.coin.eq(coin), "actual_up"].sum())} for coin in COINS} for name, part in [("train", train), ("validation", val), ("test", test)]},
        "uncertainty": bootstrap_delta(test, probabilities["Market only"], probabilities["Market + X"]),
        "extended": extended,
        "independent_check": independent_check(posts),
    }
    atomic_json(ROOT / "docs" / "report.json", report)
    frame.to_csv(ROOT / "data" / "features.csv", index=False)
    exports = {}
    for name, fitted in models.items():
        names = fitted.named_steps["prep"].get_feature_names_out().tolist()
        exports[name] = {"feature_names": names, "coefficients": fitted.named_steps["model"].coef_[0].tolist(),
                         "intercept": fitted.named_steps["model"].intercept_.tolist(),
                         "numeric_mean": fitted.named_steps["prep"].named_transformers_["numeric"].mean_.tolist(),
                         "numeric_scale": fitted.named_steps["prep"].named_transformers_["numeric"].scale_.tolist()}
    atomic_json(ROOT / "artifacts" / "coefficients.json", exports)
    files = [p for p in sorted((ROOT / "data").glob("*")) if p.is_file() and p.suffix in {".csv", ".parquet", ".json"}]
    files += [ROOT / "docs/report.json", ROOT / "artifacts/coefficients.json"]
    manifest = {"dataset_revision": DATASET_REV, "model_revision": MODEL_REV, "seed": 42, "code_sha256": digest(Path(__file__)),
                "files": {str(p.relative_to(ROOT)): digest(p) for p in files}}
    atomic_json(ROOT / "artifacts" / "manifest.json", manifest)
    for item in metric_rows:
        print(f"{item['name']}: accuracy={item['accuracy']:.4f}, balanced={item['balanced_accuracy']:.4f}, Brier={item['brier']:.4f}, n={item['n']}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["data", "sentiment", "train", "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 64:
        parser.error("batch-size must be between 1 and 64")
    raw, selected = load_posts()
    print(f"Source posts: {len(raw)}; eligible original deduplicated posts: {len(selected)}", flush=True)
    prices = {coin: load_prices(coin) for coin in COINS}
    hourly = {coin: load_hourly(coin) for coin in COINS}
    btc = load_btc()
    print("Price coverage:", {coin: len(p) for coin, p in prices.items()}, flush=True)
    print("Hourly coverage:", {coin: len(h) for coin, h in hourly.items()}, "BTC context rows:", len(btc), flush=True)
    if args.stage == "data":
        return
    scored = score_posts(selected, args.batch_size)
    if args.stage == "sentiment":
        return
    fit_and_report(raw, scored, prices, hourly, btc)


if __name__ == "__main__":
    main()
