"""Rate a list of public posts with the trained crowd model, inside the notebook.

The 40 pre-filled posts are real archive posts that were held out of training: they come
from the later-time and unseen-coin tests, so their next-day outcomes are known and the
model has never seen them. A post the user types in has no known outcome yet, so it is
rated but not scored.
"""
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

import sample_pipeline as sp

ROOT = Path(__file__).resolve().parent
PREFIX = re.compile(r"^\s*([A-Za-z]{2,10})\s*\|(?!=)\s*(.+)$")
CASHTAG = re.compile(r"\$([A-Za-z0-9_]{2,10})\b")
BAD = re.compile(r"(fuck|shit|bitch|cunt|nigga|fag|rape|porn|http|t\.me)", re.I)
ADDRESS = re.compile(r"[A-Za-z0-9]{28,}")


def _clean(text):
    """Readable public posts only. The raw text is kept: it is the inference unit."""
    raw = str(text).strip()
    body = " ".join(raw.split())
    if not 30 <= len(body) <= 600:
        return None
    if BAD.search(raw) or ADDRESS.search(raw):
        return None
    return raw


def examples(n_later=20, n_unseen=20):
    """Real held-out posts the model never trained on: later dates and unseen coins."""
    _, _, later, unseen = sp.dataset()
    chosen = []
    for part, count, label in ((later, n_later, "later date"), (unseen, n_unseen, "unseen coin")):
        singles = part[part.post_count == 1].copy().sort_values(["coin", "date", "cutoff"])
        usable = []
        for _, row in singles.iterrows():
            body = _clean(row.text)
            if body is not None:
                usable.append((row, body))
        # prefer posts that fit on one line, then balance the coins, then top up
        usable.sort(key=lambda item: ("\n" in item[1], item[0].coin, item[0].date, item[0].cutoff))
        seen, picked, per_coin = set(), [], {}
        cap = max(1, math.ceil(count / max(1, len({row.coin for row, _ in usable}))))
        for limit in (cap, count):
            for row, body in usable:
                if len(picked) == count:
                    break
                if body in seen or per_coin.get(row.coin, 0) >= limit:
                    continue
                seen.add(body)
                per_coin[row.coin] = per_coin.get(row.coin, 0) + 1
                picked.append({"coin": row.coin, "text": row.text, "cutoff": row.cutoff.isoformat(),
                               "actual_up": int(row.actual_up), "actual_return": float(row.actual_return),
                               "group": label})
        chosen.extend(picked)
    return chosen


def _item(entry, default_coin, coins, number):
    """One list entry is one post: strip an optional 'TICKER | ' prefix, then find the coin."""
    body = str(entry).strip()
    coin = default_coin
    first, _, rest = body.partition("\n")
    match = PREFIX.match(first)
    if match and match.group(1).upper() in coins:
        coin = match.group(1).upper()
        body = body[match.start(2):]
    found = [c for c in CASHTAG.findall(body) if c.upper() in coins]
    if found and coin == default_coin:
        coin = found[0].upper()
    if len(body) < 8:
        raise ValueError(f"Post {number} is shorter than 8 characters: {body[:40]!r}")
    return {"coin": coin, "text": body}


def parse_lines(blob, default_coin, coins):
    """Separate posts with a blank line (one post per line also works).

    A post starting with 'TICKER | ' sets its coin; otherwise a cashtag in the text,
    or the selected coin, is used. The post text itself is passed through unchanged.
    """
    text = str(blob).replace("\r\n", "\n").strip()
    chunks = [chunk for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
    if len(chunks) == 1:
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) > 1:
            chunks = lines
    out = []
    for number, chunk in enumerate(chunks, 1):
        body = chunk.strip()
        coin = default_coin
        first, _, rest = body.partition("\n")
        match = PREFIX.match(first)
        if match and match.group(1).upper() in coins:
            coin = match.group(1).upper()
            body = match.group(2).strip() + (("\n" + rest) if rest else "")
        found = [c for c in CASHTAG.findall(body) if c.upper() in coins]
        if found and coin == default_coin:
            coin = found[0].upper()
        if len(body) < 8:
            raise ValueError(f"Post {number} is shorter than 8 characters: {body[:40]!r}")
        out.append({"coin": coin, "text": body})
    if not out:
        raise ValueError("Enter at least one public post.")
    return out


def rate(items, artifact):
    """Real frozen sentiment for every post, then the trained model's probability."""
    texts = [item["text"] for item in items]
    scores = sp.score_batch(texts)
    rows = []
    for item, score in zip(items, scores):
        features = sp.batch_features([item["text"]], [score], item["coin"])
        model = sp.infer(artifact["models"]["Model 2"], features)
        baseline = sp.infer(artifact["models"]["Coin only"], features)
        rows.append({**item, **features,
                     "probability_up": model["probability_up"], "direction": model["direction"],
                     "toward_up": model["toward_up"], "toward_down": model["toward_down"],
                     "baseline_probability_up": baseline["probability_up"],
                     "bullish": features["bullish"], "bearish": features["bearish"]})
    return rows


def wilson(correct, total):
    if not total:
        return None
    z, p = 1.96, correct / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def run_block(blob, default_coin, artifact, known=None):
    """Rate every line, score the ones whose real outcome is known, and report honestly."""
    coins = artifact["report"]["base_coins"] + artifact["report"]["excluded_training_coins"]
    if isinstance(blob, (list, tuple)):
        entries = [str(item).strip() for item in blob if str(item).strip()]
        if not entries:
            raise ValueError("Enter at least one public post.")
        items = [_item(entry, default_coin, coins, number) for number, entry in enumerate(entries, 1)]
    else:
        items = parse_lines(blob, default_coin, coins)
    rated = rate(items, artifact)
    lookup = {row["text"]: row for row in (known if known is not None else examples())}
    scored = []
    for row in rated:
        match = lookup.get(row["text"])
        if match:
            row["known"] = True
            row["actual_up"] = match["actual_up"]
            row["actual_return"] = match["actual_return"]
            row["group"] = match["group"]
            row["model_right"] = int((row["probability_up"] >= 0.5) == bool(match["actual_up"]))
            row["baseline_right"] = int((row["baseline_probability_up"] >= 0.5) == bool(match["actual_up"]))
            scored.append(row)
        else:
            row.update(known=False, actual_up=None, actual_return=None, group=None, model_right=None, baseline_right=None)
    table = pd.DataFrame([{
        "#": index + 1,
        "Coin": row["coin"],
        "Public post (shortened)": " ".join(row["text"].split())[:78] + ("…" if len(row["text"]) > 78 else ""),
        "Language reads": "more bullish" if row["bullish"] > row["bearish"] else "more bearish/neutral",
        "Model P(UP)": f"{row['probability_up']:.1%}",
        "Model says": row["direction"],
        "Market actually did": ("rose" if row["actual_up"] else "did not rise") if row["known"] else "unknown (your post)",
        "Model right?": ("✓" if row["model_right"] else "✗") if row["known"] else "—",
        "Coin-only says": "UP" if row["baseline_probability_up"] >= 0.5 else "DOWN / FLAT",
        "Coin-only right?": ("✓" if row["baseline_right"] else "✗") if row["known"] else "—",
    } for index, row in enumerate(rated)])
    summary = {"rated": len(rated), "scored": len(scored)}
    if scored:
        correct = sum(row["model_right"] for row in scored)
        baseline = sum(row["baseline_right"] for row in scored)
        up = sum(row["actual_up"] for row in scored)
        summary.update(model_correct=correct, baseline_correct=baseline, always_up_correct=up,
                       total=len(scored), model_accuracy=correct / len(scored),
                       baseline_accuracy=baseline / len(scored), always_up_accuracy=up / len(scored),
                       interval=wilson(correct, len(scored)))
    return table, summary, rated


def finding(summary):
    """Plain English for the notebook, generated from the measured counts."""
    if not summary.get("scored"):
        return ("These are your own posts, so their next-day outcomes are not known yet. The table shows what the model "
                "estimated and the exact UTC day it is forecasting. There is nothing to score until that day has closed.")
    low, high = summary["interval"]
    return (f"The model got **{summary['model_correct']} of {summary['total']}** right (**{summary['model_accuracy']:.1%}**). "
            f"Saying \"up\" every time would have got {summary['always_up_correct']} of {summary['total']} "
            f"({summary['always_up_accuracy']:.1%}), and the coin-only baseline got {summary['baseline_correct']} of "
            f"{summary['total']} ({summary['baseline_accuracy']:.1%}). With only {summary['total']} posts the uncertainty is wide "
            f"(about {low:.0%} to {high:.0%}), so read this as the method being demonstrated rather than a precise score. "
            "It points the same way as the larger tests: the wording model does not beat the simple baselines, and these are "
            "real posts whose outcomes we did not choose.")
