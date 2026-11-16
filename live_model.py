"""Train and export the browser-runnable HypeCheck live model (wording + coin).

The full submitted model adds a frozen FinTwitBERT sentiment block, which cannot run
in a static page. This script trains the same word TF-IDF + coin pipeline WITHOUT the
sentiment block on the identical rows, target and coin-day weighting, evaluates it on
the identical held-out splits, and exports a compact artifact for docs/live.js.
Nothing here changes the submitted model or its measured results.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

import sample_pipeline as sp

ROOT = Path(__file__).resolve().parent
TARGET = "Next full UTC day after snapshot cutoff, close > open; flat is not up"


def live_encoder():
    return ColumnTransformer([
        ("text", TfidfVectorizer(), "text"),
        ("coin", OneHotEncoder(handle_unknown="ignore"), ["coin"]),
    ])


def export_linear(prep, model):
    coef = model.coef_[0]
    out = {"intercept": float(model.intercept_[0]), "blocks": []}
    for name, transformer, _ in prep.transformers_:
        if name == "remainder":
            continue
        block = {"name": name, "coefficient": coef[prep.output_indices_[name]].tolist()}
        if name == "text":
            block.update(
                vocabulary={k: int(v) for k, v in transformer.vocabulary_.items()},
                idf=transformer.idf_.tolist(),
                analyzer=transformer.analyzer,
                lowercase=bool(transformer.lowercase),
                sublinear_tf=bool(transformer.sublinear_tf),
                ngram_range=list(transformer.ngram_range),
                token_pattern=transformer.token_pattern,
            )
        else:
            block.update(categories=transformer.categories_[0].tolist())
        out["blocks"].append(block)
    return out


def main():
    frame, train, later, unseen = sp.dataset()
    prep = live_encoder()
    x = prep.fit_transform(train)
    model = LogisticRegression(max_iter=1000)
    model.fit(x, train.actual_up, sample_weight=sp.weight(train))
    coin_prep = sp.encoder(coin_only=True)
    coin_model = LogisticRegression(max_iter=1000)
    coin_model.fit(coin_prep.fit_transform(train), train.actual_up, sample_weight=sp.weight(train))
    models = {"Live model": (prep, model), "Coin only": (coin_prep, coin_model)}
    report = {}
    for label, part in [("later_time", later), ("unseen_coin_market", unseen)]:
        scores, _ = sp.evaluate(models, train, part)
        report[label] = scores
        print(label, {k: round(v["accuracy"], 4) for k, v in scores.items()})
    vocab = prep.named_transformers_["text"].get_feature_names_out()
    coef = model.coef_[0][prep.output_indices_["text"]]
    order = np.argsort(coef)
    top = {"up": [str(vocab[i]) for i in order[-12:][::-1]], "down": [str(vocab[i]) for i in order[:12]]}
    artifact = {
        "version": 1,
        "name": "HypeCheck live model - wording + coin (no neural sentiment block)",
        "target": TARGET,
        "coins": list(sp.COINS),
        "trained_coins": list(sp.BASE_COINS),
        "held_out_coins": list(sp.NEW_COINS),
        "training": {"snapshots": len(train), "coin_days": int(train[["date", "coin"]].drop_duplicates().shape[0]),
                     "first": str(train.date.min().date()), "last": str(train.date.max().date()),
                     "post_count_range": [int(frame.post_count.min()), int(frame.post_count.max())]},
        "model": export_linear(prep, model),
        "coin_baseline": export_linear(coin_prep, coin_model),
        "metrics": report,
        "top_words": top,
        "test_reuse": "Exploratory: this archive and its final dates were examined before. Not independent confirmation.",
        "sentiment_note": ("The submitted model adds a frozen FinTwitBERT sentiment block, which needs the Colab "
                           "runtime. This browser model is the identical wording + coin pipeline without that block."),
        "disclaimer": "This project is a prototype and is not financial advice.",
    }
    out = ROOT / "docs" / "live-model.json"
    out.write_text(json.dumps(artifact, separators=(",", ":")))
    print("wrote", out, round(out.stat().st_size / 1024, 1), "KB; vocab", len(artifact["model"]["blocks"][0]["vocabulary"]))
    # parity fixtures: real held-out rows scored by sklearn, for the browser test
    cases = []
    for part in (later, unseen):
        for _, row in part.iloc[::max(1, len(part) // 9)].iterrows():
            cases.append({"coin": row.coin, "text": row.text,
                          "expected": float(model.predict_proba(prep.transform(part[part.index == row.name]))[:, 1][0])})
    (ROOT / "docs" / "live-parity.json").write_text(json.dumps(cases[:18], indent=1))
    print("parity cases:", len(cases[:18]))


if __name__ == "__main__":
    main()
