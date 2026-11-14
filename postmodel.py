"""HypeCheck: the richer post-level model.

Question: does adding author behaviour, the words used, hype markers and past
trends to the market features make next-day direction predictable?

Three splits, because the split you choose decides the answer:
  A  flat random 80/20          - the split everyone reaches for. Leaks: most test
                                  rows share a calendar date, and almost every test
                                  post comes from an author already in train.
  B  chronological 80/20        - train on the past, test on the future, 1-day purge.
  C  walk-forward, 6 date blocks - 4 forward windows, refit each time.

Two controls decide whether any gain is real:
  label permutation - destroy the training labels, keep everything else.
  feature placebo   - shuffle the post columns across rows, keep the labels.

Writes postmodel.json. Never fetches anything; reads the pinned files.
"""
import argparse, json
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FLAGS = ["urgency","extreme_claim","promotion","evidence_terms","uncertainty",
         "bullish_terms","bearish_terms","exclamations","external_link"]
MEANS = ["word_count","caps_share","prior_author_posts_30d","prior_author_active_days_30d"]

def load(data_dir):
    p = pd.read_csv(f"{data_dir}/post-features-anon.csv")
    p["date"] = pd.to_datetime(p.cutoff_utc, utc=True)
    key = "author_id" if "author_id" in p.columns else "author_key"
    # author_id is empty for unattributed posts; read_csv turns that into NaN
    p["auth"] = p[key].fillna("UNKNOWN").replace("", "UNKNOWN").astype(str)
    p["known"] = (p["auth"] != "UNKNOWN").astype(int)
    A = p.groupby(["coin","date"]).agg(
        n_posts=("coin","size"), n_authors=("auth","nunique"), author_known=("known","mean"),
        **{f"s_{c}": (c,"sum") for c in FLAGS},
        **{f"m_{c}": (c,"mean") for c in MEANS}).reset_index()
    for c in FLAGS:
        A[f"share_{c}"] = A[f"s_{c}"] / A["n_posts"]
    A["log_posts"] = np.log1p(A.n_posts)
    A["polarity"] = (A.s_bullish_terms - A.s_bearish_terms) / A.m_word_count.replace(0, np.nan)
    top = p.groupby(["coin","date","auth"]).size().groupby(level=[0,1]).max().rename("top_n").reset_index()
    A = A.merge(top, on=["coin","date"], how="left")
    A["top_author_share"] = A.top_n / A.n_posts

    f = pd.read_csv(f"{data_dir}/features.csv")
    f["date"] = pd.to_datetime(f.date, utc=True)
    f = f.loc[f.date >= pd.Timestamp("2024-03-01", tz="UTC")].sort_values(["coin","date"])
    f["trail_up10"]  = f.groupby("coin").actual_up.transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    f["trail_up30"]  = f.groupby("coin").actual_up.transform(lambda s: s.shift(1).rolling(30, min_periods=5).mean())
    f["trail_ret10"] = f.groupby("coin").actual_return.transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    M = A.merge(f, on=["coin","date"], how="inner").sort_values("date").reset_index(drop=True)

    # Author-behaviour means are undefined on days where every post is unattributed
    # (64% of rows carry no author). Impute 0 and keep author_known in the model so the
    # learner can tell "no known author" from "an author with a zero". polarity is also
    # undefined for empty text. Both are flagged in the write-up, not silently hidden.
    blocks = [c for c in ALLF + OURSENT if c in M.columns]
    M[blocks] = M[blocks].astype(float).fillna(0.0)
    return M, p

MK      = ["ret1","ret3","ret7","vol7","range1","volume_change"]
TREND   = ["trail_up10","trail_up30","trail_ret10"]
EXTRA   = ["btc_ret1","btc_ret7","hourly_vol1","hourly_range1","last_hour_ret","taker_ratio1"]
POSTVOL = ["log_posts","n_posts","n_authors","top_author_share"]
WORDS   = ["polarity","m_word_count","m_caps_share"] + [f"share_{c}" for c in FLAGS]
AUTHOR  = ["author_known","m_prior_author_posts_30d","m_prior_author_active_days_30d"]
OURSENT = ["sentiment","disagreement","bullish","bearish","post_count","has_posts"]
ALLF    = MK + TREND + EXTRA + POSTVOL + WORDS + AUTHOR

LADDER = [("market (6)", MK),
          ("+ past trend (3)", MK + TREND),
          ("+ BTC & intraday (6)", MK + TREND + EXTRA),
          ("+ post volume (4)", MK + TREND + EXTRA + POSTVOL),
          ("+ the words used (12)", MK + TREND + EXTRA + POSTVOL + WORDS),
          ("+ author behaviour (3)", ALLF),
          ("+ FinTwitBERT sentiment (6)", ALLF + OURSENT)]

def model(kind, seed=0):
    if kind == "dummy":
        return DummyClassifier(strategy="most_frequent")
    if kind == "gb":
        return GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=seed)
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=4000))

def evaluate(tr, te, cols, kind, seed=0):
    m = model(kind, seed).fit(tr[cols].to_numpy(float), tr.actual_up.to_numpy(int))
    pr = m.predict_proba(te[cols].to_numpy(float))[:,1]
    y = te.actual_up.to_numpy(int)
    base = float(max(y.mean(), 1 - y.mean()))
    return dict(acc=float(accuracy_score(y, (pr > .5).astype(int))), base=base,
                edge=float(accuracy_score(y, (pr > .5).astype(int)) - base),
                brier=float(brier_score_loss(y, pr)),
                logloss=float(log_loss(y, np.clip(pr, 1e-9, 1 - 1e-9))),
                auc=float(roc_auc_score(y, pr)), n=int(len(y)))

def make_splits(M, blocks=6):
    dates = np.sort(M.date.unique())
    idx = np.arange(len(M))
    ti, ei = train_test_split(idx, test_size=0.2, random_state=20261114)
    flat = (M.iloc[ti], M.iloc[ei])
    cut = dates[int(round(len(dates) * 0.8))]
    chrono = (M.loc[M.date < (cut - pd.Timedelta(days=1))], M.loc[M.date >= cut])
    bnd = [dates[int(round(i * len(dates) / blocks))] for i in range(blocks)] + [dates[-1] + pd.Timedelta(days=1)]
    folds = []
    for i in range(1, blocks):
        lo, hi = bnd[i], bnd[i+1]
        tr = M.loc[M.date < (lo - pd.Timedelta(days=1))]
        te = M.loc[np.logical_and(M.date >= lo, M.date < hi)]
        if len(te) >= 20 and len(tr) >= 150:
            folds.append((tr, te))
    return flat, chrono, folds, cut

def show(tr, te, title, seeds_gb=3):
    base = float(max(te.actual_up.mean(), 1 - te.actual_up.mean()))
    print(f"\n=== {title}")
    print(f"    train {len(tr)} coin-days | test {len(te)} | test majority-class rate {base:.4f}")
    print(f"    {'feature block':30s}{'LOGIT acc':>11s}{'edge':>8s}{'AUC':>7s}{'Brier':>8s}"
          f"{'GBM acc':>9s}{'edge':>8s}{'AUC':>7s}{'Brier':>8s}")
    rows = {}
    for lab, cols in LADDER:
        rl = evaluate(tr, te, cols, "logit")
        g = [evaluate(tr, te, cols, "gb", s) for s in range(seeds_gb)]
        ga = float(np.mean([x["acc"] for x in g])); gu = float(np.mean([x["auc"] for x in g]))
        gb = float(np.mean([x["brier"] for x in g]))
        rows[lab] = dict(logit=rl, gbm_acc=ga, gbm_edge=ga-base, gbm_auc=gu, gbm_brier=gb)
        print(f"    {lab:30s}{rl['acc']:11.4f}{rl['edge']:+8.4f}{rl['auc']:7.4f}{rl['brier']:8.4f}"
              f"{ga:9.4f}{ga-base:+8.4f}{gu:7.4f}{gb:8.4f}")
    d = evaluate(tr, te, MK, "dummy")
    print(f"    {'majority baseline':30s}{d['acc']:11.4f}{0.0:+8.4f}{d['auc']:7.4f}{d['brier']:8.4f}")
    return rows

def show_walk(folds, seeds_gb=3):
    n = sum(len(t) for _, t in folds)
    print(f"\n=== SPLIT C - walk-forward: {len(folds)} forward windows, {n} out-of-sample coin-days")
    print(f"    {'feature block':30s}{'LOGIT acc':>11s}{'AUC':>7s}{'GBM acc':>9s}{'AUC':>7s}{'Brier':>8s}{'folds>base':>12s}")
    rows = {}
    for lab, cols in LADDER:
        la, lu, ga, gu, gb, win = [], [], [], [], [], 0
        for tr, te in folds:
            rl = evaluate(tr, te, cols, "logit")
            g = [evaluate(tr, te, cols, "gb", s) for s in range(seeds_gb)]
            a = float(np.mean([x["acc"] for x in g]))
            la.append(rl["acc"]); lu.append(rl["auc"]); ga.append(a)
            gu.append(float(np.mean([x["auc"] for x in g]))); gb.append(float(np.mean([x["brier"] for x in g])))
            if a > max(te.actual_up.mean(), 1 - te.actual_up.mean()):
                win += 1
        rows[lab] = dict(logit_acc=float(np.mean(la)), logit_auc=float(np.mean(lu)),
                         gbm_acc=float(np.mean(ga)), gbm_auc=float(np.mean(gu)),
                         gbm_brier=float(np.mean(gb)), folds_beating_base=win, folds=len(folds))
        print(f"    {lab:30s}{np.mean(la):11.4f}{np.mean(lu):7.4f}{np.mean(ga):9.4f}{np.mean(gu):7.4f}"
              f"{np.mean(gb):8.4f}{win:>9d}/{len(folds)}")
    return rows

def controls(folds, cols, label, perms=20, placebos=10, seed=0):
    rng = np.random.default_rng(seed)
    real = float(np.mean([evaluate(tr, te, cols, "gb", 0)["auc"] for tr, te in folds]))
    null = []
    for _ in range(perms):
        a = []
        for tr, te in folds:
            t2 = tr.copy(); t2["actual_up"] = rng.permutation(t2.actual_up.to_numpy())
            e2 = te.copy(); e2["actual_up"] = rng.permutation(e2.actual_up.to_numpy())
            a.append(evaluate(t2, e2, cols, "gb", 0)["auc"])
        null.append(float(np.mean(a)))
    null = np.array(null)
    plac = []
    for _ in range(placebos):
        a = []
        for tr, te in folds:
            t2 = tr.copy(); e2 = te.copy()
            for c in POSTVOL + WORDS + AUTHOR:
                v = t2[c].to_numpy().copy(); rng.shuffle(v); t2[c] = v
                v = e2[c].to_numpy().copy(); rng.shuffle(v); e2[c] = v
            a.append(evaluate(t2, e2, cols, "gb", 0)["auc"])
        plac.append(float(np.mean(a)))
    plac = np.array(plac)
    print(f"\n=== controls on the walk-forward ({label}, gradient boosting)")
    print(f"    real AUC                          {real:.4f}")
    print(f"    label-permuted null ({perms:2d} draws)   mean {null.mean():.4f}  sd {null.std():.4f}  max {null.max():.4f}  -> p = {(null >= real).mean():.2f}")
    print(f"    post-feature placebo ({placebos:2d} draws) mean {plac.mean():.4f}  sd {plac.std():.4f}")
    return dict(real_auc=real, null_mean=float(null.mean()), null_sd=float(null.std()),
                null_max=float(null.max()), p_value=float((null >= real).mean()),
                placebo_mean=float(plac.mean()), placebo_sd=float(plac.std()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--json", default="postmodel.json")
    args = ap.parse_args()

    M, p = load(args.data_dir)
    print("=" * 100)
    print("HypeCheck - the richer post-level model")
    print(f"  coin-days with any post: {len(M)}  |  dates: {M.date.nunique()}  |  "
          f"{M.date.min().date()} -> {M.date.max().date()}  |  up-rate {M.actual_up.mean():.4f}")
    print(f"  posts: {len(p)}  |  distinct anonymised authors: {p.author_id.replace('', np.nan).nunique()}"
          f"  |  posts with a known author: {p.author_known.sum()} ({p.author_known.mean():.1%})")
    print(f"  feature blocks: market {len(MK)} + trend {len(TREND)} + btc/intraday {len(EXTRA)}"
          f" + post volume {len(POSTVOL)} + words {len(WORDS)} + author {len(AUTHOR)}"
          f"  =  {len(ALLF)} model features (+{len(OURSENT)} FinTwitBERT in the last row)")

    flat, chrono, folds, cut = make_splits(M)
    ra = show(flat[0], flat[1], "SPLIT A - flat random 80/20 (the split to distrust)")
    shared = np.intersect1d(flat[1].date.unique(), flat[0].date.unique())
    au_tr = set(p.loc[p.date.isin(flat[0].date.unique()), "auth"])
    au_te = p.loc[p.date.isin(flat[1].date.unique()), "auth"]
    leak = dict(test_rows_sharing_a_date_with_train=float(flat[1].date.isin(shared).mean()),
                test_posts_from_authors_seen_in_train=float(au_te.isin(au_tr).mean()))
    print(f"    LEAKAGE: {leak['test_rows_sharing_a_date_with_train']:.1%} of test rows share a calendar date with train")
    print(f"    LEAKAGE: {leak['test_posts_from_authors_seen_in_train']:.1%} of test posts come from authors already in train")

    rb = show(chrono[0], chrono[1], f"SPLIT B - chronological 80/20, 1-day purge (cut {cut.date()})")
    rc = show_walk(folds)
    ctrl = controls(folds, ALLF, "all post+market features")

    a_best = max(ra.items(), key=lambda t: t[1]["gbm_acc"])
    b_best = max(rb.items(), key=lambda t: t[1]["gbm_acc"])
    c_best = max(rc.items(), key=lambda t: t[1]["gbm_acc"])
    print("\n" + "=" * 100)
    print("READ THIS BEFORE QUOTING ANY NUMBER")
    print(f"  flat 80/20 (leaky)   best {a_best[0]:30s} GBM acc {a_best[1]['gbm_acc']:.4f}  edge {a_best[1]['gbm_edge']:+.4f}")
    print(f"  chronological 80/20  best {b_best[0]:30s} GBM acc {b_best[1]['gbm_acc']:.4f}  edge {b_best[1]['gbm_edge']:+.4f}")
    print(f"  walk-forward         best {c_best[0]:30s} GBM acc {c_best[1]['gbm_acc']:.4f}  "
          f"auc {c_best[1]['gbm_auc']:.4f}  folds beating base {c_best[1]['folds_beating_base']}/{c_best[1]['folds']}")
    print(f"  the flat split flatters every block by {a_best[1]['gbm_acc']-c_best[1]['gbm_acc']:+.4f} accuracy vs walk-forward,")
    print(f"  because {leak['test_rows_sharing_a_date_with_train']:.0%} of its test rows share a date with training rows.")
    print(f"  controls: real AUC {ctrl['real_auc']:.4f} vs label-permuted {ctrl['null_mean']:.4f} "
          f"(max {ctrl['null_max']:.4f}, p={ctrl['p_value']:.2f}) and feature-placebo {ctrl['placebo_mean']:.4f}.")
    print("  A model that beats the majority baseline in 1 of 4 forward windows is not a trading signal.")
    print("=" * 100)

    out = dict(rows=len(M), dates=int(M.date.nunique()), posts=len(p),
               posts_with_known_author=int(p.author_known.sum()),
               distinct_anonymised_authors=int(p.author_id.replace("", np.nan).nunique()),
               n_features=len(ALLF), n_features_with_sentiment=len(ALLF + OURSENT),
               split_a_flat=ra, split_b_chrono=rb, split_c_walkforward=rc,
               leakage=leak, controls=ctrl,
               best=dict(flat=a_best[0], chrono=b_best[0], walk=c_best[0]))
    with open(args.json, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {args.json}")

if __name__ == "__main__":
    main()
