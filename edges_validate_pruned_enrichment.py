#!/usr/bin/env python3
"""
Degree-corrected enrichment analysis at a user-specified MI cutoff.

What it does
------------
1) Loads cmi_results.csv (parent_gene, target_gene, mi, prediction_score, ...).
2) Loads reference TF->target edges from TRRUST + CORE (directed).
3) Filters candidate edges to MI >= --mi_cutoff, then ranks by --score_col (default: prediction_score).
4) Reports detailed enrichment:
   - Global Precision/Recall/Enrichment@K for multiple K
   - Per-TF coverage/enrichment@K
   - Degree-corrected global permutation p-value for TP count at a chosen K

Degree-corrected null
---------------------
For each TF, preserves k = number of predicted targets (within top-K),
then samples k targets from the target universe with probability proportional
to target in-degree (how often each target appears) in the *MI-filtered candidate graph*.
This corrects "hub target" bias.

Outputs
-------
- <out_prefix>_summary.json
- <out_prefix>_global_topk_metrics.csv
- <out_prefix>_perTF_topk_metrics.csv
- <out_prefix>_degcorr_global_permutation.csv
"""

import argparse
import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter


# =============================================================
# Reference loaders (TRRUST + CORE)
# =============================================================

def load_trrust(trrust_file):
    print("[TRRUST] Loading: {}".format(trrust_file))
    trrust_raw = pd.read_csv(trrust_file, sep="\t", header=None)

    if trrust_raw.shape[1] == 4:
        print("[TRRUST] Detected 4-column format.")
        trrust_raw.columns = ["tf_symbol", "target_symbol", "direction", "pmid"]
        trrust = trrust_raw[["tf_symbol", "target_symbol"]].copy()
    elif trrust_raw.shape[1] >= 6:
        print("[TRRUST] Detected >=6-column format (using first 6 cols).")
        trrust_raw = trrust_raw.iloc[:, :6].copy()
        trrust_raw.columns = ["tf_symbol", "tf_ncbi", "target_symbol", "target_ncbi",
                              "tf_type", "target_type"]
        trrust = trrust_raw[
            (trrust_raw["tf_type"] == "TF") &
            (trrust_raw["target_type"] != "miRNA")
        ][["tf_symbol", "target_symbol"]].copy()
    else:
        raise ValueError("Unrecognized TRRUST format: {} columns".format(trrust_raw.shape[1]))

    trrust.dropna(subset=["tf_symbol", "target_symbol"], inplace=True)
    trrust["tf_symbol"] = trrust["tf_symbol"].astype(str)
    trrust["target_symbol"] = trrust["target_symbol"].astype(str)
    print("[TRRUST] Loaded {} edges.".format(len(trrust)))
    return trrust


def load_core_tf_target(core_file):
    print("[CORE] Loading: {}".format(core_file))
    core = pd.read_csv(core_file, sep="\t", header=None)

    core = core[core.iloc[:, 3].notna()].copy()
    while core.shape[1] < 6:
        core[core.shape[1]] = np.nan

    core = core.iloc[:, :6].copy()
    core.columns = ["tf_symbol", "tf_ncbi", "target_symbol", "target_ncbi",
                    "tf_type", "target_type"]

    core = core[(core["tf_type"] == "TF") & (core["target_type"] != "miRNA")].copy()
    core.dropna(subset=["tf_symbol", "target_symbol"], inplace=True)
    core["tf_symbol"] = core["tf_symbol"].astype(str)
    core["target_symbol"] = core["target_symbol"].astype(str)
    print("[CORE] Loaded {} edges after filtering.".format(len(core)))
    return core


def build_ref_tf_map(trrust_df, core_df):
    ref_tf_map = {}
    for _, r in trrust_df.iterrows():
        tf = str(r["tf_symbol"])
        tg = str(r["target_symbol"])
        ref_tf_map.setdefault(tf, set()).add(tg)
    for _, r in core_df.iterrows():
        tf = str(r["tf_symbol"])
        tg = str(r["target_symbol"])
        ref_tf_map.setdefault(tf, set()).add(tg)
    return ref_tf_map


# =============================================================
# Helpers
# =============================================================

def safe_to_numeric(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def build_labels(parent_arr, target_arr, ref_tf_map):
    y = np.zeros(len(parent_arr), dtype=int)
    for i in range(len(parent_arr)):
        tf = parent_arr[i]
        tg = target_arr[i]
        if tg in ref_tf_map.get(tf, set()):
            y[i] = 1
    return y


def compute_target_degree(target_arr):
    deg = Counter()
    for tg in target_arr:
        deg[tg] += 1
    return deg


def build_pred_tf_targets_at_topk(parent_sorted, target_sorted, K):
    tf_map = defaultdict(set)
    for i in range(int(K)):
        tf_map[parent_sorted[i]].add(target_sorted[i])
    return tf_map


def topk_metrics(score_sorted, y_sorted, k_list, base_rate):
    tp_cum = np.cumsum(y_sorted == 1)
    n_total = len(y_sorted)
    n_pos_total = int(np.sum(y_sorted == 1))

    recs = []
    for K in k_list:
        K = int(K)
        if K <= 0:
            continue
        if K > n_total:
            K = n_total
        tp = int(tp_cum[K - 1])
        prec = tp / float(K) if K > 0 else np.nan
        rec = tp / float(n_pos_total) if n_pos_total > 0 else np.nan
        enr = (prec / base_rate) if (base_rate is not None and base_rate > 0) else np.nan

        recs.append({
            "K": int(K),
            "TP": int(tp),
            "precision": float(prec),
            "recall": float(rec),
            "enrichment_over_base_rate": float(enr),
            "n_pos_total": int(n_pos_total),
            "n_total": int(n_total),
        })
    return pd.DataFrame(recs)


def degree_corrected_global_perm(ref_tf_map, pred_tf_map, targets_universe, probs, num_sims=500, seed=1):
    rng = np.random.default_rng(seed)

    # Observed TP
    T_obs = 0
    for tf, pred_tgts in pred_tf_map.items():
        ref_tgts = ref_tf_map.get(tf, set())
        if not ref_tgts:
            continue
        T_obs += len(set(pred_tgts) & set(ref_tgts))

    null = []
    for _ in range(int(num_sims)):
        T = 0
        for tf, pred_tgts in pred_tf_map.items():
            ref_tgts = ref_tf_map.get(tf, set())
            if not ref_tgts:
                continue
            k = len(pred_tgts)
            if k <= 0:
                continue
            kk = min(k, len(targets_universe))
            sampled = set(rng.choice(targets_universe, size=kk, replace=False, p=probs).tolist())
            T += len(sampled & set(ref_tgts))
        null.append(T)

    null = np.asarray(null, dtype=float)
    p_emp = float(np.mean(null >= float(T_obs)))
    return float(T_obs), float(null.mean()), float(null.std()), p_emp


# =============================================================
# MAIN
# =============================================================

def main():
    p = argparse.ArgumentParser(description="Degree-corrected enrichment at a user-set MI cutoff.")

    p.add_argument("--pred_file", default="cmi_results.csv")
    p.add_argument("--trrust_file", default="trrust_rawdata.human.tsv")
    p.add_argument("--core_file", default="human_core_TF_Target.txt")

    p.add_argument("--mi_cutoff", type=float, required=True,
                   help="Keep rows with mi >= this cutoff.")
    p.add_argument("--mi_col", default="mi")
    p.add_argument("--score_col", default="prediction_score",
                   help="Column to rank by (default: prediction_score).")

    p.add_argument("--restrict_parent_to_ref_tfs", action="store_true")
    p.add_argument("--drop_self_edges", action="store_true")

    p.add_argument("--topk", default="100,500,1000,5000,10000,25000,50000",
                   help="Comma-separated K values for top-K metrics.")

    p.add_argument("--perm_K", type=int, default=500,
                   help="K used for degree-corrected permutation test (default: 50000).")
    p.add_argument("--num_sims", type=int, default=10000)
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--out_prefix", default="enrichment_mi")
    
    p.add_argument("--no_dedup_pairs", action="store_true",
               help="Disable deduplication of (parent_gene,target_gene) pairs.")
    p.add_argument("--dedup_score_agg", default="max", choices=["max", "mean", "median"],
                   help="How to aggregate score across duplicate (parent,target) pairs (default: max).")
    args = p.parse_args()

    # -------------------------
    # 1) References
    # -------------------------
    trrust = load_trrust(args.trrust_file)
    core = load_core_tf_target(args.core_file)
    ref_tf_map = build_ref_tf_map(trrust, core)
    tf_set = set(ref_tf_map.keys())
    print("[Ref] TFs in reference: {}".format(len(tf_set)))

    # -------------------------
    # 2) Load predictions + filter MI
    # -------------------------
    df = pd.read_csv(args.pred_file)
    df.columns = df.columns.str.lower()

    mi_col = args.mi_col.lower()
    sc_col = args.score_col.lower()

    need = {"parent_gene", "target_gene", mi_col, sc_col}
    if not need.issubset(set(df.columns)):
        raise ValueError("pred_file must contain columns: {}".format(sorted(list(need))))

    parent = df["parent_gene"].astype(str)
    target = df["target_gene"].astype(str)

    mi = safe_to_numeric(df[mi_col])
    score = safe_to_numeric(df[sc_col])

    m = mi.notna() & score.notna() & (mi.values >= float(args.mi_cutoff))
    if args.drop_self_edges:
        m = m & (parent != target)
    if args.restrict_parent_to_ref_tfs:
        m = m & parent.isin(tf_set)

    if not m.any():
        raise ValueError("No rows left after MI cutoff + filters.")

    parent_f = parent[m].values
    target_f = target[m].values
    score_f = score[m].astype(float).values
    
    # -------------------------
    # 2b) OPTIONAL: Deduplicate triplets into unique (parent,target)
    # -------------------------
    if not args.no_dedup_pairs:
        tmp = pd.DataFrame({
            "parent_gene": parent_f,
            "target_gene": target_f,
            "score": score_f
        })
    
        if args.dedup_score_agg == "max":
            tmp = tmp.groupby(["parent_gene", "target_gene"], as_index=False)["score"].max()
        elif args.dedup_score_agg == "mean":
            tmp = tmp.groupby(["parent_gene", "target_gene"], as_index=False)["score"].mean()
        else:
            tmp = tmp.groupby(["parent_gene", "target_gene"], as_index=False)["score"].median()
    
        n_before = len(score_f)
    
        parent_f = tmp["parent_gene"].astype(str).values
        target_f = tmp["target_gene"].astype(str).values
        score_f  = tmp["score"].astype(float).values
    
        print("[Dedup] enabled | agg={} | rows_before={} rows_after={} dup_factor={:.3f}".format(
            args.dedup_score_agg,
            n_before,
            len(score_f),
            float(n_before) / max(1, len(score_f))
        ))
    else:
        print("[Dedup] disabled by user (--no_dedup_pairs)")

        if args.dedup_score_agg == "max":
            tmp = tmp.groupby(["parent_gene", "target_gene"], as_index=False)["score"].max()
        elif args.dedup_score_agg == "mean":
            tmp = tmp.groupby(["parent_gene", "target_gene"], as_index=False)["score"].mean()
        else:
            tmp = tmp.groupby(["parent_gene", "target_gene"], as_index=False)["score"].median()

        parent_f = tmp["parent_gene"].astype(str).values
        target_f = tmp["target_gene"].astype(str).values
        score_f = tmp["score"].astype(float).values

        print("[Dedup] Collapsed to unique pairs: {}".format(len(score_f)))

    # Sort by score desc
    order = np.argsort(-score_f)
    parent_s = parent_f[order]
    target_s = target_f[order]
    score_s = score_f[order]

    # Labels in this filtered universe
    y_s = build_labels(parent_s, target_s, ref_tf_map)
    n_total = int(len(y_s))
    n_pos = int(np.sum(y_s == 1))
    base_rate = (n_pos / float(n_total)) if n_total > 0 else np.nan

    print("[Data] mi_cutoff={} | rows={} | positives={} | base_rate={}".format(
        args.mi_cutoff, n_total, n_pos, base_rate
    ))

    # Save a small summary
    summary = {
        "pred_file": args.pred_file,
        "mi_cutoff": float(args.mi_cutoff),
        "mi_col": mi_col,
        "score_col": sc_col,
        "rows_scored": int(n_total),
        "n_pos": int(n_pos),
        "n_neg": int(n_total - n_pos),
        "base_rate": float(base_rate) if base_rate == base_rate else None,
        "restrict_parent_to_ref_tfs": bool(args.restrict_parent_to_ref_tfs),
        "drop_self_edges": bool(args.drop_self_edges),
        "perm_K": int(args.perm_K),
        "num_sims": int(args.num_sims),
        "seed": int(args.seed),
    }
    summary_out = "{}_summary.json".format(args.out_prefix)
    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    print("[Out] Saved:", summary_out)

    # -------------------------
    # 3) Global top-K metrics
    # -------------------------
    k_list = [int(x.strip()) for x in args.topk.split(",") if x.strip()]
    global_df = topk_metrics(score_s, y_s, k_list, base_rate)

    global_out = "{}_global_topk_metrics.csv".format(args.out_prefix)
    global_df.to_csv(global_out, index=False)
    print("[Out] Saved:", global_out)

    # -------------------------
    # 4) Per-TF top-K metrics
    # -------------------------
    per_tf_records = []

    targets_universe = np.array(sorted(set(target_f.tolist())), dtype=object)
    n_targets_univ = int(len(targets_universe))

    for K in k_list:
        K_eff = int(min(int(K), n_total))
        if K_eff <= 0:
            continue

        pred_tf_map_topk = build_pred_tf_targets_at_topk(parent_s, target_s, K_eff)

        for tf, pred_tgts in pred_tf_map_topk.items():
            ref_tgts = ref_tf_map.get(tf, set())
            if not ref_tgts:
                continue

            TP = len(set(pred_tgts) & set(ref_tgts))
            k_pred = len(pred_tgts)
            n_ref = len(ref_tgts)

            cov = TP / float(n_ref) if n_ref > 0 else np.nan

            # TF-specific expected under random over targets universe
            p_ref_tf = (n_ref / float(n_targets_univ)) if n_targets_univ > 0 else np.nan
            expected = k_pred * p_ref_tf if p_ref_tf is not None else np.nan
            enr = (TP / expected) if (expected is not None and expected > 0) else np.nan

            per_tf_records.append({
                "K": int(K_eff),
                "TF": tf,
                "TP": int(TP),
                "k_pred": int(k_pred),
                "n_ref_targets": int(n_ref),
                "coverage": float(cov),
                "enrichment_vs_TF_random": float(enr)
            })

    per_tf_df = pd.DataFrame(per_tf_records)
    per_tf_out = "{}_perTF_topk_metrics.csv".format(args.out_prefix)
    per_tf_df.to_csv(per_tf_out, index=False)
    print("[Out] Saved:", per_tf_out)

    # -------------------------
    # 5) Degree-corrected global permutation at perm_K
    # -------------------------
    perm_K = int(min(int(args.perm_K), n_total))
    if perm_K <= 0:
        raise ValueError("perm_K must be > 0 after clipping to n_total.")

    # TF -> set(targets) in top perm_K
    pred_tf_map_perm = build_pred_tf_targets_at_topk(parent_s, target_s, perm_K)

    # Build target-degree distribution from MI-filtered candidate graph (all filtered rows, unsorted)
    deg = compute_target_degree(target_f)
    targets_universe = np.array(sorted(deg.keys()), dtype=object)
    weights = np.array([deg.get(t, 1) for t in targets_universe], dtype=float)
    probs = weights / weights.sum()

    T_obs, T_null_mean, T_null_sd, p_emp = degree_corrected_global_perm(
        ref_tf_map=ref_tf_map,
        pred_tf_map=pred_tf_map_perm,
        targets_universe=targets_universe,
        probs=probs,
        num_sims=args.num_sims,
        seed=args.seed
    )

    dc_df = pd.DataFrame([{
        "mi_cutoff": float(args.mi_cutoff),
        "perm_K": int(perm_K),
        "T_obs": float(T_obs),
        "T_null_mean": float(T_null_mean),
        "T_null_sd": float(T_null_sd),
        "empirical_p": float(p_emp),
        "num_sims": int(args.num_sims),
        "base_rate_in_filtered_space": float(base_rate) if base_rate == base_rate else np.nan
    }])

    dc_out = "{}_degcorr_global_permutation.csv".format(args.out_prefix)
    dc_df.to_csv(dc_out, index=False)
    print("[Out] Saved:", dc_out)

    print("Done.")


if __name__ == "__main__":
    main()
