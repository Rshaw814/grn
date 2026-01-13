#!/usr/bin/env python3
"""
Scan AUROC/AUPR vs a score (prediction_score or MI),
while filtering rows by a cutoff on (CMI or MI).

IMPORTANT: cmi_results.csv contains triplets (target,parent,cond). This script
COLLAPSES triplets to unique directed pairs (parent_gene -> target_gene) to avoid
double counting the same edge multiple times.

Dedup strategy:
  - group by (parent_gene, target_gene)
  - keep max(score_col) and max(cutoff_col) by default
    (you can switch to mean/median if you want)

Positives: directed TF->target edges from TRRUST + CORE.

Outputs: a CSV with AUROC/AUPR for each cutoff.
"""

import numpy as np
import pandas as pd

# Optional sklearn
try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False


# ============================
# CONFIG (edit these)
# ============================

PRED_FILE   = "cmi_results.csv"
TRRUST_FILE = "trrust_rawdata.human.tsv"
CORE_FILE   = "human_core_TF_Target.txt"

# Choose cutoff column and score column
# cutoff_col can be "cmi" or "mi"
# score_col  can be "prediction_score" or "mi"
CUTOFF_COL = "mi"                 # e.g. "mi" or "cmi"
SCORE_COL  = "prediction_score"   # e.g. "prediction_score" or "mi"

# Filters
RESTRICT_PARENT_TO_REF_TFS = True
DROP_SELF_EDGES = True

# Scan grid
MIN_CUTOFF = 0.0
MAX_CUTOFF = 0.10
NUM_CUTOFFS = 21
# Or specify explicit cutoffs (set to None to use linspace)
EXPLICIT_CUTOFFS = None  # e.g. [0.0, 0.005, 0.01, 0.02, 0.05]

OUT_CSV = "cutoff_scan_auroc_aupr_dedup_pairs.csv"

# Dedup aggregation across triplets for the same (parent,target)
# Options: "max", "mean", "median"
DEDUP_AGG = "max"


# ============================
# Reference loaders
# ============================

def load_trrust(trrust_file):
    print("[TRRUST] Loading: {}".format(trrust_file))
    trrust_raw = pd.read_csv(trrust_file, sep="\t", header=None)

    if trrust_raw.shape[1] == 4:
        trrust_raw.columns = ["tf_symbol", "target_symbol", "direction", "pmid"]
        trrust = trrust_raw[["tf_symbol", "target_symbol"]].copy()

    elif trrust_raw.shape[1] >= 6:
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


# ============================
# Metrics (fallback if no sklearn)
# ============================

def auroc_numpy(y_true, y_score):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return np.nan

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=float)

    # average ranks for ties
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg_rank = 0.5 * (ranks[order[i]] + ranks[order[j]])
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
        i = j + 1

    sum_ranks_pos = float(np.sum(ranks[y_true == 1]))
    U = sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)
    return float(U / float(n_pos * n_neg))


def aupr_numpy(y_true, y_score):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return np.nan

    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / float(n_pos)

    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return float(np.trapz(precision, recall))


def compute_au(y_true, y_score):
    if _HAVE_SK and (len(np.unique(y_true)) == 2):
        return float(roc_auc_score(y_true, y_score)), float(average_precision_score(y_true, y_score)), "sklearn"
    return auroc_numpy(y_true, y_score), aupr_numpy(y_true, y_score), "numpy_fallback"


# ============================
# Dedup / aggregation
# ============================

def agg_series(x, mode):
    if mode == "max":
        return float(np.nanmax(x.values))
    if mode == "mean":
        return float(np.nanmean(x.values))
    if mode == "median":
        return float(np.nanmedian(x.values))
    raise ValueError("Unknown DEDUP_AGG: {}".format(mode))


def dedup_triplets_to_pairs(df, score_col, cutoff_col, agg_mode):
    """
    Collapse triplets to unique directed pairs (parent_gene, target_gene).
    Aggregates score_col and cutoff_col across cond_gene (or any other multiplicity).
    """
    keep_cols = ["parent_gene", "target_gene", score_col, cutoff_col]
    d = df[keep_cols].copy()

    # numeric
    d[score_col] = pd.to_numeric(d[score_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    d[cutoff_col] = pd.to_numeric(d[cutoff_col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    d = d.dropna(subset=["parent_gene", "target_gene", score_col, cutoff_col])

    g = d.groupby(["parent_gene", "target_gene"], sort=False)
    out = g.agg({
        score_col: lambda x: agg_series(x, agg_mode),
        cutoff_col: lambda x: agg_series(x, agg_mode),
    }).reset_index()

    return out


# ============================
# MAIN
# ============================

def main():
    # References
    trrust = load_trrust(TRRUST_FILE)
    core = load_core_tf_target(CORE_FILE)
    ref_tf_map = build_ref_tf_map(trrust, core)
    tf_set = set(ref_tf_map.keys())
    print("[Ref] TFs in reference: {}".format(len(tf_set)))

    # Load predictions
    df = pd.read_csv(PRED_FILE)
    df.columns = df.columns.str.lower()
    print("[Pred] Columns:", df.columns.tolist())
    print("[Pred] Head:\n", df.head(3))


    score_col = SCORE_COL.lower()
    cutoff_col = CUTOFF_COL.lower()
    for c in ["parent_gene", "target_gene", score_col, cutoff_col]:
        if c in df.columns:
            print("[Pred] non-null {}: {}".format(c, int(df[c].notna().sum())))
        else:
            print("[Pred] MISSING COLUMN:", c)

    need = {"parent_gene", "target_gene", score_col, cutoff_col}
    if not need.issubset(set(df.columns)):
        raise ValueError("Pred file must contain columns: {}".format(sorted(list(need))))

    # Dedup triplets -> pairs
    pairs = dedup_triplets_to_pairs(df, score_col=score_col, cutoff_col=cutoff_col, agg_mode=DEDUP_AGG)
    print("[Dedup] Triplets -> unique pairs: {} rows".format(len(pairs)))

    parent = pairs["parent_gene"].astype(str)
    target = pairs["target_gene"].astype(str)
    score = pairs[score_col].astype(float)
    cutoff_vals = pairs[cutoff_col].astype(float)

    base_m = score.notna() & cutoff_vals.notna()
    if DROP_SELF_EDGES:
        base_m = base_m & (parent != target)
    if RESTRICT_PARENT_TO_REF_TFS:
        base_m = base_m & parent.isin(tf_set)

    if not base_m.any():
        raise ValueError("No rows left after base filtering.")

    parent_b = parent[base_m].values
    target_b = target[base_m].values
    score_b = score[base_m].values
    cutoff_b = cutoff_vals[base_m].values

    # Labels once (directed TF->target)
    y_b = np.zeros(len(parent_b), dtype=int)
    for i in range(len(parent_b)):
        tf = parent_b[i]
        tg = target_b[i]
        if tg in ref_tf_map.get(tf, set()):
            y_b[i] = 1

    # Cutoff grid
    if EXPLICIT_CUTOFFS is not None:
        cutoffs = sorted([float(x) for x in EXPLICIT_CUTOFFS])
    else:
        cutoffs = np.linspace(float(MIN_CUTOFF), float(MAX_CUTOFF), int(NUM_CUTOFFS)).tolist()

    print("[Scan] cutoff_col={} | score_col={} | n_cutoffs={}".format(cutoff_col, score_col, len(cutoffs)))

    records = []
    for c in cutoffs:
        m = (cutoff_b >= float(c))
        if not np.any(m):
            records.append({
                "cutoff": float(c),
                "cutoff_col": cutoff_col,
                "score_col": score_col,
                "rows_scored": 0,
                "n_pos": 0,
                "n_neg": 0,
                "auroc": np.nan,
                "aupr": np.nan,
                "implementation": "none",
                "dedup_agg": DEDUP_AGG,
                "restrict_parent_to_ref_tfs": bool(RESTRICT_PARENT_TO_REF_TFS),
                "drop_self_edges": bool(DROP_SELF_EDGES),
            })
            continue

        y = y_b[m]
        s = score_b[m]
        n = int(len(y))
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))

        if len(np.unique(y)) < 2:
            auroc = np.nan
            aupr = np.nan
            impl = "one_class"
        else:
            auroc, aupr, impl = compute_au(y, s)

        records.append({
            "cutoff": float(c),
            "cutoff_col": cutoff_col,
            "score_col": score_col,
            "rows_scored": n,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "auroc": auroc,
            "aupr": aupr,
            "implementation": impl,
            "dedup_agg": DEDUP_AGG,
            "restrict_parent_to_ref_tfs": bool(RESTRICT_PARENT_TO_REF_TFS),
            "drop_self_edges": bool(DROP_SELF_EDGES),
        })

    out = pd.DataFrame(records)
    out.to_csv(OUT_CSV, index=False)
    print("[Done] Wrote:", OUT_CSV)


if __name__ == "__main__":
    main()
