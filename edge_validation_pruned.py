#!/usr/bin/env python3
import argparse
import os
from collections import defaultdict
import numpy as np
import pandas as pd

def norm_edge(a, b):
    a = str(a); b = str(b)
    return (a, b) if a <= b else (b, a)

# ---------------- STRING helpers ----------------

def load_string_info(info_path):
    df = pd.read_csv(info_path, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    cn = {c.lower(): c for c in df.columns}

    pref_col = None
    for cand in ["preferred_name", "preferredname", "preferred name", "gene_name", "gene", "symbol"]:
        if cand in cn:
            pref_col = cn[cand]; break
    if pref_col is None:
        raise ValueError(f"Could not find preferred_name-like column in {info_path}. cols={list(df.columns)}")

    pid_col = None
    for cand in ["#string_protein_id", "string_protein_id", "protein_external_id", "protein_id", "proteinid", "protein"]:
        if cand in cn:
            pid_col = cn[cand]; break
    if pid_col is None:
        raise ValueError(f"Could not find string protein id column in {info_path}. cols={list(df.columns)}")

    pref2pid = {}
    for pref, pid in zip(df[pref_col].astype(str), df[pid_col].astype(str)):
        if pref and pref != "nan" and pid and pid != "nan":
            if pref not in pref2pid:
                pref2pid[pref] = pid
    return pref2pid

def load_string_aliases(aliases_path):
    df = pd.read_csv(aliases_path, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    cn = {c.lower(): c for c in df.columns}
    pid_col = cn.get("#string_protein_id") or cn.get("string_protein_id")
    alias_col = cn.get("alias")
    if pid_col is None or alias_col is None:
        raise ValueError(f"Aliases file missing required columns. cols={list(df.columns)}")

    alias2pid = {}
    for pid, alias in zip(df[pid_col].astype(str), df[alias_col].astype(str)):
        if alias and alias != "nan" and pid and pid != "nan":
            if alias not in alias2pid:
                alias2pid[alias] = pid
    return alias2pid

def stream_string_links_as_pid_edges(links_path, pid_set, min_score=500, score_col="combined_score"):
    kept = set()
    with open(links_path, "r") as f:
        header = f.readline().strip().split()
        sc_idx = header.index(score_col) if score_col in header else (len(header) - 1)

        for line in f:
            parts = line.strip().split()
            if len(parts) <= sc_idx:
                continue
            p1, p2 = parts[0], parts[1]
            if p1 not in pid_set or p2 not in pid_set:
                continue
            try:
                sc = int(parts[sc_idx])
            except:
                continue
            if sc < min_score:
                continue
            kept.add((p1, p2) if p1 <= p2 else (p2, p1))
    return kept

# ---------------- metrics ----------------

def auroc_aupr(y_true, y_score):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan, np.nan

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)

    uniq, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    for u_i, c in enumerate(counts):
        if c > 1:
            idx = np.where(inv == u_i)[0]
            ranks[idx] = np.mean(ranks[idx])

    sum_ranks_pos = np.sum(ranks[y_true == 1])
    n_pos = len(pos); n_neg = len(neg)
    U = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    auroc = U / (n_pos * n_neg)

    idx = np.argsort(-y_score)
    y = y_true[idx]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp[-1], 1)

    aupr = 0.0
    prev_r = 0.0
    for p, r in zip(precision, recall):
        aupr += p * (r - prev_r)
        prev_r = r

    return float(auroc), float(aupr)
    
def degree_correlation(edges, string_gene_edges, all_nodes):
    """
    Per-node degree correlation between predicted graph and STRING graph
    over a FIXED node set.
    """
    deg_pred = defaultdict(int)
    for a, b in edges:
        deg_pred[a] += 1
        deg_pred[b] += 1

    deg_str = defaultdict(int)
    for a, b in string_gene_edges:
        if a in all_nodes and b in all_nodes:
            deg_str[a] += 1
            deg_str[b] += 1

    nodes = sorted(all_nodes)
    x = np.array([deg_pred.get(n, 0) for n in nodes], dtype=float)
    y = np.array([deg_str.get(n, 0) for n in nodes], dtype=float)

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])

# ---------------- scoring + collapse ----------------

def score_row(r, score_mode):
    if score_mode == "cmi":
        return float(r["cmi"])
    if score_mode == "mi_minus_cmi":
        return float(r["mi"] - r["cmi"])
    # ratio
    v = r.get("cmi_ratio", np.nan)
    return float(v) if pd.notnull(v) else 0.0

def collapse_edges(pruned_df, score_mode):
    """
    pruned_df has target_gene,parent_gene,mi,cmi,cmi_ratio
    Returns edges(list[tuple]), scores(np.array)
    """
    edge_best = {}
    for _, r in pruned_df.iterrows():
        e = norm_edge(r["target_gene"], r["parent_gene"])
        sc = score_row(r, score_mode)
        if (e not in edge_best) or (sc > edge_best[e]):
            edge_best[e] = sc
    edges = list(edge_best.keys())
    scores = np.array([edge_best[e] for e in edges], dtype=float)
    return edges, scores

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--mi_cmi_table", default="cmi_results.csv",
                    help="CSV with target_gene,parent_gene,mi,cmi,...")
    ap.add_argument("--out_dir", default="pruned_validation_out")
    ap.add_argument("--out_prefix", default="pruned")

    # sweep controls
    ap.add_argument("--cmi_cutoffs", nargs="+", type=float, default=None,
                    help="Explicit list of CMI cutoffs to sweep (e.g. 0.0 0.05 0.1 0.15 0.2).")
    ap.add_argument("--cmi_cutoff_grid", nargs=3, type=float, default=None,
                    help="Grid as: min max n_points (e.g. 0.0 0.5 21). Ignored if --cmi_cutoffs provided.")

    # optional ratio criterion (kept fixed during sweep)
    ap.add_argument("--min_ratio", type=float, default=None,
                    help="If set, also keep edges with cmi/mi >= min_ratio (OR with cmi cutoff).")

    # scoring for AUROC/AUPR
    ap.add_argument("--score_mode", choices=["cmi", "mi_minus_cmi", "ratio"], default="cmi")

    # write per-cutoff predicted edges
    ap.add_argument("--write_edges_per_cutoff", action="store_true",
                    help="If set, writes predicted edge table for each cutoff (can be many files).")
    ap.add_argument("--max_edges_per_cutoff", type=int, default=0,
                    help="If >0, only write top-N edges per cutoff when --write_edges_per_cutoff is on.")

    # STRING option A
    ap.add_argument("--string_gene_edges", default=None,
                    help="CSV with geneA,geneB for STRING edges (already mapped).")

    # STRING option B
    ap.add_argument("--string_info", default="9606.protein.info.v12.0.txt")
    ap.add_argument("--string_aliases", default="9606.protein.aliases.v12.0.txt")
    ap.add_argument("--string_links", default="9606.protein.links.v12.0.txt")
    ap.add_argument("--string_min_score", type=int, default=500)
    ap.add_argument("--string_score_col", default="combined_score")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.mi_cmi_table)
    required = {"target_gene", "parent_gene", "mi", "cmi"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"mi/cmi table missing columns: {sorted(missing)}")

    df["mi"] = pd.to_numeric(df["mi"], errors="coerce")
    df["cmi"] = pd.to_numeric(df["cmi"], errors="coerce")
    df["cmi_ratio"] = df["cmi"] / df["mi"].replace(0, np.nan)
    
    
    # --- Collapse triplets to unique undirected edges once (keeps max evidence per edge) ---
    df["geneA"] = df["target_gene"].astype(str)
    df["geneB"] = df["parent_gene"].astype(str)
    
    # undirected key
    a = df["geneA"].values
    b = df["geneB"].values
    df["u"] = np.where(a <= b, a, b)
    df["v"] = np.where(a <= b, b, a)
    
    # aggregate: "edge passes if any triplet passes" + "score uses best triplet"
    df_edge = (
        df.groupby(["u", "v"], as_index=False)
          .agg({
              "mi": "max",
              "cmi": "max",
              "cmi_ratio": "max",
          })
    )
    
    # replace df with the edge-collapsed view
    df = df_edge.rename(columns={"u": "geneA", "v": "geneB"}).copy()
    
    # for compatibility with the rest of your code:
    df = df.rename(columns={"geneA": "target_gene", "geneB": "parent_gene"})

    # ---- Build cutoff list
    if args.cmi_cutoffs is not None and len(args.cmi_cutoffs) > 0:
        cutoffs = sorted([float(x) for x in args.cmi_cutoffs])
    elif args.cmi_cutoff_grid is not None:
        cmin, cmax, npts = args.cmi_cutoff_grid
        npts = int(npts)
        if npts < 2:
            cutoffs = [float(cmin)]
        else:
            cutoffs = np.linspace(float(cmin), float(cmax), npts).tolist()
    else:
        # sensible default sweep
        cutoffs = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3]

    # ---- Load STRING gene edges once
    if args.string_gene_edges is not None:
        s = pd.read_csv(args.string_gene_edges)
        s.columns = [c.strip() for c in s.columns]
        if "gene1" in s.columns and "gene2" in s.columns:
            s = s.rename(columns={"gene1": "geneA", "gene2": "geneB"})
        if not ("geneA" in s.columns and "geneB" in s.columns):
            raise ValueError(f"--string_gene_edges must have geneA,geneB (or gene1,gene2). cols={list(s.columns)}")
        string_gene_edges = set(norm_edge(a, b) for a, b in zip(s["geneA"], s["geneB"]))
        print(f"[STRING] loaded mapped gene edges: {len(string_gene_edges)}")
        mapping_mode = "pre_mapped"
    else:
        if not (args.string_info and args.string_aliases and args.string_links):
            raise ValueError("Provide either --string_gene_edges OR all of --string_info --string_aliases --string_links")
        pref2pid = load_string_info(args.string_info)
        alias2pid = load_string_aliases(args.string_aliases)
        mapping_mode = "raw_string_files"

        # We will map STRING pid edges -> gene edges PER SWEEP using predicted gene list;
        # but that would be expensive. Instead we build gene2pid on the union of all genes in df.
        all_genes = sorted(set(df["target_gene"].astype(str)) | set(df["parent_gene"].astype(str)))

        def resolve(g):
            g0 = str(g).split(".")[0]
            return pref2pid.get(g0) or pref2pid.get(g0.upper()) or \
                   alias2pid.get(g0) or alias2pid.get(g0.upper()) or alias2pid.get(g0.lower())

        gene2pid_all = {g: resolve(g) for g in all_genes}
        mapped_all = [g for g in all_genes if gene2pid_all[g] is not None]
        pid_set_all = set(gene2pid_all[g] for g in mapped_all)

        print(f"[STRING] (union) genes_total={len(all_genes)} mapped_genes={len(mapped_all)}")

        string_pid_edges = stream_string_links_as_pid_edges(
            args.string_links,
            pid_set=pid_set_all,
            min_score=args.string_min_score,
            score_col=args.string_score_col
        )
        print(f"[STRING] (union) mapped_edges_kept={len(string_pid_edges)} at min_score={args.string_min_score}")

        # pid->gene representative (one gene per pid)
        pid2gene = {}
        for g in mapped_all:
            pid2gene[gene2pid_all[g]] = g

        string_gene_edges = set()
        for p1, p2 in string_pid_edges:
            g1 = pid2gene.get(p1); g2 = pid2gene.get(p2)
            if g1 is None or g2 is None:
                continue
            string_gene_edges.add(norm_edge(g1, g2))

    # ---- Sweep
    rows = []
    for ccut in cutoffs:
        ccut = float(ccut)

        keep = (df["mi"] >= ccut)
        if args.min_ratio is not None:
            keep = keep | (df["cmi_ratio"] >= float(args.min_ratio))

        pruned = df[keep].copy()
        n_tests = pruned.shape[0]

        edges, scores = collapse_edges(pruned, args.score_mode)
        n_edges = len(edges)
        if n_edges == 0:
            rows.append({
                "cmi_cutoff": ccut,
                "min_ratio": args.min_ratio if args.min_ratio is not None else np.nan,
                "score_mode": args.score_mode,
                "n_tests_kept": int(n_tests),
                "n_edges_pred": 0,
                "n_tp": 0,
                "tp_rate": np.nan,
                "AUROC": np.nan,
                "AUPR": np.nan,
            })
            continue

        y_true = np.array([1 if e in string_gene_edges else 0 for e in edges], dtype=int)
        auroc, aupr = auroc_aupr(y_true, scores)
        deg_corr = degree_correlation(edges, string_gene_edges, all_genes)

        rows.append({
            "cmi_cutoff": ccut,
            "min_ratio": args.min_ratio if args.min_ratio is not None else np.nan,
            "score_mode": args.score_mode,
            "n_tests_kept": int(n_tests),
            "n_edges_pred": int(n_edges),
            "n_tp": int(y_true.sum()),
            "tp_rate": float(y_true.mean()),
            "AUROC": auroc,
            "AUPR": aupr,
            "deg_corr": deg_corr,
        })

        if args.write_edges_per_cutoff:
            pred_df = pd.DataFrame({
                "geneA": [e[0] for e in edges],
                "geneB": [e[1] for e in edges],
                "score": scores
            }).sort_values("score", ascending=False)

            if args.max_edges_per_cutoff and int(args.max_edges_per_cutoff) > 0:
                pred_df = pred_df.head(int(args.max_edges_per_cutoff)).copy()

            outp = os.path.join(args.out_dir, f"{args.out_prefix}_predicted_edges__cmi{ccut:.6g}.csv")
            pred_df.to_csv(outp, index=False)

        print(f"[SWEEP] cmi>={ccut:.6g} | tests_kept={n_tests} | edges={n_edges} | tp={int(y_true.sum())} | AUROC={auroc} | AUPR={aupr} | deg_corr={deg_corr}")

    metrics_df = pd.DataFrame(rows).sort_values("cmi_cutoff")
    out_path = os.path.join(args.out_dir, f"{args.out_prefix}_STRING_metrics_by_cmi.csv")
    metrics_df.to_csv(out_path, index=False)
    print(f"[WRITE] {out_path}")
    print(metrics_df.to_string(index=False))

if __name__ == "__main__":
    main()
