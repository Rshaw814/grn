#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import os
from collections import defaultdict

from scipy import io as spio
from scipy import sparse
from scipy.stats import spearmanr

# =============================================================
# GTF PARSER TO BUILD ENSG -> SYMBOL MAPPING
# =============================================================

def build_mapping_from_gtf(gtf_file, mapping_out):
    """
    Parse a GTF file and build ENSG -> gene_symbol mapping.
    Saves to mapping_out as CSV with columns: ensembl_id,gene_symbol
    """
    print("[GTF] Building ENSG -> symbol mapping from: {}".format(gtf_file))
    ensg_ids = []
    symbols = []

    with open(gtf_file, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue

            feature_type = parts[2]
            if feature_type != "gene":
                continue

            attr_field = parts[8]
            attrs = {}
            for item in attr_field.split(";"):
                item = item.strip()
                if item == "":
                    continue
                if " " in item:
                    key, val = item.split(" ", 1)
                    val = val.strip().strip('"')
                    attrs[key] = val

            gene_id = attrs.get("gene_id", None)
            gene_name = attrs.get("gene_name", None)

            if gene_id is None or gene_name is None:
                continue

            gene_id_clean = gene_id.split(".")[0]
            ensg_ids.append(gene_id_clean)
            symbols.append(gene_name)

    mapping_df = pd.DataFrame({
        "ensembl_id": ensg_ids,
        "gene_symbol": symbols
    }).drop_duplicates(subset=["ensembl_id"])

    print("[GTF] Built mapping for {} genes.".format(len(mapping_df)))
    if mapping_out is not None:
        mapping_df.to_csv(mapping_out, index=False)
        print("[GTF] Saved mapping to: {}".format(mapping_out))

    return mapping_df


# =============================================================
# EXPRESSION MATRIX LOADING (mtx + rows + cols)
# =============================================================

def load_expression_matrix(mtx_file, rows_file, cols_file, mapping):
    """
    Load a Matrix Market RNA-seq matrix (genes x samples) plus row/col labels.
    Returns:
        expr_csr: scipy.sparse.csr_matrix (genes x samples)
        gene_symbols: np.array of gene symbols (aligned to rows)
        symbol_to_idx: dict mapping gene_symbol -> row index
    """
    print("[Expr] Loading matrix from: {}".format(mtx_file))
    expr = spio.mmread(mtx_file)
    if not sparse.isspmatrix(expr):
        expr = sparse.csr_matrix(expr)
    else:
        expr = expr.tocsr()

    print("[Expr] Matrix shape (rows x cols): {} x {}".format(expr.shape[0], expr.shape[1]))

    print("[Expr] Loading row labels from: {}".format(rows_file))
    rows_df = pd.read_csv(rows_file, sep="\t", header=None, comment="#")
    if rows_df.shape[1] < 1:
        raise ValueError("mtx_rows file must have at least 1 column with gene IDs.")
    rows_df.columns = ["gene_id_raw"] + ["extra_{}".format(i) for i in range(1, rows_df.shape[1])]

    if rows_df.shape[0] != expr.shape[0]:
        print("[WARN] Number of row labels ({}) != number of matrix rows ({}). "
              "Assuming first min(n) rows align.".format(rows_df.shape[0], expr.shape[0]))
        n = min(rows_df.shape[0], expr.shape[0])
        rows_df = rows_df.iloc[:n, :]
        expr = expr[:n, :]

    if cols_file is not None and os.path.exists(cols_file):
        print("[Expr] Loading column labels from: {}".format(cols_file))
        cols_df = pd.read_csv(cols_file, sep="\t", header=None, comment="#")
        if cols_df.shape[0] != expr.shape[1]:
            print("[WARN] Number of col labels ({}) != number of matrix cols ({}). "
                  "Continuing anyway.".format(cols_df.shape[0], expr.shape[1]))
    else:
        print("[Expr] No cols_file provided or file not found; skipping column labels.")

    rows_df["gene_id_clean"] = rows_df["gene_id_raw"].astype(str).str.split(".").str[0]

    mapping = mapping.drop_duplicates(subset=["ensembl_id"])
    merged = rows_df.merge(
        mapping.rename(columns={"ensembl_id": "gene_id_clean"}),
        on="gene_id_clean",
        how="left"
    )

    merged["final_symbol"] = merged["gene_symbol"]
    merged.loc[merged["final_symbol"].isna(), "final_symbol"] = merged["gene_id_raw"]

    gene_symbols = merged["final_symbol"].values

    symbol_to_idx = {}
    for idx, sym in enumerate(gene_symbols):
        if isinstance(sym, str) and sym not in symbol_to_idx:
            symbol_to_idx[sym] = idx

    print("[Expr] Mapped {} unique gene symbols to matrix rows.".format(len(symbol_to_idx)))
    return expr, gene_symbols, symbol_to_idx


# =============================================================
# MUTUAL INFORMATION (DISCRETE, VIA BINNING)
# =============================================================

def discretize_vector(x, n_bins):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.array([], dtype=np.int16)

    if np.all(x == x[0]):
        return np.zeros_like(x, dtype=np.int16)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    try:
        edges = np.quantile(x, quantiles)
    except Exception:
        edges = np.linspace(np.min(x), np.max(x), n_bins + 1)

    edges = np.unique(edges)
    if edges.size <= 2:
        return np.zeros_like(x, dtype=np.int16)

    bins = np.digitize(x, edges[1:-1], right=False)
    return bins.astype(np.int16)


def compute_mi_discrete(x_bins, y_bins):
    x_bins = np.asarray(x_bins, dtype=np.int64)
    y_bins = np.asarray(y_bins, dtype=np.int64)
    if x_bins.shape != y_bins.shape:
        raise ValueError("x_bins and y_bins must have same shape.")

    n = x_bins.size
    if n == 0:
        return 0.0

    x = x_bins - x_bins.min()
    y = y_bins - y_bins.min()

    nx = int(x.max()) + 1
    ny = int(y.max()) + 1

    joint = np.bincount(x + nx * y, minlength=nx * ny).astype(float)
    joint = joint.reshape(nx, ny)
    joint /= float(n)

    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)

    with np.errstate(divide='ignore', invalid='ignore'):
        denom = px * py
        ratio = np.where(denom > 0, joint / denom, 1.0)
        log_ratio = np.where(joint > 0, np.log(ratio), 0.0)
        mi = np.sum(joint * log_ratio)

    return float(mi)


def mi_with_sequential_permutations(x_bins, y_bins, perm_indices, alpha=0.05, K_step=20, K_total=500):
    mi_obs = compute_mi_discrete(x_bins, y_bins)
    n_perm = perm_indices.shape[0]

    count = 0
    used = 0

    for k in range(min(n_perm, K_total)):
        x_shuf = x_bins[perm_indices[k]]
        mi_k = compute_mi_discrete(x_shuf, y_bins)
        used += 1
        if mi_k >= mi_obs:
            count += 1

        if used % K_step == 0:
            best_possible = (1.0 + count) / (1.0 + K_total)
            if best_possible > alpha:
                break

    p_val = (1.0 + count) / (1.0 + used)
    return mi_obs, p_val


# =============================================================
# BUILD TOP-K NEIGHBORS PER GENE (UPDATED FOR NEW EDGE TABLE)
# =============================================================

def build_topk_neighbors_from_edge_table(pred, top_k, score_col):
    """
    pred: DataFrame with columns:
        geneA, geneB, <score_col>
    Build dict: target_gene -> list of (neighbor_gene, score) length up to top_k.
    Undirected: add both directions.
    """
    target_to_parents = defaultdict(dict)

    for _, row in pred.iterrows():
        ga = row["geneA"]
        gb = row["geneB"]
        score = row[score_col]

        if not isinstance(ga, str) or not isinstance(gb, str):
            continue

        for t, p in ((ga, gb), (gb, ga)):
            parents_dict = target_to_parents[t]
            if p not in parents_dict or score > parents_dict[p]:
                parents_dict[p] = score

    target_to_parents_sorted = {}
    for t, pdict in target_to_parents.items():
        pairs = [(p, s) for p, s in pdict.items() if p != t]
        pairs.sort(key=lambda x: x[1], reverse=True)
        if top_k is not None and top_k > 0:
            pairs = pairs[:top_k]
        target_to_parents_sorted[t] = pairs

    print("[TopK] Built neighbor lists for {} targets.".format(len(target_to_parents_sorted)))
    return target_to_parents_sorted


# =============================================================
# MAIN
# =============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run mutual information tests on top-K edges per gene using scRNA-seq data."
    )

    # NOTE: now this is your *new* edge table (already gene symbols)
    parser.add_argument("--edge_file", default="strong_edges_frequency.csv",
                        help="CSV edge table with geneA,geneB and stability/frequency columns.")

    # Which edge score to use for top-K selection
    parser.add_argument("--edge_score", default="pcorr_t_stability",
                        help="Column in edge_file to rank neighbors by (e.g. pcorr_t_stability, fraction_of_files, files_strong).")

    # Expression matrix files
    parser.add_argument("--mapping_file", default=None,
                        help="Optional pre-built ENSG->symbol CSV (ensembl_id,gene_symbol).")
    parser.add_argument("--gtf_file", default="Homo_sapiens.GRCh38.109.gtf",
                        help="GTF file to build ENSG->symbol mapping if mapping_file is not provided.")
    parser.add_argument("--mapping_out", default="ensembl_to_symbol.csv",
                        help="Where to save mapping built from GTF (if used).")

    parser.add_argument("--mtx_file", default="E-ANND-3.aggregated_filtered_normalised_counts.mtx",
                        help="Matrix Market file with counts (genes x samples).")
    parser.add_argument("--mtx_rows", default="E-ANND-3.aggregated_filtered_normalised_counts.mtx_rows",
                        help="File with row (gene) labels.")
    parser.add_argument("--mtx_cols", default="E-ANND-3.aggregated_filtered_normalised_counts.mtx_cols",
                        help="File with column (sample) labels (not strictly required).")

    # MI configuration
    parser.add_argument("--top_k", type=int, default=15,
                        help="Top K neighbors per gene to keep.")
    parser.add_argument("--n_bins", type=int, default=5,
                        help="Number of quantile bins for MI discretization.")
    parser.add_argument("--n_perm", type=int, default=0,
                        help="Number of permutations for MI p-values (0 = no permutations).")
    parser.add_argument("--min_nonzero_cells", type=int, default=10,
                        help="Minimum number of samples with nonzero expression required for a gene to be used.")

    parser.add_argument("--out_file", default="mi_results.csv",
                        help="Where to save MI results CSV.")

    # Correlation reporting
    parser.add_argument("--corr_out", default="mi_vs_edge_table.csv",
                        help="Where to save merged MI+edge scores for correlation analysis.")

    args = parser.parse_args()

    # 1) Mapping (needed to turn ENSG in matrix rows into symbols)
    if args.mapping_file is not None and os.path.exists(args.mapping_file):
        print("[Mapping] Using existing mapping file: {}".format(args.mapping_file))
        mapping = pd.read_csv(args.mapping_file)
        mapping.columns = mapping.columns.str.lower()
        if not {"ensembl_id", "gene_symbol"}.issubset(set(mapping.columns)):
            raise ValueError("Mapping file must have columns: ensembl_id,gene_symbol")
        mapping = mapping.drop_duplicates(subset=["ensembl_id"])
    else:
        mapping = build_mapping_from_gtf(args.gtf_file, args.mapping_out)

    # 2) Load edge table (already gene symbols)
    print("[Edges] Loading edge table from: {}".format(args.edge_file))
    edges = pd.read_csv(args.edge_file)

    required = {"geneA", "geneB", args.edge_score}
    missing = [c for c in required if c not in edges.columns]
    if len(missing) > 0:
        raise ValueError("edge_file missing required columns: {}. Columns present: {}".format(missing, list(edges.columns)))

    # Keep only valid score rows
    edges = edges.dropna(subset=[args.edge_score])
    # For stability: we typically want magnitude
    if args.edge_score == "pcorr_t_stability":
        edges["_edge_score_abs"] = edges[args.edge_score].abs()
        score_col = "_edge_score_abs"
    else:
        score_col = args.edge_score
        
    # 2.1) Normalize edge gene IDs and map ENSG -> symbol if needed
    edges["geneA"] = edges["geneA"].astype(str).str.strip().str.split(".").str[0]
    edges["geneB"] = edges["geneB"].astype(str).str.strip().str.split(".").str[0]

    # Build ENSG -> symbol dict from mapping
    ensg_to_symbol = dict(zip(mapping["ensembl_id"].astype(str), mapping["gene_symbol"].astype(str)))

    def maybe_map_ensg(x):
        if isinstance(x, str) and x.startswith("ENSG"):
            return ensg_to_symbol.get(x, x)
        return x

    # If edge table is ENSG, convert to symbols so it matches symbol_to_idx
    edges["geneA"] = edges["geneA"].map(maybe_map_ensg)
    edges["geneB"] = edges["geneB"].map(maybe_map_ensg)

    # Quick diagnostics (helps immediately)
    nA_hit = int(edges["geneA"].isin(symbol_to_idx).sum()) if "symbol_to_idx" in locals() else 0
    nB_hit = int(edges["geneB"].isin(symbol_to_idx).sum()) if "symbol_to_idx" in locals() else 0

    # 3) Top-K neighbors from edge table
    target_to_parents = build_topk_neighbors_from_edge_table(edges, args.top_k, score_col)

    # 4) Expression matrix
    expr_csr, gene_symbols, symbol_to_idx = load_expression_matrix(
        args.mtx_file, args.mtx_rows, args.mtx_cols, mapping
    )

    # 4.5) Precompute permutation indices (shared across all edges)
    perm_indices = None
    if args.n_perm is not None and args.n_perm > 0:
        n_samples = expr_csr.shape[1]
        print("[Perm] Precomputing {} permutations for {} samples".format(args.n_perm, n_samples))
        perm_indices = np.zeros((args.n_perm, n_samples), dtype=np.int32)
        for k in range(args.n_perm):
            perm_indices[k] = np.random.permutation(n_samples)

    nonzero_counts = np.array(expr_csr.getnnz(axis=1)).reshape(-1)

    # 5) MI per target-parent
    results = []
    n_edges_total = 0
    n_edges_skipped_expr = 0
    n_edges_evaluated = 0

    # Cache: gene_symbol -> (bins, nonzero_count)
    bin_cache = {}

    all_targets = list(target_to_parents.keys())
    print("[MI] Starting MI computation for {} targets.".format(len(all_targets)))

    for t_idx, target in enumerate(all_targets):
        if (t_idx + 1) % 100 == 0:
            print("[MI] Processed {} / {} targets".format(t_idx + 1, len(all_targets)))

        if target not in symbol_to_idx:
            continue

        target_row = symbol_to_idx[target]
        target_nz = int(nonzero_counts[target_row])
        if target_nz < args.min_nonzero_cells:
            continue

        if target in bin_cache:
            y_bins, _ = bin_cache[target]
        else:
            y_vec = expr_csr.getrow(target_row).toarray().ravel()
            y_vec = np.log1p(y_vec)
            y_bins = discretize_vector(y_vec, args.n_bins)
            bin_cache[target] = (y_bins, target_nz)

        parents = target_to_parents[target]
        for parent, score in parents:
            n_edges_total += 1

            if parent not in symbol_to_idx:
                n_edges_skipped_expr += 1
                continue

            parent_row = symbol_to_idx[parent]
            parent_nz = int(nonzero_counts[parent_row])
            if parent_nz < args.min_nonzero_cells:
                n_edges_skipped_expr += 1
                continue

            if parent in bin_cache:
                x_bins, _ = bin_cache[parent]
            else:
                x_vec = expr_csr.getrow(parent_row).toarray().ravel()
                x_vec = np.log1p(x_vec)
                x_bins = discretize_vector(x_vec, args.n_bins)
                bin_cache[parent] = (x_bins, parent_nz)

            if perm_indices is not None:
                mi_val, p_val = mi_with_sequential_permutations(x_bins, y_bins, perm_indices)
            else:
                mi_val = compute_mi_discrete(x_bins, y_bins)
                p_val = None

            n_edges_evaluated += 1

            results.append({
                "target_gene": target,
                "parent_gene": parent,
                "edge_score_used": float(score),
                "mi": mi_val,
                "p_value": p_val,
                "n_samples": int(expr_csr.shape[1]),
                "target_nonzero_samples": target_nz,
                "parent_nonzero_samples": parent_nz
            })

    print("[MI] Total candidate edges (top-K): {}".format(n_edges_total))
    print("[MI] Skipped edges due to missing/low expression: {}".format(n_edges_skipped_expr))
    print("[MI] Evaluated edges: {}".format(n_edges_evaluated))

    if len(results) == 0:
        print("[MI] No MI results computed; nothing to save.")
        return

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out_file, index=False)
    print("[MI] Saved MI results to: {}".format(args.out_file))

    # =========================================================
    # NEW: Correlate MI with stability (and other edge metrics)
    # =========================================================
    # Merge MI results back onto edge table (undirected match)
    a = out_df[["target_gene", "parent_gene", "mi"]].copy()
    a["geneA"] = a[["target_gene", "parent_gene"]].min(axis=1)
    a["geneB"] = a[["target_gene", "parent_gene"]].max(axis=1)
    a = a.drop(columns=["target_gene", "parent_gene"])

    e = edges.copy()
    e["geneA_m"] = e[["geneA", "geneB"]].min(axis=1)
    e["geneB_m"] = e[["geneA", "geneB"]].max(axis=1)

    merged = a.merge(
        e,
        left_on=["geneA", "geneB"],
        right_on=["geneA_m", "geneB_m"],
        how="left"
    )

    merged.to_csv(args.corr_out, index=False)
    print("[Corr] Saved merged MI + edge metrics to: {}".format(args.corr_out))

    # Compute correlation MI vs stability if available
    if "pcorr_t_stability" in merged.columns:
        tmp = merged[["mi", "pcorr_t_stability"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(tmp) >= 10:
            pear = np.corrcoef(tmp["mi"].values, np.abs(tmp["pcorr_t_stability"].values))[0, 1]
            spear = spearmanr(tmp["mi"].values, np.abs(tmp["pcorr_t_stability"].values)).correlation
            print("[Corr] MI vs |pcorr_t_stability|: Pearson r = {:.4f}, Spearman rho = {:.4f} (n={})".format(
                pear, spear, len(tmp)
            ))
        else:
            print("[Corr] Not enough overlap to compute MI vs stability correlation.")
    else:
        print("[Corr] pcorr_t_stability column not present; cannot compute MI correlation.")


if __name__ == "__main__":
    main()
