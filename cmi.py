#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import os

from scipy import io as spio
from scipy import sparse
from collections import defaultdict

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

    mapping: DataFrame with columns ensembl_id,gene_symbol

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

    # Row labels
    print("[Expr] Loading row labels from: {}".format(rows_file))
    rows_df = pd.read_csv(rows_file, sep="\t", header=None, comment="#")
    if rows_df.shape[1] < 1:
        raise ValueError("mtx_rows file must have at least 1 column with gene IDs.")
    rows_df.columns = ["gene_id_raw"] + ["extra_{}".format(i) for i in range(1, rows_df.shape[1])]

    if rows_df.shape[0] != expr.shape[0]:
        print("[WARN] Row labels ({}) != matrix rows ({}). Truncating to min.".format(
            rows_df.shape[0], expr.shape[0]
        ))
        n = min(rows_df.shape[0], expr.shape[0])
        rows_df = rows_df.iloc[:n, :]
        expr = expr[:n, :]

    # Optional col labels
    if cols_file is not None and os.path.exists(cols_file):
        print("[Expr] Loading column labels from: {}".format(cols_file))
        cols_df = pd.read_csv(cols_file, sep="\t", header=None, comment="#")
        if cols_df.shape[0] != expr.shape[1]:
            print("[WARN] Col labels ({}) != matrix cols ({}). Continuing anyway.".format(
                cols_df.shape[0], expr.shape[1]
            ))
    else:
        print("[Expr] No cols_file provided or not found; skipping column labels.")

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
# DISCRETIZATION + MI / CMI
# =============================================================

def discretize_vector(x, n_bins):
    """
    Discretize a 1D numpy array into n_bins using quantiles.
    Returns int16 bin indices.
    """
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


def compute_cmi_discrete(x_bins, y_bins, z_bins):
    """
    Compute conditional mutual information I(X;Y | Z) for discrete variables
    given as integer arrays (same shape).
    Uses natural log (nats).
    """
    x_bins = np.asarray(x_bins, dtype=np.int64)
    y_bins = np.asarray(y_bins, dtype=np.int64)
    z_bins = np.asarray(z_bins, dtype=np.int64)

    if not (x_bins.shape == y_bins.shape == z_bins.shape):
        raise ValueError("x_bins, y_bins, and z_bins must have same shape.")

    n = x_bins.size
    if n == 0:
        return 0.0

    # Shift to start at 0
    x = x_bins - x_bins.min()
    y = y_bins - y_bins.min()
    z = z_bins - z_bins.min()

    nx = int(x.max()) + 1
    ny = int(y.max()) + 1
    nz = int(z.max()) + 1

    # Joint counts over (x, y, z)
    idx_xyz = x + nx * (y + ny * z)
    joint = np.bincount(idx_xyz, minlength=nx * ny * nz).astype(float)
    joint = joint.reshape(nx, ny, nz)
    joint /= float(n)  # p(x,y,z)

    # p(x,z), p(y,z), p(z)
    pxz = joint.sum(axis=1)      # sum over y -> (nx, nz)
    pyz = joint.sum(axis=0)      # sum over x -> (ny, nz)
    pz  = joint.sum(axis=(0, 1)) # (nz,)

    with np.errstate(divide='ignore', invalid='ignore'):
        # p(x,y|z) = p(x,y,z) / p(z)
        # p(x|z)   = p(x,z)   / p(z)
        # p(y|z)   = p(y,z)   / p(z)
        # ratio = p(x,y|z) / (p(x|z) p(y|z))
        #       = p(x,y,z) * p(z) / (p(x,z) p(y,z))

        num   = joint * pz          # broadcast (nx,ny,nz) * (nz,)
        denom = pxz[:, None, :] * pyz[None, :, :]  # (nx,1,nz)*(1,ny,nz)

        ratio = np.where(denom > 0.0, num / denom, 1.0)
        log_ratio = np.where(joint > 0.0, np.log(ratio), 0.0)
        cmi = np.sum(joint * log_ratio)

    return float(cmi)


# =============================================================
# MAIN: CMI ON EDGES FROM MI RESULTS
# =============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute conditional mutual information (CMI) for GRN edges using MI results."
    )

    parser.add_argument("--mi_file", default="mi_results.csv",
                        help="CSV with MI results (target_gene,parent_gene,mi,p_value,...)")
    parser.add_argument("--pval_thresh", type=float, default=0.05,
                        help="Keep edges with p_value <= this threshold.")
    parser.add_argument("--mapping_file", default=None,
                        help="Optional ENSG->symbol CSV (ensembl_id,gene_symbol).")
    parser.add_argument("--gtf_file", default="Homo_sapiens.GRCh38.109.gtf",
                        help="GTF file to build mapping if mapping_file not provided.")
    parser.add_argument("--mapping_out", default="ensembl_to_symbol.csv",
                        help="Where to save mapping built from GTF (if used).")

    parser.add_argument("--mtx_file", default="E-ANND-3.aggregated_filtered_normalised_counts.mtx",
                        help="Matrix Market file with counts (genes x samples).")
    parser.add_argument("--mtx_rows", default="E-ANND-3.aggregated_filtered_normalised_counts.mtx_rows",
                        help="Row labels file (gene IDs).")
    parser.add_argument("--mtx_cols", default="E-ANND-3.aggregated_filtered_normalised_counts.mtx_cols",
                        help="Column labels file (sample IDs).")

    parser.add_argument("--n_bins", type=int, default=5,
                        help="Number of quantile bins for discretization.")
    parser.add_argument("--min_nonzero_cells", type=int, default=10,
                        help="Minimum nonzero samples required for a gene.")

    parser.add_argument("--out_file", default="cmi_results.csv",
                        help="Output CSV for CMI results.")

    args = parser.parse_args()

    # 1. Load MI results and filter by p-value
    print("[MI] Loading MI results from: {}".format(args.mi_file))
    mi_df = pd.read_csv(args.mi_file)

    required_cols = {"target_gene", "parent_gene", "mi"}
    if "p_value" in mi_df.columns:
        required_cols.add("p_value")

    if not required_cols.issubset(set(mi_df.columns)):
        raise ValueError("MI file must contain columns at least: {}".format(required_cols))

    if "p_value" in mi_df.columns:
        mi_df = mi_df[mi_df["p_value"].notna()]
        mi_df = mi_df[mi_df["p_value"] <= args.pval_thresh]

    print("[MI] Kept {} edges with p_value <= {}".format(mi_df.shape[0], args.pval_thresh))

    # Only keep genes that appear as target or parent
    all_genes = set(mi_df["parent_gene"].astype(str).tolist()) | \
                set(mi_df["target_gene"].astype(str).tolist())
    print("[MI] Unique genes in filtered edges: {}".format(len(all_genes)))

    # 2. Mapping
    if args.mapping_file is not None and os.path.exists(args.mapping_file):
        print("[Mapping] Using existing mapping file: {}".format(args.mapping_file))
        mapping = pd.read_csv(args.mapping_file)
        mapping.columns = mapping.columns.str.lower()
        if not {"ensembl_id", "gene_symbol"}.issubset(set(mapping.columns)):
            raise ValueError("Mapping file must have columns: ensembl_id,gene_symbol")
        mapping = mapping.drop_duplicates(subset=["ensembl_id"])
    else:
        if args.gtf_file is None:
            raise ValueError("No mapping_file and no gtf_file provided.")
        mapping = build_mapping_from_gtf(args.gtf_file, args.mapping_out)

    # 3. Expression matrix
    expr_csr, gene_symbols, symbol_to_idx = load_expression_matrix(
        args.mtx_file, args.mtx_rows, args.mtx_cols, mapping
    )
    nonzero_counts = np.array(expr_csr.getnnz(axis=1)).reshape(-1)

    # 4. Precompute discretized expression for all genes we need
    bin_cache = {}
    usable_genes = set()

    print("[Bins] Precomputing bins for genes in MI results...")
    for g in all_genes:
        if g not in symbol_to_idx:
            continue
        row_idx = symbol_to_idx[g]
        nz = int(nonzero_counts[row_idx])
        if nz < args.min_nonzero_cells:
            continue
        vec = expr_csr.getrow(row_idx).toarray().ravel()
        vec = np.log1p(vec)
        bins = discretize_vector(vec, args.n_bins)
        bin_cache[g] = (bins, nz)
        usable_genes.add(g)

    print("[Bins] Binned {} / {} genes with sufficient expression.".format(
        len(usable_genes), len(all_genes)
    ))

    # 5. Group edges by target, build parent lists
    target_to_parents = defaultdict(list)
    for _, row in mi_df.iterrows():
        tgt = str(row["parent_gene"])
        par = str(row["target_gene"])
        target_to_parents[tgt].append(row)

    # 6. Compute CMI for each (target, parent, cond_parent) triple
    results = []
    print("[CMI] Starting CMI computation...")

    for t_idx, (target, rows) in enumerate(target_to_parents.items()):
        if (t_idx + 1) % 100 == 0:
            print("[CMI] Processed {} targets".format(t_idx + 1))

        if target not in bin_cache:
            continue
        y_bins, target_nz = bin_cache[target]

        # list of parent genes (filtered & usable)
        parents = []
        for r in rows:
            pg = str(r["geneB_x"])
            if pg in bin_cache:
                parents.append((pg, r))
        if len(parents) < 2:
            continue  # need at least two parents for conditioning

        # For each ordered pair (parent, cond_parent)
        for i in range(len(parents)):
            geneB_x, row_i = parents[i]
            x_bins, parent_nz = bin_cache[geneB_x]

            for j in range(len(parents)):
                if i == j:
                    continue
                cond_gene, row_j = parents[j]
                z_bins, cond_nz = bin_cache[cond_gene]

                cmi_val = compute_cmi_discrete(x_bins, y_bins, z_bins)

                results.append({
                    "target_gene": target,          # this is geneA_x for this group
                    "parent_gene": parent_gene,         # this is geneB_x for this edge
                    "cond_gene": cond_gene,
                    "mi": float(row_i["mi"]),
                    "cmi": float(cmi_val),
                    "n_samples": int(expr_csr.shape[1]),
                    "target_nonzero_samples": int(target_nz),
                    "parent_nonzero_samples": int(parent_nz),
                    "cond_nonzero_samples": int(cond_nz)
                })

    print("[CMI] Computed CMI for {} triples.".format(len(results)))

    if len(results) == 0:
        print("[CMI] No results to save.")
    else:
        out_df = pd.DataFrame(results)
        out_df.to_csv(args.out_file, index=False)
        print("[CMI] Saved CMI results to: {}".format(args.out_file))


if __name__ == "__main__":
    main()
