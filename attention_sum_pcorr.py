#!/usr/bin/env python3
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import math
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from transformers import AutoModelForMaskedLM


# ============================================================
# Settings
# ============================================================
model_dir    = "/groups/clairemcwhite/rshaw_workspace/stiffness_prediction/geneformer_model/Geneformer-V2-104M"
dataset_path = "single_cell/tokenized/tok_single_cell.dataset"
map_filename = "/groups/clairemcwhite/rshaw_workspace/stiffness_prediction/full_ESR1_analysis_wCorr.csv"  # optional
vocab_path   = "/groups/clairemcwhite/rshaw_workspace/stiffness_prediction/geneformer_model/geneformer/token_dictionary_gc104M.pkl"

# Set to None to run all cells
MAX_CELLS_TO_PROCESS = None

MAX_TOKENS = 1024

# Cap unique genes per cell
MAX_UNIQUE_GENES_PER_CELL = 256

# Optional: if mapping file is loaded, restrict to mapping_ids
RESTRICT_TO_MAPPING_IDS = False

# Optional: prune when updating stats (0.0 = no pruning)
MIN_ABS_STORE = 0.0

# Numerical stability for pcorr inversion (higher = more stable, slightly more bias)
PCORR_SHRINK = 1e-3

# Output dirs
matrix_dir = "single_cell/mult_sample_matricies"
os.makedirs(matrix_dir, exist_ok=True)

emb_dir = "attention/embeddings"
os.makedirs(emb_dir, exist_ok=True)

# Final stability output
stability_out_csv = os.path.join(matrix_dir, "layer_last_sum_heads_pcorr_stability.csv")


# ============================================================
# PCORR UTILITIES
# ============================================================
def compute_partial_corr(rho, shrink=1e-3):
    """
    Given a correlation matrix rho, return partial correlation matrix pcorr.
    Uses shrinkage + pseudo-inverse for robustness to near-singular rho.
    """
    rho2 = rho.copy()
    np.fill_diagonal(rho2, 1.0)
    rho2 = (1.0 - shrink) * rho2 + shrink * np.eye(rho2.shape[0], dtype=rho2.dtype)

    precision = np.linalg.pinv(rho2)
    d = np.diag(precision)
    denom = np.sqrt(np.outer(d, d)) + 1e-12
    pcorr = -precision / denom
    np.fill_diagonal(pcorr, 1.0)
    return pcorr


# ============================================================
# ONLINE (STREAMING) STATS FOR EDGE STABILITY
# stability(edge) = mean(pcorr) / std(pcorr)
# Uses Welford's algorithm
# ============================================================
class OnlineStats(object):
    __slots__ = ("n", "mean", "M2")
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / float(self.n)
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def finalize(self):
        if self.n < 2:
            return self.n, self.mean, 0.0
        var = self.M2 / float(self.n - 1)
        if var < 0.0:
            var = 0.0
        return self.n, self.mean, math.sqrt(var)


def update_edge_stats_from_pcorr(pcorr, genes, edge_stats, min_abs_store):
    """
    pcorr: (G x G) numpy array for this cell
    genes: list of gene IDs aligned to pcorr rows/cols
    edge_stats: dict[(gene_i, gene_j)] -> OnlineStats
    min_abs_store: optional pruning threshold; 0.0 stores everything
    """
    G = pcorr.shape[0]
    for i in range(G):
        gi = genes[i]
        row = pcorr[i]
        for j in range(i + 1, G):
            x = float(row[j])
            if (min_abs_store > 0.0) and (abs(x) < min_abs_store):
                continue
            gj = genes[j]
            if gi < gj:
                key = (gi, gj)
            else:
                key = (gj, gi)
            edge_stats[key].update(x)


# ============================================================
# Load model
# ============================================================
print("📖 Loading model...")
model = AutoModelForMaskedLM.from_pretrained(model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
print("Device:", device)


# ============================================================
# Load tokenized dataset
# ============================================================
print("📂 Loading tokenized dataset...")
ds = load_from_disk(dataset_path)
num_total_cells = len(ds)
print("Total cells in dataset:", num_total_cells)

if MAX_CELLS_TO_PROCESS is None:
    max_samples = num_total_cells
else:
    max_samples = min(int(MAX_CELLS_TO_PROCESS), num_total_cells)

print("Will process:", max_samples, "cells")


# ============================================================
# Load vocab (token_id → gene_id string)
# ============================================================
with open(vocab_path, "rb") as f:
    vocab = pickle.load(f)
vocab_list = list(vocab)
print("vocab_list (first 10):", vocab_list[:10])


# ============================================================
# Optional mapping (gene_id → pretty label) (kept for compatibility)
# ============================================================
id_to_name = {}
mapping_ids = set()

try:
    mapping_df = pd.read_csv(map_filename)

    mapping_df["gene_id_norm"] = (
        mapping_df["gene_id"]
        .astype(str)
        .str.strip()
        .str.split(".")
        .str[0]
    )

    mapping_df["label"] = mapping_df.get("symbol_interaction")
    if "label" in mapping_df.columns:
        mask_na = mapping_df["label"].isna()
        if "name_interaction" in mapping_df.columns:
            mapping_df.loc[mask_na, "label"] = mapping_df.loc[mask_na, "name_interaction"]
        mask_na = mapping_df["label"].isna()
        mapping_df.loc[mask_na, "label"] = mapping_df.loc[mask_na, "gene_id_norm"]

        id_to_name = dict(zip(mapping_df["gene_id_norm"], mapping_df["label"]))
        mapping_ids = set(mapping_df["gene_id_norm"].tolist())

    print("Total rows in mapping_df:", len(mapping_df))
    print("Total mapping_ids:", len(mapping_ids))
except Exception as e:
    print("⚠️ Could not load mapping file, proceeding without pretty labels.")
    print("Error:", e)
    id_to_name = {}
    mapping_ids = set()

if RESTRICT_TO_MAPPING_IDS and (len(mapping_ids) == 0):
    print("⚠️ RESTRICT_TO_MAPPING_IDS=True but mapping_ids is empty; disabling restriction.")
    RESTRICT_TO_MAPPING_IDS = False


# ============================================================
# Embedding output: single memmap (instead of 25k .npy files)
# ============================================================
hidden_dim = int(model.config.hidden_size)
emb_mm_path = os.path.join(emb_dir, "embeddings_fp16.memmap")
emb_mm = np.memmap(emb_mm_path, dtype=np.float16, mode="w+", shape=(max_samples, hidden_dim))
print("🧠 Embeddings memmap:", emb_mm_path, "shape=", (max_samples, hidden_dim))


# ============================================================
# Aggregators for stability
# ============================================================
edge_stats = defaultdict(OnlineStats)
last_layer_idx_seen = None


# ============================================================
# MAIN LOOP
# Saves:
#   - per-cell sum-heads matrix as .npz (float16 + compressed)
#   - per-cell pcorr matrix as .npz (float16 + compressed)
#   - embeddings to a single memmap (float16)
# Updates:
#   - global edge stability via OnlineStats
# ============================================================
for sample_idx in range(max_samples):
    if sample_idx % 50 == 0:
        print("\n====================")
        print("Processing sample", sample_idx, "/", max_samples)
        print("====================")

    # ---- Get token ids for this sample ----
    token_ids = ds[sample_idx]["input_ids"][:MAX_TOKENS]

    # ---- Map token IDs → gene_ids_clean ----
    gene_ids = [vocab_list[i] for i in token_ids]
    gene_ids_clean = [str(g).strip().split(".")[0] for g in gene_ids]

    # ---- Prepare tensors ----
    input_ids = torch.tensor([token_ids], device=device)
    attention_mask = torch.ones_like(input_ids, device=device)

    # ---- Forward ----
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
        )

    attentions = outputs.attentions        # list of (1, heads, T, T)
    hidden_states = outputs.hidden_states  # list of (1, T, hidden)

    if last_layer_idx_seen is None:
        last_layer_idx_seen = len(attentions) - 1
        print("Using last attention layer index:", last_layer_idx_seen)

    # =========================================
    # 1) EMBEDDING (mean over tokens) -> memmap
    # =========================================
    last_hidden = hidden_states[-1].squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)  # (T, hidden)
    mean_emb = last_hidden.mean(axis=0)  # (hidden,)
    emb_mm[sample_idx, :] = mean_emb.astype(np.float16, copy=False)

    # =========================================
    # 2) SUM HEADS FOR LAST LAYER & SUBSET TO UNIQUE GENES
    # =========================================
    last_layer = attentions[-1].squeeze(0)  # (heads, T, T)
    sum_att_full = last_layer.sum(dim=0).detach().cpu().numpy().astype(np.float32, copy=False)  # (T, T)

    idx_by_gene = {}
    for pos, gid in enumerate(gene_ids_clean):
        if str(gid).startswith("<"):
            continue
        if RESTRICT_TO_MAPPING_IDS and (gid not in mapping_ids):
            continue
        if gid not in idx_by_gene:
            idx_by_gene[gid] = pos
            if len(idx_by_gene) >= MAX_UNIQUE_GENES_PER_CELL:
                break

    if len(idx_by_gene) < 2:
        # Cleanup big tensors and continue
        del outputs, attentions, hidden_states, last_hidden, mean_emb, last_layer, sum_att_full
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        continue

    genes_this_cell = list(idx_by_gene.keys())
    positions = np.array([idx_by_gene[g] for g in genes_this_cell], dtype=np.int64)
    submat = sum_att_full[np.ix_(positions, positions)]  # (G, G)

    # =========================================
    # 3) PCORR COMPUTE (no pandas)
    #    A = submat
    #    C = A A^T
    #    rho = normalized C
    #    pcorr = partial corr of rho
    # =========================================
    A = submat.astype(np.float32, copy=False)
    C = A @ A.T
    d = np.diag(C)
    rho = C / (np.sqrt(np.outer(d, d)) + 1e-12)

    pcorr = compute_partial_corr(rho, shrink=PCORR_SHRINK).astype(np.float32, copy=False)

    # =========================================
    # 4) SAVE per-cell matrices as compressed NPZ (float16)
    # =========================================
    sum_path = os.path.join(
        matrix_dir,
        "sample{0}_layer{1}_sum_heads.npz".format(sample_idx, last_layer_idx_seen)
    )
    pcorr_path = os.path.join(
        matrix_dir,
        "sample{0}_layer{1}_pcorr.npz".format(sample_idx, last_layer_idx_seen)
    )

    np.savez_compressed(
        sum_path,
        genes=np.array(genes_this_cell, dtype=object),
        sum_att=submat.astype(np.float16, copy=False)
    )

    np.savez_compressed(
        pcorr_path,
        genes=np.array(genes_this_cell, dtype=object),
        pcorr=pcorr.astype(np.float16, copy=False)
    )

    # =========================================
    # 5) STREAMING UPDATE OF EDGE STABILITY STATS
    # =========================================
    update_edge_stats_from_pcorr(
        pcorr=pcorr,
        genes=genes_this_cell,
        edge_stats=edge_stats,
        min_abs_store=MIN_ABS_STORE
    )

    # =========================================
    # Cleanup big per-cell objects ASAP
    # =========================================
    del outputs, attentions, hidden_states
    del last_hidden, mean_emb
    del last_layer, sum_att_full
    del submat, A, C, d, rho, pcorr
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


print("\nAll samples processed.")


# ============================================================
# FINAL: write stability table
# ============================================================
print("\nComputing edge stability summary (mean(pcorr)/std(pcorr))...")

EPS = 1e-12
rows = []
for key, st in edge_stats.items():
    gi, gj = key
    n, mean, std = st.finalize()
    if n >= 2:
        stability = mean / (std + EPS)
    else:
        stability = np.nan
    rows.append((
        gi, gj, n,
        float(mean), float(std),
        float(stability) if not np.isnan(stability) else np.nan,
        float(abs(stability)) if not np.isnan(stability) else np.nan
    ))

stability_df = pd.DataFrame(
    rows,
    columns=[
        "gene_i",
        "gene_j",
        "n_cells",
        "pcorr_mean",
        "pcorr_std",
        "pcorr_stability_mean_over_std",
        "abs_pcorr_stability"
    ]
)

stability_df = stability_df.sort_values("abs_pcorr_stability", ascending=False)
stability_df.to_csv(stability_out_csv, index=False)
print("✅ Saved stability summary →", stability_out_csv)

# Flush embeddings memmap
emb_mm.flush()
print("✅ Saved embeddings memmap →", emb_mm_path)

print("Done.")
