# Attention-Derived Gene Regulatory Network Inference

This repository implements a research-grade pipeline for inferring and validating
gene–gene relationships from single-cell RNA-seq data using attention weights from
a pretrained Geneformer model.

The central idea is that transformer attention encodes *context-dependent conditional
dependencies* between genes. By aggregating attention-derived partial correlations
across many cells, we obtain a stable, data-driven gene interaction network that can
be validated using expression-based information theory and external regulatory
resources.

This repository contains the **core analysis pipeline only** (no large datasets,
models, or outputs).

---

## Conceptual Overview

The pipeline consists of five main stages:

1. **Attention → Partial Correlation (PCORR)**
   - Extract last-layer attention from Geneformer
   - Sum attention across heads
   - Compute partial correlations between genes within each cell
   - Aggregate edge stability across cells using streaming statistics

2. **Expression-Based Validation (MI)**
   - Validate attention-derived edges using mutual information (MI)
     computed from scRNA-seq expression data
   - Restrict tests to top-K neighbors per gene to control complexity

3. **Conditional Independence (CMI)**
   - Use conditional mutual information (CMI) to identify and prune
     indirect edges

4. **External Benchmarking**
   - Compare inferred edges to STRING protein–protein interaction networks

5. **Directed TF–Target Enrichment**
   - Evaluate directed regulatory edges against curated TF–target databases
     (TRRUST and CORE)
   - Perform degree-corrected enrichment analyses

Each stage produces interpretable intermediate outputs that can be inspected
independently.

---

## Pipeline Scripts (Execution Order)

### 1. `attention_sum_pcorr.py`
**Purpose:**  
Extract attention-derived gene–gene relationships and compute edge stability.

**What it does:**
- Loads a pretrained Geneformer model
- Extracts last-layer attention matrices per cell
- Sums attention across heads
- Subsets to unique genes per cell
- Computes partial correlation matrices from attention
- Aggregates edge statistics across cells using online (streaming) estimators

**Key outputs:**
- `layer_last_sum_heads_pcorr_stability.csv`  
  A global edge table containing:
  - gene pairs
  - number of cells observed
  - mean partial correlation
  - standard deviation
  - stability score (mean / std)

Optional intermediate outputs (often disabled for large runs):
- Per-cell attention matrices
- Per-cell partial correlation matrices
- Per-cell embeddings (memmap)

---

### 2. `information_theory.py`
**Purpose:**  
Validate attention-derived edges using expression-based mutual information (MI).

**What it does:**
- Loads scRNA-seq expression data (Matrix Market format)
- Selects top-K candidate edges per gene from the attention-derived edge table
- Computes discretized mutual information (MI) for each candidate edge
- Optionally performs permutation testing
- Produces merged tables for downstream correlation analysis

**Key outputs:**
- `mi_results.csv`  
  MI values per (target, parent) gene pair
- `mi_vs_edge_table.csv`  
  MI values merged with attention-derived edge scores

---

### 3. `cmi.py`
**Purpose:**  
Prune indirect interactions using conditional mutual information (CMI).

**What it does:**
- Loads MI results
- For each gene, evaluates whether a parent–target relationship remains
  after conditioning on alternative parents
- Computes discrete CMI:  
  \[
  I(X;Y \mid Z)
  \]
- Produces triplet-level evidence for direct vs indirect edges

**Key outputs:**
- `cmi_results.csv`  
  Contains MI, CMI, and related statistics for candidate edges

---

### 4. `edge_validation_pruned.py`
**Purpose:**  
Benchmark pruned networks against STRING.

**What it does:**
- Sweeps CMI (or MI-derived) cutoffs
- Collapses triplet-level evidence to edge-level scores
- Evaluates predicted networks against STRING PPI edges
- Computes:
  - AUROC
  - AUPR
  - Degree correlation
- Optionally writes predicted edge lists per cutoff

**Key outputs:**
- CSV files containing performance metrics across cutoff sweeps
- Optional predicted edge tables

---

### 5. `edge_validate_pruned_tf.py`
**Purpose:**  
Evaluate directed TF–target predictions using curated regulatory databases.

**What it does:**
- Uses TRRUST and CORE TF–target references
- Deduplicates multiple evidences per TF–target pair
- Scans performance as a function of MI / CMI cutoffs
- Computes AUROC and AUPR for directed regulatory inference

**Key outputs:**
- AUROC/AUPR vs cutoff tables for TF–target prediction

---

### 6. `edges_validate_pruned_enrichment_tf.py`
**Purpose:**  
Perform degree-corrected enrichment analysis of TF–target predictions.

**What it does:**
- Evaluates top-K predicted targets per TF
- Computes:
  - Precision, recall, enrichment over baseline
  - Per-TF coverage statistics
- Uses degree-preserving null models to assess significance
- Performs global permutation tests that correct for target hub bias

**Key outputs:**
- Global enrichment summaries
- Per-TF enrichment metrics
- Degree-corrected permutation test results

---

## Dependencies

This pipeline requires a scientific Python environment including:

- Python ≥ 3.9
- PyTorch
- HuggingFace Transformers
- NumPy, Pandas, SciPy
- scikit-learn (optional; NumPy fallbacks included)

Large assets (Geneformer model, scRNA-seq matrices, STRING, GTF files)
are not included in this repository.

---

## Design Philosophy

- **Interpretability first**: each stage produces explicit, inspectable outputs
- **Streaming statistics**: enables scaling to very large cell counts
- **Multiple validation axes**:
  - Expression-based (MI / CMI)
  - Physical interactions (STRING)
  - Regulatory ground truth (TRRUST / CORE)
- **Degree-aware evaluation** to avoid hub-driven artifacts

---

## Status

This repository reflects the **current working research pipeline**.
The code prioritizes correctness and transparency over packaging.
Future work may include:
- Modularization into a Python package
- Unified configuration files
- Lightweight example datasets
- End-to-end runners

---

## Citation / Contact

If you use or adapt this code, please cite appropriately or contact the author.
