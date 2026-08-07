# GNN Drug Ranking — Training Guide

---

## Pre-Training Validation Results ✅

All checks passed:

| Check | Result |
|-------|--------|
| TCGA limma data | ✅ 977/978 genes per cancer (CAST is genuinely absent — zero-filled) |
| Cancer t-stat ranges | BRCA [-25.6, 30.8], LUAD [-19.5, 26.3], PRAD [-12.0, 13.2] |
| Drug signatures | ✅ 160,908 tissue-filtered (U2OS excluded) |
| Z-score coverage | ✅ 100% across all 5 cancer types |
| Graph | ✅ 978 nodes, 20,819 edges |
| Context vectors | ✅ 6-dim one-hot, correct per cell line |

---

## Edge Weight Design — Your Question Answered

You asked whether current weights should be decomposed as:
- **Intra-type**: PPI vs PPI importance
- **Inter-type**: PPI vs Reactome importance

**Short answer: yes, the current design already does both — but there's an issue worth knowing.**

### What the code currently does

```
PPI edge weight    = 1.0 × STRING_score   ∈ [0.70, 0.999]
Pathway edge weight = 0.2 (fixed)          = 0.200
Both (overlap)     = max(PPI_weight, 0.2)  = PPI_weight (since PPI ≥ 0.7 > 0.2)
```

**Intra-type PPI**: Handled naturally — a PPI edge with score 0.999 (EGFR-GRB2) carries `4.99×` more weight than the weakest passing edge (score 0.700). ✅

**Inter-type scale**: Every PPI edge, even the weakest (0.70), already outweighs every Reactome edge (0.20). So the threshold `≥ 0.7` acts as an implicit inter-type boundary. This is **intentional and correct** — PPI edges should dominate.

### The actual consequence in GATv2

In `GATv2Conv`, edge weights are passed as `edge_attr` and *modulate the attention coefficients* — they don't directly scale messages. The attention mechanism learns to up-weight or down-weight neighbors partly based on edge_attr. So:

- The **relative magnitude** of weights matters less than you might think
- What matters is whether weights are **meaningful within each type** (PPI: yes, graded 0.7–1.0) and **distinguishable across types** (PPI vs Reactome: yes, 0.7–1.0 vs 0.2)

### Current design is fine — no change needed

The only alternative worth considering would be normalizing each type separately:

```python
# Option: normalize PPI to [0, 1] within type
ppi_norm = (score - 0.7) / (1.0 - 0.7)  # → [0, 1]
# Then scale: ppi_weight = ppi_norm, pathway_weight = 0.2
```

This would give weak PPI edges (score=0.70 → normalized=0.0) the same weight as Reactome edges. That would actually be *worse* — you'd lose the intra-PPI differentiation. **Keep current design.**

---

## Step 0: Install PyTorch + PyG

```powershell
# Check your CUDA version first (for GPU training)
nvidia-smi

# Option A: CPU only (slower but works everywhere)
myenv\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
myenv\Scripts\pip install torch_geometric
myenv\Scripts\pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.5.0+cpu.html

# Option B: CUDA 12.1 (check nvidia-smi for your version)
myenv\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
myenv\Scripts\pip install torch_geometric
myenv\Scripts\pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
```

> [!TIP]
> Match the PyG version to your PyTorch version. Use `torch.__version__` after installing PyTorch to confirm, then substitute in the URLs above.

Verify:
```powershell
myenv\Scripts\python -c "import torch; from torch_geometric.nn import GATv2Conv; print('OK', torch.__version__)"
```

---

## Step 1: Validate the Pipeline (already done ✅)

```powershell
cd "d:\Drug Repurposing\Drug_Repurpose"
myenv\Scripts\python scripts/validate_gnn_pipeline.py
```

---

## Step 2: Pre-compute Weak Labels (KS Scores)

This is the most time-intensive step: computing KS connectivity scores for every (cancer, signature) pair.

**Estimated load:**
- BRCA: 37,316 signatures × 978 genes = 36.5M operations
- LUAD/LUSC: ~39,718 each
- PRAD: 59,775
- COAD_READ: 24,099
- **Total: ~200,916 signatures**

```powershell
myenv\Scripts\python scripts/precompute_weak_labels.py \
    --disease-sig-dir data/interim/disease_signatures \
    --siginfo data/processed/siginfo_with_annotations.csv \
    --sig-index data/processed/lincs_zscores/sig_index.parquet \
    --chunks-dir data/processed/lincs_zscores/chunks \
    --gene-order data/processed/lincs_zscores/gene_order.parquet \
    --out-dir data/processed/weak_labels \
    --positive-pct 0.12 \
    --negative-pct 0.45
```

> [!NOTE]
> This script needs to be created (see Step 2 implementation below). It caches KS scores to parquet so training doesn't recompute them.

---

## Step 3: Train the Model

```powershell
myenv\Scripts\python scripts/train_drug_ranker.py \
    --graph-dir data/processed/graphs/landmark_gene_graph \
    --weak-labels-dir data/processed/weak_labels \
    --disease-sig-dir data/interim/disease_signatures \
    --siginfo data/processed/siginfo_with_annotations.csv \
    --sig-index data/processed/lincs_zscores/sig_index.parquet \
    --chunks-dir data/processed/lincs_zscores/chunks \
    --gene-order data/processed/lincs_zscores/gene_order.parquet \
    --tcga-mutation-dir data/processed/tcga_mutation \
    --test-cancer LUAD \
    --val-cancer BRCA \
    --hidden-dim 128 \
    --num-heads 4 \
    --num-layers 3 \
    --dropout 0.2 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --batch-size 64 \
    --max-epochs 100 \
    --patience 10 \
    --reversal-lambda 0.3 \
    --checkpoint-dir models/drug_ranker \
    --device cpu
```

> [!IMPORTANT]
> Replace `--device cpu` with `--device cuda` if GPU is available. Training will be **10–30× faster** on GPU.

---

## Step 4: Evaluate

```powershell
myenv\Scripts\python scripts/evaluate_drug_ranker.py \
    --checkpoint models/drug_ranker/best_model.pt \
    --graph-dir data/processed/graphs/landmark_gene_graph \
    --weak-labels-dir data/processed/weak_labels \
    --cancer LUAD \
    --out-dir reports/gnn/evaluation
```

---

## Scripts That Still Need to Be Created

Before training, these two scripts need implementation:

### `scripts/precompute_weak_labels.py`
Reads all drug signatures for each matched cancer, computes KS connectivity scores, thresholds at positive/negative percentiles, and saves to parquet.

**Output structure:**
```
data/processed/weak_labels/
    BRCA/
        positives.parquet   # top 12% by KS score
        negatives.parquet   # bottom 45% by KS score
        all_scores.parquet  # full ranked list
    LUAD/ ...
    LUSC/ ...
    PRAD/ ...
    COAD_READ/ ...
```

### `scripts/train_drug_ranker.py`
Full training CLI that:
1. Loads graph + features
2. Constructs ranking pairs per epoch (balanced across train cancers)
3. Runs the DrugRanker forward pass
4. Computes BPR + softplus reversal loss
5. Validates on val cancer, saves best checkpoint

---

## Data Path Reference

```
data/
├── interim/disease_signatures/{CANCER}/deg_limma.full.pc.parquet  ← cancer t-stats ✅
├── processed/
│   ├── graphs/landmark_gene_graph/                                 ← graph ✅
│   ├── lincs_zscores/
│   │   ├── sig_index.parquet                                       ← ✅
│   │   ├── gene_order.parquet                                      ← ✅
│   │   └── chunks/chunk_NNNN.parquet                               ← drug z-scores ✅
│   ├── siginfo_with_annotations.csv                                ← ✅
│   ├── tcga_mutation/{CANCER}/                                     ← mutation ✅
│   └── weak_labels/{CANCER}/                                       ← ⏳ to generate
models/
└── drug_ranker/                                                    ← ⏳ training output
```

---

## Summary: What to Do Next

1. **Install PyTorch + PyG** (Step 0) — required for model training
2. **Create `precompute_weak_labels.py`** — KS scoring is the bottleneck
3. **Create `train_drug_ranker.py`** — the training CLI
4. **Run weak label pre-computation** (Step 2)
5. **Run training** (Step 3)

The model architecture, all feature modules, losses, trainer, and evaluation metrics are all implemented and validated. The remaining work is the two CLI scripts that wire everything together.
