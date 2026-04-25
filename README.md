# Early Rumour Detection via Bidirectional Cascade Graph Learning

> **Can we detect misinformation before fact-checkers can respond?**

**Macro-F1 = 0.813 at 30 minutes | Gap vs Full Cascade = 0.003 | Burstiness Importance = 44%**

---

## Overview

This project investigates whether the first 30 minutes of a Twitter cascade contain enough structural and temporal signal for reliable automated rumour detection. We adapt the BiGCN architecture (Bian et al., 2020) for the PHEME dataset, replacing TF-IDF node features with BERT embeddings augmented with structural and temporal features, and introduce a time-based early detection evaluation framework.

**Key finding:** BiGCN achieves Macro-F1 = 0.813 at 30 minutes vs. 0.810 at full cascade observation — a gap of just 0.003. The veracity signal is overwhelmingly front-loaded in a cascade's earliest phase.

---

## Repository Structure

```
misinformation-detection/
│
├── data/
│   ├── pheme/                          # Raw PHEME JSON files (9 events)
│   ├── twitter15/                      # Twitter15 dataset
│   │   ├── tree/                       # One .txt file per cascade
│   │   ├── label.txt                   # label:tweet_id format
│   │   └── source_tweets.txt           # tweet_id \t text format
│   └── twitter16/                      # Twitter16 dataset (same structure)
│
├── src/
│   ├── load_cascade.py                 # PHEME JSON → NetworkX DiGraph parser
│   ├── features.py                     # Structural + temporal feature extraction
│   └── build_graph.py                  # NetworkX → PyG Data object builder
│
├── outputs/
│   ├── cascade_features.csv            # 5,802 cascades × structural/temporal features
│   ├── texts_and_labels.json           # Source tweet texts + binary labels
│   ├── all_node_embeddings.npy         # (103,572 × 768) BERT embeddings — all nodes
│   ├── node_id_to_idx.json             # tweet_id → row index in embeddings array
│   ├── source_tweet_embeddings.npy     # (5,802 × 768) root node embeddings only
│   ├── all_cascades.pkl                # Serialised NetworkX graphs
│   ├── rf_structural.pkl               # Trained RF — structural features
│   ├── rf_temporal.pkl                 # Trained RF — temporal features
│   ├── rf_all.pkl                      # Trained RF — all features
│   ├── rf_early_detection_scores.csv   # RF Macro-F1 at 30/60/120/full
│   ├── bert_preds.npy                  # BERT OOF predictions
│   ├── bert_probs.npy                  # BERT OOF probabilities
│   ├── bert_results.csv                # BERT macro_f1, roc_auc
│   ├── bigcn_preds.npy                 # BiGCN full cascade predictions
│   ├── bigcn_probs.npy                 # BiGCN full cascade probabilities
│   ├── bigcn_valid_idx.npy             # Indices of valid cascades (non-degenerate)
│   ├── bigcn_early_detection_scores.csv # BiGCN Macro-F1 at 30/60/120/full
│   ├── error_analysis.csv              # 15 misclassified examples with features
│   ├── early_detection_curve_final.png # Main result figure
│   ├── error_analysis_plots.png        # Error analysis visualisations
│   └── twitter/                        # Twitter15/16 experiment outputs
│       ├── t15_source_embeddings.npy
│       ├── t16_source_embeddings.npy
│       ├── t15_embedding_ids.json
│       ├── t16_embedding_ids.json
│       ├── Twitter15_embeddings_viz.png
│       ├── Twitter16_embeddings_viz.png
│       ├── twitter_results.csv
│       └── twitter_early_detection_curves.png
│
├── week1.ipynb                         # EDA, cascade parsing, RF baseline, BERT fine-tuning
├── bigcn.ipynb                   # BiGCN on PHEME — main experiments
├── week3_twitter.ipynb                 # BiGCN on Twitter15/16 — BERT features
├── twitter_tfidf.ipynb           # BiGCN on Twitter15/16 — TF-IDF features
├── error_analysis.ipynb          # Error analysis — 15 misclassified examples
└── README.md
```

---

## Datasets

### Primary — PHEME
- **5,802** Twitter conversation threads across 5 real-world events (Charlie Hebdo, Ferguson, Germanwings Crash, Ottawa Shooting, Sydney Siege)
- **Binary labels:** Rumour (34%) vs Non-Rumour (66%)
- **Full reply structure with timestamps** — enables cascade reconstruction and time-based truncation
- Source: [PHEME Dataset](https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078)

| Event | Total | Rumours | Non-Rumours | Avg Nodes |
|---|---|---|---|---|
| Charlie Hebdo | 2,079 | 458 | 1,621 | 18.2 |
| Ferguson | 1,143 | 284 | 859 | 21.4 |
| Germanwings Crash | 469 | 238 | 231 | 16.8 |
| Ottawa Shooting | 470 | 470 | 0 | 15.3 |
| Sydney Siege | 522 | 522 | 0 | 17.9 |
| **Total** | **5,802** | **1,972** | **3,830** | **17.9** |

### Secondary — Twitter15/16
- **Twitter15:** 1,490 cascades — 4-class (false/true/unverified/non-rumour), perfectly balanced (~375 per class)
- **Twitter16:** 818 cascades — 4-class, perfectly balanced (~205 per class)
- **API constraint:** Only source tweet text available — reply texts inaccessible under 2023+ Twitter API restrictions
- Source: [ACL 2017 Dataset](https://www.dropbox.com/s/46r50ctrfa0ur1o/rumdect.zip)

---

## Methodology

### Node Feature Representation (771-d)

Each tweet node in the cascade graph receives a 771-dimensional feature vector:

```
x_v = [e_v || d_v^norm || δ_v^norm || t_v^norm] ∈ ℝ^771
```

| Component | Dimensions | Description |
|---|---|---|
| BERT [CLS] embedding | 768 | `bert-base-uncased`, max 128 tokens |
| Depth (normalised) | 1 | BFS depth from root / max depth |
| Degree (normalised) | 1 | Node degree / max degree in cascade |
| Temporal arrival (normalised) | 1 | t_min / max t_min in cascade |

**PHEME:** Every node has unique BERT embedding from its own tweet text (103,572 unique embeddings).  
**Twitter15/16:** All nodes share the source tweet embedding — reply texts inaccessible under API restrictions.

### BiGCN Architecture

The model runs two parallel GCN passes on the cascade tree:

- **Top-Down (TD) GCN:** Edges follow parent → child direction — captures how the original claim propagates downward
- **Bottom-Up (BU) GCN:** Edges reversed to child → parent — captures crowd reaction bubbling back toward the source

Each GCN branch has 2 layers with BatchNorm and feature dropout (p=0.3).

**PHEME-specific adaptations vs original BiGCN:**

| Component | Original BiGCN (Bian 2020) | Our PHEME Adaptation |
|---|---|---|
| Node features | 5,000-dim TF-IDF | 771-dim BERT + structural + temporal |
| Root tiling | After each GCN layer | Not used (shallow cascades) |
| Graph readout | scatter_mean (all nodes) | Root-node readout |
| Regularisation | DropEdge (p=0.2) | Feature dropout (p=0.3) + BatchNorm |
| Loss | NLL loss | Weighted cross-entropy (N_neg/N_pos) |
| LR schedule | None | StepLR (γ=0.5, step=20 epochs) |
| Grad clipping | Not specified | max_norm=2.0 |
| Classes | 4 | 2 (binary) |

**Rationale for root-node readout:** PHEME cascades average 17.9 nodes with depth 2-3. Two GCN layers provide a 2-hop receptive field — sufficient to propagate all reply information into the root node for shallow PHEME cascades. Mean pooling over 2-3 nodes adds minimal signal.

### Early Detection Framework

Cascades are truncated at four time thresholds applied at **graph construction time** (no future information leakage):

- **T = 30 minutes** — critical early window
- **T = 60 minutes**
- **T = 120 minutes**
- **T = full** — complete cascade observation

All models are trained and evaluated independently at each threshold under 5-fold stratified cross-validation.

---

## Results

### PHEME — Primary Results

| Model | Input | 30 min | 60 min | 120 min | Full | ROC-AUC |
|---|---|---|---|---|---|---|
| RF (structural) | Depth, branching, size | 0.514 | 0.514 | 0.514 | 0.514 | 0.523 |
| RF (temporal) | Burstiness, growth, inter-arrival | 0.528 | 0.528 | 0.528 | 0.528 | 0.545 |
| RF (all features) | All above | 0.532 | 0.532 | 0.532 | 0.532 | 0.567 |
| BERT (text-only) | Source tweet text | 0.847 | 0.847 | 0.847 | 0.847 | 0.933 |
| **BiGCN (ours)** | **Graph + BERT + time** | **0.813** | **0.812** | **0.809** | **0.810** | **0.902** |

**Key finding:** BiGCN at 30 min (0.813) vs full cascade (0.810) — gap of 0.003. The early detection curve is remarkably flat, confirming that veracity signal is front-loaded in the first 30 minutes of a cascade's life.

### Feature Importance (Random Forest)

| Feature | Importance |
|---|---|
| Burstiness | 44% |
| Growth@30min | 15% |
| Cascade size | 14% |
| Branching factor | 12% |
| Depth | 9% |
| Other | 6% |

### Twitter15/16 — Secondary Results

| Dataset | Model | Node Features | Macro-F1 | Accuracy |
|---|---|---|---|---|
| Twitter15 | BiGCN (ours) | BERT 771-d | 0.654 | 0.654 |
| Twitter15 | BiGCN (ours) | TF-IDF 5,000-d | 0.811 | 0.810 |
| Twitter15 | Bian et al. 2020 | TF-IDF 5,000-d | — | 0.886 |
| Twitter16 | BiGCN (ours) | BERT 771-d | 0.715 | 0.715 |
| Twitter16 | BiGCN (ours) | TF-IDF 5,000-d | 0.832 | 0.833 |
| Twitter16 | Bian et al. 2020 | TF-IDF 5,000-d | — | 0.880 |

**The ~0.16 gap between BERT and TF-IDF results directly quantifies reply text contribution to BiGCN performance.** Twitter15/16 BERT results are a lower bound — not a ceiling — due to API constraints.

---

## Error Analysis

15 most confidently wrong predictions from BiGCN (full cascade) — highest model confidence on wrong class:

**False Positives (8 cases) — Non-rumour predicted as rumour:**
- Breaking news tweets from verified accounts misidentified due to fast growth and high burstiness mimicking rumour structure
- Conspiratorial reply threads on true events — BU pass picks up challenging replies and incorrectly signals rumour

**False Negatives (7 cases) — Rumour predicted as non-rumour:**
- Emotionally neutral or solidarity language ("thoughts go out to everyone") — no linguistic or structural rumour signal
- Tiny cascades (size=2) with single uninformative reply — insufficient graph structure
- Sarcastic source tweets — irony undetectable from text or cascade structure
- Off-topic reply threads that hijack cascade signal

All 15 examples had model confidence = 1.0000, indicating systematic rather than borderline failures.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/dhanya0412/misinformation-detection.git
cd misinformation-detection

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install torch-scatter
pip install transformers
pip install scikit-learn pandas numpy networkx tqdm matplotlib seaborn umap-learn
```

### Environment
- Python 3.10+
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- CUDA 11.8+ (GPU required for BERT embedding extraction and BiGCN training)
- Lightning AI Studio (development environment)

---

## Reproducing Results

### Step 1 — Data Preparation and EDA (`week1.ipynb`)

Run all cells in order. This notebook:
1. Parses raw PHEME JSON files into NetworkX DiGraphs
2. Computes structural and temporal features per cascade → `cascade_features.csv`
3. Trains Random Forest baselines → `rf_*.pkl`, `rf_early_detection_scores.csv`
4. Fine-tunes BERT for binary classification → `bert_preds.npy`, `bert_results.csv`
5. Extracts BERT embeddings for all 103,572 nodes → `all_node_embeddings.npy`
6. Saves all cascade graphs → `all_cascades.pkl`

**Runtime:** ~45 minutes on GPU (dominated by BERT embedding extraction)

### Step 2 — BiGCN on PHEME (`bigcn.ipynb`)

Run all cells in order. This notebook:
1. Loads all saved files from Step 1
2. Builds PyG Data objects at 4 time cuts (30/60/120/full)
3. Trains BiGCN with 5-fold stratified CV at each time cut
4. Saves results and generates the early detection curve

**Runtime:** ~1-2 hours on GPU (50 epochs × 4 cuts × 5 folds)

### Step 3 — Error Analysis (`error_analysis.ipynb`)

Run all cells in order. Requires outputs from Steps 1 and 2.

**Runtime:** ~5 minutes

### Step 4 — Twitter15/16 Experiments

**BERT features with time cuts (`week3_twitter.ipynb`):**
1. Loads Twitter15/16 datasets
2. Embeds source tweets with BERT
3. Builds PyG Data objects with source embedding propagated to all nodes
4. Trains BiGCN with time cuts (60/120/240/full)

**TF-IDF features (`twitter_tfidf.ipynb`):**
1. Parses preprocessed TF-IDF features from BiGCN repo
2. Builds PyG Data objects with 5,000-dim sparse TF-IDF node features
3. Trains original BiGCN architecture (root tiling + scatter_mean)

**Runtime:** ~3-4 hours on GPU for both notebooks

---

## Model Checkpoints

Model weights are not saved by default (early stopping selects best checkpoint per fold, used immediately for evaluation). To save checkpoints, add to `run_cv`:

```python
# Save best model per fold
torch.save(model.state_dict(),
           f'{OUTPUTS}/bigcn_fold{fold+1}_best.pt')
```

---

## Limitations

1. **Static graph snapshots** — BiGCN processes a frozen cascade at each time cut. It cannot model the burst → correction evolution dynamically. Continuous-time temporal GNNs would be more principled.

2. **API-constrained node features (Twitter15/16)** — All nodes sharing the source tweet embedding removes per-node textual variation, neutralising BiGCN's root tiling mechanism. Twitter15/16 BERT results are a lower bound, not a ceiling.

3. **Binary classification on PHEME** — PHEME supports 4 veracity levels. We collapse to binary (rumour vs non-rumour). Fine-grained multi-class may reveal additional early-detection structure.

4. **Event-level generalisation** — Ottawa Shooting and Sydney Siege are 100% rumour, creating event-level confounding in 5-fold CV. Leave-one-event-out evaluation is needed for stronger generalisation claims.

5. **2-layer GCN receptive field** — Adequate for PHEME (avg depth 2-3) but insufficient for Twitter15/16 (depth 10-20+). Deeper architectures required for larger cascade datasets.

---

## References

- Bian, T. et al. (2020). Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks. *AAAI*. [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/5393)
- Zubiaga, A. et al. (2016). Analysing How People Orient to and Spread Rumours. *PLOS ONE*. [[paper]](https://doi.org/10.1371/journal.pone.0150989)
- Kwon, S. et al. (2013). Prominent Features of Rumor Propagation in Online Social Media. *IEEE ICDM*. [[paper]](https://doi.org/10.1109/ICDM.2013.61)
- Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL*. [[paper]](https://arxiv.org/abs/1810.04805)
- Kipf, T. & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *ICLR*. [[paper]](https://arxiv.org/abs/1609.02907)
- Cui, L. & Jia, J. (2024). RAGCL: Rumor Detection with Adaptive Graph Contrastive Learning. *ACL*.
- Sun, M. et al. (2022). GACL: Stance Detection with Graph Adversarial Contrastive Learning. *ACL*.

---

## Acknowledgements

- Bian et al. for the BiGCN codebase: [https://github.com/TianBian95/BiGCN](https://github.com/TianBian95/BiGCN)
- PHEME dataset: Zubiaga et al., University of Warwick
- Twitter15/16 dataset: Ma et al., ACL 2017