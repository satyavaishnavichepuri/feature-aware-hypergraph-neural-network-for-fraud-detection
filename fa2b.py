# ============================================================
# Step FA2b — Final Configuration
#   k=5  |  L2 normalization  |  prune_p75
#   Runs on BOTH Elliptic and Elliptic++ sequentially
#   Reports: overall F1 + connected/isolated breakdown
# ============================================================

import subprocess, sys

def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

import torch
_tv  = torch.__version__.split("+")[0]
_cv  = ("cu" + torch.version.cuda.replace(".", "")) if torch.cuda.is_available() else "cpu"
_whl = f"https://data.pyg.org/whl/torch-{_tv}+{_cv}.html"
print(f"Installing PyG wheels  torch={_tv}  cuda={_cv}")
pip("torch_geometric")
pip("torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv",
    "--find-links", _whl)
pip("scikit-learn", "pandas", "numpy", "matplotlib")

# ── Imports ──────────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report)
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings, copy
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}\n")

# ════════════════════════════════════════════════════════════
# DATASET CONFIGS
# Two entries — one per dataset.  Only the paths and
# temporal split cutoffs differ; all other hyperparameters
# are identical so results are directly comparable.
# ════════════════════════════════════════════════════════════

DATASETS = [
    {
        "name"          : "Elliptic",
        "features_path" : "elliptic_txs_features.csv",
        "classes_path"  : "elliptic_txs_classes.csv",
        "edges_path"    : "elliptic_txs_edgelist.csv",
        # Elliptic has 49 timesteps; 70/10/20 split → train ts 1-34
        "train_frac"    : 0.70,
        "val_frac"      : 0.10,
        # Column layout: no header, col0=txid, col1=timestep, rest=features
        "has_header"    : False,
        "txid_col"      : 0,
        "ts_col"        : 1,
        "feat_start"    : 2,
    },
    {
        "name"          : "Elliptic++",
        "features_path" : "txs_features.csv",
        "classes_path"  : "txs_classes.csv",
        "edges_path"    : "txs_edgelist.csv",
        # Elliptic++ has 43 timesteps; 70/10/20 → train ts 1-30
        "train_frac"    : 0.70,
        "val_frac"      : 0.10,
        # Elliptic++ has a header row
        # txs_classes.csv uses "txId" (capital I) and "class"
        # txs_features.csv uses "txid" and "time step"
        "has_header"    : True,
        "txid_col"      : "txId",       # features file col
        "ts_col"        : "Time step",  # features file col
        "cls_txid_col"  : "txId",       # classes file col (capital I)
        "feat_start"    : None,         # auto-detected from non-id cols
        # Elliptic++ label mapping: 1=illicit, 2=licit, 3=unknown
        "label_map"     : {1: 1, 2: 0},
        # Higher pos_weight ceiling — illicit rate ~10.86%
        "pw_max"        : 10.0,
    },
]

# ════════════════════════════════════════════════════════════
# FA2b FIXED HYPERPARAMETERS
# ════════════════════════════════════════════════════════════

TARGET_HE_SIZE   = 5      # k=5: smaller clusters → more hyperedges
                           #      → more illicit nodes covered
MIN_CLUSTER_SIZE = 3
MAX_HE_PER_NODE  = 2
MAX_HE_SIZE      = 10     # scaled down from 15 to match k=5
PRUNE_PCT        = 75     # hard prune bottom 75% by coherence

NORMALIZE_L2     = True   # P4 finding: improves coherence quality

POS_WEIGHT_MIN   = 3.0
POS_WEIGHT_MAX   = 4.0

THRESHOLD_LOW    = 0.30
THRESHOLD_HIGH   = 0.75
THRESHOLD_STEPS  = 40

CONSISTENCY_K    = 1

HIDDEN_DIM       = 128
DROPOUT          = 0.3
LR               = 1e-3
WEIGHT_DECAY     = 1e-4
EPOCHS           = 100
LOG_EVERY        = 10

# ════════════════════════════════════════════════════════════
# 1. DATA LOADING  (handles both Elliptic and Elliptic++ layouts)
# ════════════════════════════════════════════════════════════

def load_dataset(cfg):
    print(f"\n{'═'*60}")
    print(f"  DATASET: {cfg['name']}")
    print(f"{'═'*60}")
    print("Loading CSVs …")

    # ── Features ─────────────────────────────────────────────
    if cfg["has_header"]:
        feat_df = pd.read_csv(cfg["features_path"])
        txid_col = cfg["txid_col"]
        ts_col   = cfg["ts_col"]
        # All columns that are not txid or timestep are features
        non_feat = {txid_col, ts_col}
        feat_cols = [c for c in feat_df.columns if c not in non_feat]
        feat_df = feat_df.rename(columns={txid_col: "txid", ts_col: "timestep"})
        feat_df["_feat_cols"] = None   # placeholder; actual cols in feat_cols
    else:
        feat_df = pd.read_csv(cfg["features_path"], header=None)
        feat_df.columns = (
            ["txid", "timestep"] +
            [f"f{i}" for i in range(1, feat_df.shape[1] - 1)]
        )
        feat_cols = [c for c in feat_df.columns
                     if c not in ("txid", "timestep")]

    print(f"  Transactions     : {len(feat_df):,}")
    print(f"  Feature columns  : {len(feat_cols)}")

    # ── Classes ──────────────────────────────────────────────
    cls_df = pd.read_csv(cfg["classes_path"])
    # Use exact column name from config if provided, else auto-detect
    cls_txid_col = cfg.get("cls_txid_col", None)
    if cls_txid_col and cls_txid_col in cls_df.columns:
        # Rename to lowercase "txid" for consistent merge key
        cls_df = cls_df.rename(columns={cls_txid_col: "txid"})
    else:
        # Fallback: normalise all column names and find txid-like col
        cls_df.columns = [c.strip() for c in cls_df.columns]
        txid_like = next(
            (c for c in cls_df.columns
             if c.lower() in ["txid", "tx_id", "id", "transaction_id"]),
            cls_df.columns[0]
        )
        cls_df = cls_df.rename(columns={txid_like: "txid"})
    # Ensure class column is named "class"
    non_txid = [c for c in cls_df.columns if c != "txid"]
    if non_txid and non_txid[0] != "class":
        cls_df = cls_df.rename(columns={non_txid[0]: "class"})

    return feat_df, cls_df, feat_cols


# ════════════════════════════════════════════════════════════
# 2. PREPROCESSING  (StandardScaler + optional L2 norm)
# ════════════════════════════════════════════════════════════

def preprocess(feat_df, cls_df, feat_cols, normalize_l2=True,
               label_map=None, pw_max=4.0):
    print("\nPreprocessing …")
    df = feat_df.merge(cls_df, on="txid", how="left")

    # Handle label mapping
    # Default (Elliptic): "unknown" → drop, "1"→illicit, "2"→licit
    # Elliptic++: 1→illicit, 2→licit, 3→drop (via label_map)
    if label_map is not None:
        df["class"] = pd.to_numeric(df["class"], errors="coerce")
        df = df[df["class"].isin(label_map.keys())].copy()
        df["label"] = df["class"].map(label_map).astype(int)
    else:
        df = df[~df["class"].isin(["unknown", 3, "3"])].copy()
        df["label"] = df["class"].apply(
            lambda x: 1 if str(x).strip() == "1" else 0
        ).astype(int)

    df = df.reset_index(drop=True)

    X = df[feat_cols].values.astype(np.float32)

    # NaN imputation
    nan_count = int(np.isnan(X).sum())
    if nan_count > 0:
        print(f"  NaN values       : {nan_count:,} → imputing with median")
        imp = SimpleImputer(strategy="median")
        X   = imp.fit_transform(X).astype(np.float32)
    else:
        print(f"  NaN values       : none")

    X_scaled = StandardScaler().fit_transform(X).astype(np.float32)

    if normalize_l2:
        norms    = np.linalg.norm(X_scaled, axis=1, keepdims=True)
        X_scaled = X_scaled / (norms + 1e-8)
        print("  L2 normalization : applied")

    labels    = df["label"].values.astype(np.int64)
    timesteps = df["timestep"].values.astype(np.int64)
    txids     = df["txid"].values

    print(f"  Labeled txs      : {len(df):,}  "
          f"(illicit={labels.sum():,}, "
          f"licit={(labels==0).sum():,})")
    print(f"  Illicit rate     : {100*labels.mean():.2f}%")
    print(f"  Unique timesteps : {len(np.unique(timesteps))}  "
          f"(range {timesteps.min()}–{timesteps.max()})")
    print(f"  Feature dims     : {X_scaled.shape[1]}")

    return X_scaled, labels, timesteps, txids


# ════════════════════════════════════════════════════════════
# 3. COHERENCE METRIC  (hybrid: cosine + compactness)
# ════════════════════════════════════════════════════════════

def compute_coherence(X_members: np.ndarray) -> float:
    n = len(X_members)
    if n < 2:
        return 0.0
    cos_mat       = cosine_similarity(X_members)
    mask          = np.triu(np.ones((n, n), dtype=bool), k=1)
    mean_cos      = float(cos_mat[mask].mean()) if mask.sum() > 0 else 0.0
    mean_cos_norm = (mean_cos + 1.0) / 2.0
    centroid      = X_members.mean(axis=0, keepdims=True)
    dists         = np.linalg.norm(X_members - centroid, axis=1)
    d             = X_members.shape[1]
    compactness   = max(0.0, 1.0 - float(dists.mean()) / (np.sqrt(d) + 1e-8))
    return 0.5 * mean_cos_norm + 0.5 * compactness


# ════════════════════════════════════════════════════════════
# 4. KMEANS CLUSTERING  (per timestep, target size = k=5)
# ════════════════════════════════════════════════════════════

def _kmeans_clusters(X_local, n_clusters, random_state=42):
    if len(X_local) <= n_clusters:
        return np.arange(len(X_local))
    if len(X_local) > 2000:
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state,
                             batch_size=512, n_init=3)
    else:
        km = KMeans(n_clusters=n_clusters, random_state=random_state,
                    n_init=5, max_iter=100)
    return km.fit_predict(X_local)


def build_all_clusters(X_scaled, labels, timesteps, txids,
                       target_he_size, min_cluster_size,
                       max_he_per_node, max_he_size):
    n_tx      = len(X_scaled)
    unique_ts = np.sort(np.unique(timesteps))
    participation  = np.zeros(n_tx, dtype=np.int32)
    raw_hyperedges = []
    raw_coherences = []
    n_dropped      = 0

    for ts in unique_ts:
        mask      = timesteps == ts
        local_idx = np.where(mask)[0]
        n_local   = len(local_idx)
        if n_local < min_cluster_size:
            n_dropped += n_local
            continue

        n_clusters     = max(1, n_local // target_he_size)
        cluster_labels = _kmeans_clusters(X_scaled[local_idx], n_clusters)

        for c in range(n_clusters):
            members_global = local_idx[cluster_labels == c]
            if len(members_global) < min_cluster_size:
                n_dropped += len(members_global)
                continue

            if len(members_global) > max_he_size:
                sub_k      = max(2, len(members_global) // target_he_size)
                sub_labels = _kmeans_clusters(X_scaled[members_global], sub_k)
                sub_groups = [members_global[sub_labels == sc]
                              for sc in range(sub_k)]
            else:
                sub_groups = [members_global]

            for sc_members in sub_groups:
                if len(sc_members) < min_cluster_size:
                    n_dropped += len(sc_members)
                    continue
                eligible = sc_members[
                    participation[sc_members] < max_he_per_node
                ]
                if len(eligible) < min_cluster_size:
                    continue
                coh = compute_coherence(X_scaled[eligible])
                raw_hyperedges.append(eligible)
                raw_coherences.append(coh)
                participation[eligible] += 1

    raw_coherences = np.array(raw_coherences, dtype=np.float32)
    print(f"\nBuilding clusters (k={target_he_size}, L2={NORMALIZE_L2}) …")
    print(f"  Raw hyperedges   : {len(raw_hyperedges):,}")
    print(f"  Nodes dropped    : {n_dropped:,}")
    return raw_hyperedges, raw_coherences, participation, n_tx


# ════════════════════════════════════════════════════════════
# 5. PRUNE at p75
# ════════════════════════════════════════════════════════════

def prune_p75(raw_hyperedges, raw_coherences, n_tx, pct=75):
    c_min  = raw_coherences.min()
    c_max  = raw_coherences.max()
    normed = (raw_coherences - c_min) / (c_max - c_min + 1e-8)

    print(f"\nCoherence distribution (normalised):")
    for p in [25, 50, 75, 90]:
        print(f"  p{p:>2d} = {np.percentile(normed, p):.4f}")

    threshold = float(np.percentile(normed, pct))
    keep_mask = normed >= threshold
    kept      = [he for he, k in zip(raw_hyperedges, keep_mask) if k]

    participation = np.zeros(n_tx, dtype=np.int32)
    for he in kept:
        participation[he] += 1
    isolated = int((participation == 0).sum())

    print(f"\nPruning at p{pct} (thr={threshold:.4f}):")
    print(f"  Retained         : {len(kept):,} / {len(raw_hyperedges):,} "
          f"({100*len(kept)/len(raw_hyperedges):.1f}%)")
    print(f"  Isolated nodes   : {isolated:,} / {n_tx:,} "
          f"({100*isolated/n_tx:.1f}%)")
    print(f"  Avg HE size      : "
          f"{np.mean([len(h) for h in kept]):.2f}" if kept else "  (none)")

    return kept, participation, isolated, threshold


# ════════════════════════════════════════════════════════════
# 6. GRAPH CONSTRUCTION
# ════════════════════════════════════════════════════════════

def build_graph(X_scaled, labels, timesteps, hyperedges, n_tx):
    n_he        = len(hyperedges)
    he_features = np.zeros((n_he, X_scaled.shape[1]), dtype=np.float32)
    for i, members in enumerate(hyperedges):
        he_features[i] = X_scaled[members].mean(axis=0)

    X_all = np.vstack([X_scaled, he_features])
    y_all = np.concatenate([labels, np.full(n_he, -1, dtype=np.int64)])

    src, dst = [], []
    for he_i, members in enumerate(hyperedges):
        he_node = n_tx + he_i
        for tx in members:
            src.append(int(tx));  dst.append(he_node)
            src.append(he_node);  dst.append(int(tx))

    data = Data(
        x          = torch.tensor(X_all, dtype=torch.float),
        edge_index = torch.tensor([src, dst], dtype=torch.long),
        y          = torch.tensor(y_all,  dtype=torch.float),
    )
    data.n_tx         = n_tx
    data.n_he         = n_he
    data.tx_timesteps = torch.tensor(timesteps, dtype=torch.long)
    return data


# ════════════════════════════════════════════════════════════
# 7. TEMPORAL SPLIT
# ════════════════════════════════════════════════════════════

def temporal_split(data, timesteps, train_frac, val_frac):
    unique_ts = np.sort(np.unique(timesteps))
    n_ts      = len(unique_ts)
    train_end = int(n_ts * train_frac)
    val_end   = int(n_ts * (train_frac + val_frac))

    train_set = set(unique_ts[:train_end])
    val_set   = set(unique_ts[train_end:val_end])
    test_set  = set(unique_ts[val_end:])

    n_total    = data.x.shape[0]
    train_mask = torch.zeros(n_total, dtype=torch.bool)
    val_mask   = torch.zeros(n_total, dtype=torch.bool)
    test_mask  = torch.zeros(n_total, dtype=torch.bool)

    for i, ts in enumerate(timesteps):
        if   ts in train_set: train_mask[i] = True
        elif ts in val_set:   val_mask[i]   = True
        elif ts in test_set:  test_mask[i]  = True

    data.train_mask = train_mask
    data.val_mask   = val_mask
    data.test_mask  = test_mask

    print(f"\nTemporal split  "
          f"(train ts {unique_ts[0]}–{unique_ts[train_end-1]} | "
          f"val ts {unique_ts[train_end]}–{unique_ts[val_end-1]} | "
          f"test ts {unique_ts[val_end]}–{unique_ts[-1]}):")
    print(f"  Train : {train_mask[:len(timesteps)].sum().item():,} nodes")
    print(f"  Val   : {val_mask[:len(timesteps)].sum().item():,} nodes")
    print(f"  Test  : {test_mask[:len(timesteps)].sum().item():,} nodes")
    return data


# ════════════════════════════════════════════════════════════
# 8. CLASS WEIGHTS
# ════════════════════════════════════════════════════════════

def compute_pos_weight(data, pw_min=3.0, pw_max=4.0):
    y_tr    = data.y[data.train_mask]
    n_pos   = (y_tr == 1).sum().item()
    n_neg   = (y_tr == 0).sum().item()
    raw_pw  = n_neg / max(n_pos, 1)
    clamped = float(np.clip(raw_pw, pw_min, pw_max))
    print(f"  pos_weight       : raw={raw_pw:.1f} → clamped={clamped:.2f}")
    return torch.tensor([clamped], dtype=torch.float)


# ════════════════════════════════════════════════════════════
# 9. MODEL
# ════════════════════════════════════════════════════════════

class GCNFraudDetector(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1   = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels // 2)
        self.bn2   = nn.BatchNorm1d(hidden_channels // 2)
        self.head  = nn.Linear(hidden_channels // 2, 1)
        self.drop  = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index); x = self.bn1(x); x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)
        x = self.conv2(x, edge_index); x = self.bn2(x); x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)
        return self.head(x).squeeze(-1)


# ════════════════════════════════════════════════════════════
# 10. TRAINING
# ════════════════════════════════════════════════════════════

def train(data, pos_weight, label=""):
    model = GCNFraudDetector(
        in_channels=data.x.shape[1],
        hidden_channels=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))

    print(f"\n── Training [{label}] ────────────────────────────────")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss   = criterion(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4d}/{EPOCHS}  loss={loss.item():.4f}")
    print("  Training complete.")
    return model


# ════════════════════════════════════════════════════════════
# 11. THRESHOLD SEARCH
# ════════════════════════════════════════════════════════════

@torch.no_grad()
def find_threshold(model, data):
    model.eval()
    probs  = torch.sigmoid(model(data.x, data.edge_index))
    y_true = data.y[data.val_mask].cpu().numpy().astype(int)
    p_arr  = probs[data.val_mask].cpu().numpy()

    best_f1, best_thr = -1.0, 0.5
    for thr in np.linspace(THRESHOLD_LOW, THRESHOLD_HIGH, THRESHOLD_STEPS):
        f1 = f1_score(y_true, (p_arr >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    print(f"  Best threshold   : {best_thr:.2f}  val_F1={best_f1:.4f}")
    return best_thr


# ════════════════════════════════════════════════════════════
# 12. STRUCTURAL CONSISTENCY FILTER
# ════════════════════════════════════════════════════════════

def consistency_filter(preds_binary, edge_index, n_tx, k=1):
    if k <= 0:
        return preds_binary
    n_total  = preds_binary.shape[0]
    preds_np = preds_binary.cpu().numpy().copy()
    filtered = preds_np.copy()
    adj      = [[] for _ in range(n_total)]
    for s, d in zip(edge_index[0].cpu().tolist(),
                    edge_index[1].cpu().tolist()):
        adj[s].append(d)

    flipped = 0
    for tx_i in [i for i in range(n_tx) if preds_np[i] == 1]:
        co_fraud = 0
        for he in adj[tx_i]:
            if he < n_tx:
                continue
            for tx_j in adj[he]:
                if tx_j < n_tx and tx_j != tx_i and preds_np[tx_j] == 1:
                    co_fraud += 1
                    break
            if co_fraud >= k:
                break
        if co_fraud < k:
            filtered[tx_i] = 0
            flipped += 1

    print(f"  Consistency filter: {flipped:,} flipped → licit")
    return torch.tensor(filtered, dtype=torch.long,
                        device=preds_binary.device)


# ════════════════════════════════════════════════════════════
# 13. EVALUATION + P1 BREAKDOWN
# ════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, data, threshold, n_tx, participation, label=""):
    model.eval()
    probs     = torch.sigmoid(model(data.x, data.edge_index))
    preds_raw = (probs >= threshold).long()
    preds_fin = consistency_filter(preds_raw, data.edge_index,
                                   n_tx, k=CONSISTENCY_K)

    mask   = data.test_mask
    y_true = data.y[mask].cpu().numpy().astype(int)
    y_pred = preds_fin[mask].cpu().numpy().astype(int)

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score   (y_true, y_pred, zero_division=0)
    f1   = f1_score       (y_true, y_pred, zero_division=0)

    print(f"\n── [{label}] Overall Test Metrics ───────────────────")
    print(f"  Threshold        : {threshold:.2f}")
    print(f"  Precision        : {prec:.4f}")
    print(f"  Recall           : {rec:.4f}")
    print(f"  F1               : {f1:.4f}")
    print(f"  Predicted fraud  : {y_pred.sum():,}  "
          f"/ true fraud : {y_true.sum():,}")
    print(classification_report(y_true, y_pred,
                                target_names=["Licit","Illicit"],
                                zero_division=0))

    # ── P1: Connected vs Isolated breakdown ──────────────────
    test_idx   = torch.where(mask[:n_tx])[0].cpu().numpy()
    test_part  = participation[test_idx]
    conn_mask  = test_part > 0
    isol_mask  = test_part == 0

    print(f"── P1: Connected vs Isolated Breakdown ──────────────")
    print(f"  Connected nodes  : {conn_mask.sum():,}  "
          f"(illicit={y_true[conn_mask].sum():,})")
    print(f"  Isolated  nodes  : {isol_mask.sum():,}  "
          f"(illicit={y_true[isol_mask].sum():,})")

    breakdown = {}
    for name, gmask in [("connected", conn_mask), ("isolated", isol_mask)]:
        if gmask.sum() == 0:
            continue
        yt = y_true[gmask]; yp = y_pred[gmask]
        if yt.sum() == 0:
            breakdown[name] = dict(f1=float("nan"), precision=float("nan"),
                                   recall=float("nan"),
                                   n=int(gmask.sum()), n_ill=0)
            print(f"  [{name:>9}]  no illicit nodes — skipped")
            continue
        gf1 = f1_score       (yt, yp, zero_division=0)
        gpr = precision_score(yt, yp, zero_division=0)
        grc = recall_score   (yt, yp, zero_division=0)
        breakdown[name] = dict(f1=gf1, precision=gpr, recall=grc,
                               n=int(gmask.sum()), n_ill=int(yt.sum()))
        print(f"  [{name:>9}]  F1={gf1:.4f}  "
              f"Prec={gpr:.4f}  Rec={grc:.4f}  "
              f"(n={gmask.sum():,}, illicit={yt.sum():,})")

    if "connected" in breakdown and "isolated" in breakdown:
        cf1 = breakdown["connected"]["f1"]
        if1 = breakdown["isolated"]["f1"]
        if not (np.isnan(cf1) or np.isnan(if1)):
            delta = cf1 - if1
            verdict = (
                "✓ Hyperedges carry genuine signal (connected >> isolated)."
                if delta > 0.05 else
                "~ Marginal benefit from hyperedge structure."
            )
            print(f"\n  ΔF1 (conn − isol)  : {delta:+.4f}")
            print(f"  Verdict            : {verdict}")

    return dict(f1=f1, precision=prec, recall=rec,
                threshold=threshold,
                n_fraud_pred=int(y_pred.sum()),
                n_true_fraud=int(y_true.sum()),
                breakdown=breakdown)


# ════════════════════════════════════════════════════════════
# 14. SINGLE DATASET PIPELINE
# ════════════════════════════════════════════════════════════

def run_dataset(cfg):
    feat_df, cls_df, feat_cols = load_dataset(cfg)
    X_scaled, labels, timesteps, txids = preprocess(
        feat_df, cls_df, feat_cols,
        normalize_l2 = NORMALIZE_L2,
        label_map    = cfg.get("label_map", None),
        pw_max       = cfg.get("pw_max", POS_WEIGHT_MAX),
    )
    n_tx = len(X_scaled)

    raw_he, raw_coh, _, _ = build_all_clusters(
        X_scaled, labels, timesteps, txids,
        target_he_size  = TARGET_HE_SIZE,
        min_cluster_size= MIN_CLUSTER_SIZE,
        max_he_per_node = MAX_HE_PER_NODE,
        max_he_size     = MAX_HE_SIZE,
    )

    kept_he, participation, isolated, thr_val = prune_p75(
        raw_he, raw_coh, n_tx, pct=PRUNE_PCT
    )

    data = build_graph(X_scaled, labels, timesteps, kept_he, n_tx)
    data = temporal_split(data, timesteps,
                          cfg["train_frac"], cfg["val_frac"])

    pw = compute_pos_weight(data, POS_WEIGHT_MIN,
                            cfg.get("pw_max", POS_WEIGHT_MAX))

    # Move to device
    dev = copy.copy(data)
    dev.x          = data.x.to(DEVICE)
    dev.edge_index = data.edge_index.to(DEVICE)
    dev.y          = data.y.to(DEVICE)
    dev.train_mask = data.train_mask.to(DEVICE)
    dev.val_mask   = data.val_mask.to(DEVICE)
    dev.test_mask  = data.test_mask.to(DEVICE)

    model    = train(dev, pw, label=cfg["name"])
    best_thr = find_threshold(model, dev)
    results  = evaluate(model, dev, best_thr, n_tx,
                        participation, label=cfg["name"])
        # ── F1 per timestep plot ──────────────────────────────────
    te_ts   = timesteps[data.test_mask[:n_tx].cpu().numpy().astype(bool)]
    te_uts  = np.sort(np.unique(te_ts))
    ts_f1s  = []

    for ts in te_uts:
        idx = te_ts == ts
        yt  = data.y[data.test_mask[:n_tx]].cpu().numpy().astype(int)[idx]
        yp  = preds_fin[data.test_mask[:n_tx]].cpu().numpy().astype(int)[idx]
        if yt.sum() == 0:
            ts_f1s.append(float("nan"))
            continue
        ts_f1s.append(f1_score(yt, yp, zero_division=0))

    plt.figure(figsize=(9, 4))
    plt.plot(te_uts, ts_f1s, marker="o", markersize=4)
    plt.axhline(results["f1"], linestyle="--", color="gray",
                label=f"Overall F1={results['f1']:.3f}")
    plt.xlabel("Timestep")
    plt.ylabel("F1 (illicit)")
    plt.title(f"F1 per Timestep — {cfg['name']} Test Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/f1_per_timestep_{cfg['name'].replace('+','p')}.png", dpi=120)
    plt.close()
    print(f"  Plot saved: plots/f1_per_timestep_{cfg['name']}.png")

    results["dataset"]       = cfg["name"]
    results["n_hyperedges"]  = len(kept_he)
    results["n_isolated"]    = isolated
    results["prune_thr"]     = thr_val
    results["target_k"]      = TARGET_HE_SIZE
    results["l2_norm"]       = NORMALIZE_L2
    return results


# ════════════════════════════════════════════════════════════
# 15. MAIN
# ════════════════════════════════════════════════════════════

def main():
    print(f"\nFA2b Configuration:")
    print(f"  target_he_size : {TARGET_HE_SIZE}  (k=5)")
    print(f"  prune_pct      : {PRUNE_PCT}  (retain top 25%)")
    print(f"  L2 norm        : {NORMALIZE_L2}")
    print(f"  Datasets       : {[d['name'] for d in DATASETS]}")

    all_results = {}
    for cfg in DATASETS:
        r = run_dataset(cfg)
        all_results[cfg["name"]] = r

    # ════════════════════════════════════════════════════════
    # FINAL COMBINED SUMMARY TABLE
    # ════════════════════════════════════════════════════════
    print("\n\n" + "═"*72)
    print("  FA2b FINAL RESULTS — k=5, L2 norm, prune_p75")
    print("═"*72)
    print(f"  {'Dataset':<14}  {'#HE':>6}  {'F1':>6}  "
          f"{'Prec':>6}  {'Rec':>6}  "
          f"{'F1_conn':>8}  {'F1_isol':>8}  "
          f"{'%isolated':>10}")
    print("  " + "─"*68)

    for name, r in all_results.items():
        n_tx_total = r["n_true_fraud"] + r.get("n_isolated", 0)
        pct_isol   = 100 * r["n_isolated"] / max(
            r["n_hyperedges"] + r["n_isolated"], 1
        )
        bd   = r.get("breakdown", {})
        cf1  = bd.get("connected", {}).get("f1", float("nan"))
        if1  = bd.get("isolated",  {}).get("f1", float("nan"))
        cf1s = f"{cf1:.4f}" if not np.isnan(cf1) else "N/A"
        if1s = f"{if1:.4f}" if not np.isnan(if1) else "N/A"
        print(f"  {name:<14}  {r['n_hyperedges']:>6,}  "
              f"{r['f1']:>6.4f}  {r['precision']:>6.4f}  "
              f"{r['recall']:>6.4f}  "
              f"{cf1s:>8}  {if1s:>8}  "
              f"{r['n_isolated']:>6,} ({pct_isol:.0f}%)")

    print("═"*72)

    # Comparison to FA1 p75 k=10 (no L2) results from paper
    fa1_results = {
        "Elliptic"  : dict(f1=0.4435, precision=0.4209, recall=0.4686),
        "Elliptic++": dict(f1=0.5150, precision=0.4265, recall=0.6499),
    }

    print(f"\n  Delta vs FA1 p75 k=10 (no L2 norm):")
    print(f"  {'Dataset':<14}  {'FA1 F1':>8}  {'FA2b F1':>8}  {'ΔF1':>8}")
    print("  " + "─"*44)
    for name, r in all_results.items():
        if name in fa1_results:
            fa1_f1 = fa1_results[name]["f1"]
            delta  = r["f1"] - fa1_f1
            print(f"  {name:<14}  {fa1_f1:>8.4f}  "
                  f"{r['f1']:>8.4f}  {delta:>+8.4f}")

    print("═"*72)
    print("\nFA2b complete.")
    return all_results


if __name__ == "__main__":
    results = main()