from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict
import torch
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import DeepFM

from .features import SPARSE_COLS, DENSE_COLS

def build_deepfm_feature_columns(vocab_sizes: Dict[str, int], embed_dim: int):
    fixlen_feature_columns = (
        [SparseFeat(c, vocabulary_size=vocab_sizes[c], embedding_dim=embed_dim)
         for c in SPARSE_COLS] +
        [DenseFeat(c, 1) for c in DENSE_COLS]
    )
    feature_names = get_feature_names(fixlen_feature_columns + fixlen_feature_columns * 0)
    return fixlen_feature_columns, feature_names

def to_model_inputs(df_: pd.DataFrame, feature_names):
    return {name: df_[name].values for name in feature_names if name in df_.columns}

def create_model(vocab_sizes, embed_dim: int, lr: float):
    linear_feature_columns, feature_names = build_deepfm_feature_columns(vocab_sizes, embed_dim)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeepFM(linear_feature_columns, linear_feature_columns, task="binary",
                   l2_reg_embedding=1e-6, dnn_hidden_units=(256, 128, 64),
                   dnn_dropout=0.1, device=device)
    model.compile(optimizer=torch.optim.Adam(model.parameters(), lr=lr),
                  loss="binary_crossentropy", metrics=["auc"])
    return model, feature_names, device

def dcg_at_k(rel, k):
    rel = np.asarray(rel, dtype=float)[:k]
    if rel.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, rel.size + 2))
    return float((rel / discounts).sum())

def ndcg_at_k(truth_items, ranked_items, k):
    topk = ranked_items[:k]
    rel = [1.0 if it in truth_items else 0.0 for it in topk]
    dcg = dcg_at_k(rel, k)
    ideal_rel = [1.0] * min(k, len(truth_items))
    idcg = dcg_at_k(ideal_rel, k)
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_topk(scored_df: pd.DataFrame, k: int = 10):
    metrics = {"precision": [], "recall": [], "hit_rate": [], "ndcg": []}
    from .data import USER_COL
    for (u, t), grp in scored_df.groupby([USER_COL, "BasketTime"]):
        truth = set(grp.loc[grp["label"] == 1, "MatNum"].tolist())
        if not truth:
            continue
        ranked = grp.sort_values("score", ascending=False)["MatNum"].tolist()
        topk = ranked[:k]
        hits = sum(1 for i in topk if i in truth)
        precision = hits / k
        recall = hits / len(truth)
        hit_rate = 1.0 if hits > 0 else 0.0
        ndcg = ndcg_at_k(truth, ranked, k)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["hit_rate"].append(hit_rate)
        metrics["ndcg"].append(ndcg)
    return {m: float(np.mean(v)) if v else 0.0 for m, v in metrics.items()}
