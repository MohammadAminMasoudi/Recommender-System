from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple

from .data import USER_COL, ITEM_COL
from .data import build_item_aggs, latest_user_item_ctx, user_last_seen

SPARSE_COLS = [USER_COL, ITEM_COL, "CustGrpName", "MatName"]
DENSE_COLS = ["item_combined_rating_avg", "item_freq_log1p",
              "item_orders_log1p", "days_since_user_seen_clip"]

def add_features_hist(df_hist: pd.DataFrame,
                      imp: pd.DataFrame) -> pd.DataFrame:
    item_aggs = build_item_aggs(df_hist)
    latest_ctx = latest_user_item_ctx(df_hist)
    last_seen = user_last_seen(df_hist)

    x = imp.merge(latest_ctx, on=[USER_COL, ITEM_COL], how="left")
    x["CustGrpName"] = x["CustGrpName"].fillna("--")
    x["MatName"] = x["MatName"].fillna("Unknown")
    x = x.merge(item_aggs, on=ITEM_COL, how="left")
    med = item_aggs["item_combined_rating_avg"].median()
    x["item_combined_rating_avg"] = x["item_combined_rating_avg"].fillna(med)
    x["item_freq"] = x["item_freq"].fillna(0)
    x["item_orders"] = x["item_orders"].fillna(0)
    x["item_freq_log1p"] = np.log1p(x["item_freq"])
    x["item_orders_log1p"] = np.log1p(x["item_orders"])
    x = x.merge(last_seen, on=USER_COL, how="left")
    x["days_since_user_seen_clip"] = (x["BasketTime"] - x["last_seen_hist"]).dt.days
    x["days_since_user_seen_clip"] = x["days_since_user_seen_clip"].fillna(999).clip(-1, 60)
    x.drop(columns=["last_seen_hist"], inplace=True)
    return x

def fit_label_encoders(train_x: pd.DataFrame, val_x: pd.DataFrame):
    encoders: Dict[str, LabelEncoder] = {}
    index_maps: Dict[str, Dict[str, int]] = {}
    vocab_sizes: Dict[str, int] = {}
    for c in SPARSE_COLS:
        enc = LabelEncoder()
        enc.fit(pd.concat([train_x[c], val_x[c]], ignore_index=True).astype(str))
        encoders[c] = enc
        index_maps[c] = {tok: (i + 1) for i, tok in enumerate(enc.classes_)}
        vocab_sizes[c] = len(enc.classes_) + 1
    return encoders, index_maps, vocab_sizes

def apply_encode(df_: pd.DataFrame,
                 index_maps,
                 dense_default: float = 0.0) -> pd.DataFrame:
    x = df_.copy()
    for c in SPARSE_COLS:
        x[c] = x[c].astype(str).map(index_maps[c]).fillna(0).astype("int64")
    for c in DENSE_COLS:
        x[c] = x[c].astype("float32").fillna(dense_default)
    return x
