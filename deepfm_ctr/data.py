from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List

USER_COL = "user_id"
ITEM_COL = "MatNum"
DATE_COL = "BillingDate"

CUSTGRP_MAPPING = {
    'Road-Fast food': 'Fast Food',
    'Road-Restaurant': 'Restauran',
    'Road-Supermarket': 'Supermarket',
    'Mehdu-Wholesale': 'Wholesale(B)',
}

def load_raw(path: str) -> pd.DataFrame:
    """Load the Excel file and apply basic cleaning.

    This mirrors the logic from the Colab notebooks but in a compact form.
    """
    df = pd.read_excel(path)
    if USER_COL not in df.columns:
        df[USER_COL] = df.get("ShipToNum", df.index)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, USER_COL, ITEM_COL]).copy()
    df["CustGrpName"] = df.get("CustGrpName", "--").replace(CUSTGRP_MAPPING).fillna("--")
    df["MatName"] = df.get("MatName", "Unknown").fillna("Unknown")
    # combined_rating fallback
    if "combined_rating" not in df.columns:
        from sklearn.preprocessing import MinMaxScaler
        vcol = "BillingNetValue" if "BillingNetValue" in df.columns else None
        qcol = "BillingNetWeight" if "BillingNetWeight" in df.columns else None
        if vcol and qcol:
            sv, sq = MinMaxScaler(), MinMaxScaler()
            df["_v"] = sv.fit_transform(df[[vcol]])
            df["_q"] = sq.fit_transform(df[[qcol]])
            df["combined_rating"] = 0.7 * df["_v"] + 0.3 * df["_q"]
            df.drop(columns=["_v", "_q"], inplace=True)
        else:
            df["combined_rating"] = 1.0
    df["BasketTime"] = df[DATE_COL].dt.floor("D")
    return df

def make_next_basket_split(df: pd.DataFrame, q_train: float = 0.8, q_val: float = 0.9):
    baskets = (df.groupby([USER_COL, "BasketTime"])[ITEM_COL]
               .apply(list).reset_index()
               .sort_values(["BasketTime", USER_COL]))
    q1 = baskets["BasketTime"].quantile(q_train)
    q2 = baskets["BasketTime"].quantile(q_val)
    train_b = baskets[baskets["BasketTime"] <= q1].copy()
    val_b   = baskets[(baskets["BasketTime"] > q1) & (baskets["BasketTime"] <= q2)].copy()
    test_b  = baskets[baskets["BasketTime"] > q2].copy()
    return train_b, val_b, test_b

def make_loo_last_basket_split(df: pd.DataFrame):
    baskets = (df.groupby([USER_COL, "BasketTime"])[ITEM_COL]
               .apply(list).reset_index()
               .sort_values([USER_COL, "BasketTime"]))
    last_time = baskets.groupby(USER_COL)["BasketTime"].transform("max")
    is_last = baskets["BasketTime"] == last_time
    test_b = baskets[is_last].copy()
    hist_b = baskets[~is_last].copy()
    cut = hist_b["BasketTime"].quantile(0.9)
    train_b = hist_b[hist_b["BasketTime"] <= cut].copy()
    val_b   = hist_b[hist_b["BasketTime"] > cut].copy()
    return train_b, val_b, test_b, hist_b

def build_item_aggs(df_hist: pd.DataFrame) -> pd.DataFrame:
    aggs = (df_hist.groupby(ITEM_COL)
            .agg(item_freq=(USER_COL, "nunique"),
                 item_orders=("BasketTime", "nunique"),
                 item_combined_rating_avg=("combined_rating", "mean"))
            .reset_index())
    aggs["item_freq"]   = aggs["item_freq"].fillna(0).astype(int)
    aggs["item_orders"] = aggs["item_orders"].fillna(0).astype(int)
    aggs["item_combined_rating_avg"] = aggs["item_combined_rating_avg"].fillna(
        aggs["item_combined_rating_avg"].median()
    )
    return aggs

def latest_user_item_ctx(df_hist: pd.DataFrame) -> pd.DataFrame:
    return (df_hist.sort_values(DATE_COL)
            .drop_duplicates(subset=[USER_COL, ITEM_COL], keep="last")
            .loc[:, [USER_COL, ITEM_COL, "CustGrpName", "MatName"]])

def user_last_seen(df_hist: pd.DataFrame) -> pd.Series:
    return df_hist.groupby(USER_COL)[DATE_COL].max().rename("last_seen_hist")
