from __future__ import annotations
import pandas as pd
from typing import Dict
from .config import USER_COL, ITEM_COL, DATE_COL, RATING_COL

CUSTGRP_MAPPING = {
    'Road-Fast food': 'Fast Food',
    'Road-Restaurant': 'Restauran',
    'Road-Supermarket': 'Supermarket',
    'Mehdu-Wholesale': 'Wholesale(B)',
}

def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    if USER_COL not in df.columns:
        df[USER_COL] = df.get("ShipToNum", df.index)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, USER_COL, ITEM_COL]).copy()
    df["CustGrpName"] = df.get("CustGrpName", "--").replace(CUSTGRP_MAPPING).fillna("--")
    df["MatName"] = df.get("MatName", "Unknown").fillna("Unknown")
    if RATING_COL not in df.columns:
        from sklearn.preprocessing import MinMaxScaler
        vcol, qcol = "BillingNetValue", "BillingNetWeight"
        if vcol in df.columns and qcol in df.columns:
            sv, sq = MinMaxScaler(), MinMaxScaler()
            df["_v"] = sv.fit_transform(df[[vcol]])
            df["_q"] = sq.fit_transform(df[[qcol]])
            df[RATING_COL] = 0.7 * df["_v"] + 0.3 * df["_q"]
            df.drop(columns=["_v", "_q"], inplace=True)
        else:
            df[RATING_COL] = 1.0
    df["BasketTime"] = df[DATE_COL].dt.floor("D")
    return df

def split_by_custgrp(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for g, gdf in df.groupby("CustGrpName"):
        out[str(g)] = gdf.copy()
    return out
