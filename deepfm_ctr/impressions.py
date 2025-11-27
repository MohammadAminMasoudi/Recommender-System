from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Set, Tuple
from .data import USER_COL, ITEM_COL

def make_item_pop(df_hist: pd.DataFrame):
    item_pop = df_hist.groupby(ITEM_COL)[USER_COL].nunique().sort_values(ascending=False)
    items = item_pop.index.values
    probs = (item_pop / item_pop.sum()).values if item_pop.sum() > 0 else None
    return items, probs

def sample_negatives(rng: np.random.Generator,
                     all_items,
                     pop_items,
                     pop_probs,
                     exclude: Set,
                     n: int) -> List[int]:
    if n <= 0:
        return []
    if pop_items is None or pop_probs is None or len(pop_items) == 0:
        pool = [it for it in all_items if it not in exclude]
        if not pool:
            return []
        return rng.choice(pool, size=min(n, len(pool)), replace=len(pool) < n).tolist()
    res, tries = [], 0
    while len(res) < n and tries < n * 20:
        cand = int(rng.choice(pop_items, p=pop_probs))
        if cand not in exclude:
            res.append(cand)
        tries += 1
    if len(res) < n:
        pool = [it for it in all_items if it not in exclude]
        if pool:
            take = min(n - len(res), len(pool))
            res.extend(rng.choice(pool, size=take, replace=len(pool) < take).tolist())
    return res[:n]

def expand_baskets_to_impressions(baskets_df: pd.DataFrame,
                                  df_hist: pd.DataFrame,
                                  neg_per_pos: int,
                                  seed: int = 42) -> pd.DataFrame:
    from .data import ITEM_COL, USER_COL
    all_items = sorted(df_hist[ITEM_COL].dropna().unique().tolist())
    pop_items, pop_probs = make_item_pop(df_hist)
    rng = np.random.default_rng(seed)
    rows = []
    for _, row in baskets_df.iterrows():
        u, t, pos_items = row[USER_COL], row["BasketTime"], list(set(row[ITEM_COL]))
        if not pos_items:
            continue
        exclude = set(pos_items)
        for i_pos in pos_items:
            rows.append((u, int(i_pos), t, 1))
            for i_neg in sample_negatives(rng, all_items, pop_items, pop_probs, exclude, neg_per_pos):
                rows.append((u, int(i_neg), t, 0))
    return pd.DataFrame(rows, columns=[USER_COL, ITEM_COL, "BasketTime", "label"])
