from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Dict

from pyspark.sql import functions as F, Window
from pyspark.ml.recommendation import ALS

from .config import USER_COL, ITEM_COL, DATE_COL, RATING_COL, SEGMENT_COL, HybridConfig

def make_loo_split_basket_inclusive(pdf: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    g = pdf[[USER_COL, ITEM_COL, DATE_COL]].copy()
    g["_row_id"] = np.arange(len(g))
    g["__date_parsed"] = pd.to_datetime(g[DATE_COL], errors="coerce")
    any_valid = g.groupby(USER_COL)["__date_parsed"].apply(lambda s: s.notna().any()).astype(bool)
    u_valid = set(any_valid[any_valid].index)
    u_invalid = set(any_valid[~any_valid].index)
    gv = g[g[USER_COL].isin(u_valid)].copy()
    last_date = gv.groupby(USER_COL)["__date_parsed"].transform("max")
    last_rows_valid = gv[gv["__date_parsed"] == last_date].copy()
    gi = g[g[USER_COL].isin(u_invalid)].copy()
    if not gi.empty:
        max_idx = gi.groupby(USER_COL)["_row_id"].transform("max")
        last_rows_invalid = gi[gi["_row_id"] == max_idx].copy()
    else:
        last_rows_invalid = gi
    last_rows = pd.concat([last_rows_valid, last_rows_invalid], ignore_index=True)
    last_rows["_item_int"] = last_rows[ITEM_COL].astype(int)
    test_baskets = (last_rows.groupby(USER_COL)["_item_int"]
                    .apply(lambda s: sorted(set(s.tolist())))
                    .reset_index(name="test_items"))
    test_baskets["test_size"] = test_baskets["test_items"].apply(len)
    test_baskets = test_baskets[test_baskets["test_size"] > 0].copy()
    train_mask = ~g["_row_id"].isin(last_rows["_row_id"])
    train_df = g.loc[train_mask, [USER_COL, ITEM_COL, DATE_COL]].copy()
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL], errors="coerce")
    return train_df, test_baskets

def train_als_scores(spark, train_pdf: pd.DataFrame, topn: int) :
    sdf = spark.createDataFrame(train_pdf[[USER_COL, ITEM_COL, RATING_COL]])
    sdf = sdf.select(
        F.col(USER_COL).alias("user"),
        F.col(ITEM_COL).alias("item"),
        F.col(RATING_COL).alias("rating"),
    )
    als = ALS(
        implicitPrefs=True,
        rank=10,
        maxIter=10,
        regParam=0.01,
        alpha=40.0,
        userCol="user",
        itemCol="item",
        ratingCol="rating",
        coldStartStrategy="drop",
        seed=42,
    )
    model = als.fit(sdf)
    recs = model.recommendForAllUsers(topn)
    exploded = recs.select("user", F.explode("recommendations").alias("rec"))
    als_scores = exploded.select(
        F.col("user").alias("user_id"),
        F.col("rec.item").alias("item"),
        F.col("rec.rating").alias("als_score"),
    )
    return als_scores

def cooccurrence_summary(train_pdf: pd.DataFrame):
    from collections import defaultdict
    from itertools import combinations
    if train_pdf.empty:
        return pd.DataFrame(columns=["item", "co_sum", "co_max", "co_avg"])
    dfc = train_pdf[[USER_COL, ITEM_COL, DATE_COL]].dropna().copy()
    dfc[ITEM_COL] = dfc[ITEM_COL].astype(int)
    dfc[DATE_COL] = pd.to_datetime(dfc[DATE_COL], errors="coerce")
    dfc = dfc.dropna(subset=[DATE_COL])
    dfc["__basket_day"] = dfc[DATE_COL].dt.floor("D")
    baskets = dfc.groupby([USER_COL, "__basket_day"])[ITEM_COL].apply(lambda s: sorted(set(s))).to_dict()
    co_counts = defaultdict(int)
    for items in baskets.values():
        if len(items) < 2:
            continue
        for a, b in combinations(items, 2):
            if a > b:
                a, b = b, a
            co_counts[(a, b)] += 1
    if not co_counts:
        return pd.DataFrame(columns=["item", "co_sum", "co_max", "co_avg"])
    co_df = pd.DataFrame([(a, b, c) for (a, b), c in co_counts.items()],
                         columns=["item1", "item2", "count"])
    co_pairs = pd.concat(
        [
            co_df[["item1", "count"]].rename(columns={"item1": "item", "count": "cnt"}),
            co_df[["item2", "count"]].rename(columns={"item2": "item", "count": "cnt"}),
        ],
        ignore_index=True,
    )
    agg = (
        co_pairs.groupby("item")["cnt"]
        .agg(co_sum="sum", co_max="max", co_avg="mean")
        .reset_index()
    )
    return agg

def popularity_from_train(train_pdf: pd.DataFrame, decay: float = 0.9):
    if train_pdf.empty or "BillingNetValue" not in train_pdf.columns:
        return pd.DataFrame(columns=[ITEM_COL, "popularity_score"])
    g = train_pdf[[ITEM_COL, DATE_COL, "BillingNetValue"]].dropna().copy()
    g[DATE_COL] = pd.to_datetime(g[DATE_COL], errors="coerce")
    g = g.dropna(subset=[DATE_COL])
    ref = g[DATE_COL].max()
    g["days_diff"] = (ref - g[DATE_COL]).dt.days
    g["w"] = decay ** g["days_diff"]
    g["value_w"] = g["BillingNetValue"] * g["w"]
    pop = g.groupby(ITEM_COL)["value_w"].sum().reset_index(name="popularity_score")
    return pop

def hybrid_topk_from_train(spark,
                           als_scores,
                           co_summary_pdf,
                           pop_pdf,
                           cfg: HybridConfig):
    if als_scores is None:
        return None
    if co_summary_pdf is None or co_summary_pdf.empty:
        co_sdf = spark.createDataFrame([], "item long, co_sum double, co_max double, co_avg double")
    else:
        co_sdf = spark.createDataFrame(co_summary_pdf)
    if pop_pdf is None or pop_pdf.empty:
        pop_sdf = spark.createDataFrame([], "item long, popularity_score double")
    else:
        pop_sdf = (
            spark.createDataFrame(pop_pdf)
            .select(F.col(ITEM_COL).alias("item"),
                    F.col("popularity_score").cast("double").alias("popularity_score"))
            .groupBy("item")
            .agg(F.max("popularity_score").alias("popularity_score"))
        )
    joined = (
        als_scores
        .join(co_sdf, on="item", how="left")
        .join(pop_sdf, on="item", how="left")
        .fillna({"co_sum": 0.0, "co_max": 0.0, "co_avg": 0.0, "popularity_score": 0.0})
        .withColumn("co_signal", F.log1p(F.col("co_sum")))
        .withColumn("pop_signal", F.log1p(F.col("popularity_score")))
    )
    w_user = Window.partitionBy("user_id")
    eps = F.lit(1e-9)
    def mm(c):
        return (F.col(c) - F.min(c).over(w_user)) / (F.max(c).over(w_user) - F.min(c).over(w_user) + eps)
    scored = (
        joined
        .withColumn("als_n", mm("als_score"))
        .withColumn("co_n", mm("co_signal"))
        .withColumn("pop_n", mm("pop_signal"))
        .withColumn("hybrid_score",
                    cfg.weight_als * F.col("als_n") +
                    cfg.weight_cooc * F.col("co_n") +
                    cfg.weight_pop * F.col("pop_n"))
    )
    uniq = scored.groupBy("user_id", "item").agg(
        F.max("hybrid_score").alias("hybrid_score"),
        F.max("als_n").alias("als_n"),
    )
    w = Window.partitionBy("user_id").orderBy(
        F.col("hybrid_score").desc(),
        F.col("als_n").desc(),
        F.col("item").asc(),
    )
    topk = (
        uniq.withColumn("rank", F.row_number().over(w))
        .filter(F.col("rank") <= cfg.candidates)
        .select("user_id", "item", "hybrid_score", "als_n", "rank")
    )
    return topk

def build_item_segment_map(pdf_raw: pd.DataFrame):
    m = (
        pdf_raw[[ITEM_COL, SEGMENT_COL]]
        .copy()
        .dropna(subset=[ITEM_COL])
        .fillna({SEGMENT_COL: "Unknown"})
        .drop_duplicates()
    )
    m[ITEM_COL] = m[ITEM_COL].astype(int)
    m[SEGMENT_COL] = m[SEGMENT_COL].astype(str)
    return m

def user_segment_softmax_probs(train_pdf: pd.DataFrame,
                               item_seg_map_pdf: pd.DataFrame,
                               users_to_expand,
                               cfg: HybridConfig):
    # simplified version of the notebook logic
    if train_pdf is None or train_pdf.empty:
        base = item_seg_map_pdf[[SEGMENT_COL]].dropna().copy()
        if base.empty:
            return pd.DataFrame(columns=[USER_COL, SEGMENT_COL, "p_seg"])
        base["score"] = 1.0
        pri = base.groupby(SEGMENT_COL)["score"].sum()
        pri = (pri / pri.sum()).reset_index(name="p_seg")
        rows = []
        for u in users_to_expand:
            for _, r in pri.iterrows():
                rows.append({USER_COL: u, SEGMENT_COL: r[SEGMENT_COL], "p_seg": float(r["p_seg"])})
        return pd.DataFrame(rows)
    g = train_pdf[[USER_COL, ITEM_COL, DATE_COL, "BillingNetValue"]].dropna().copy()
    g = g.merge(item_seg_map_pdf, on=ITEM_COL, how="left").dropna(subset=[SEGMENT_COL])
    g[DATE_COL] = pd.to_datetime(g[DATE_COL], errors="coerce")
    g = g.dropna(subset=[DATE_COL])
    ref = g[DATE_COL].max()
    g["days_diff"] = (ref - g[DATE_COL]).dt.days
    g["w"] = (cfg.decay_pop ** g["days_diff"]).astype(float)
    g["val_w"] = g["BillingNetValue"].astype(float) * g["w"]
    scores = g.groupby([USER_COL, SEGMENT_COL])["val_w"].sum().reset_index(name="score")
    def _softmax(x):
        x = x.astype(float)
        x = (x - x.max()) / max(cfg.temp_softmax, 1e-6)
        ex = np.exp(x)
        return ex / ex.sum()
    scores["p_seg"] = scores.groupby(USER_COL)["score"].transform(_softmax)
    out = scores[[USER_COL, SEGMENT_COL, "p_seg"]].copy()
    return out

def rerank_with_segment(spark, candidate_sdf, user_seg_pdf, item_seg_map_pdf, cfg: HybridConfig):
    if candidate_sdf is None:
        return None
    seg_map_pdf = item_seg_map_pdf.rename(columns={ITEM_COL: "item", SEGMENT_COL: "segment"})
    seg_map_sdf = spark.createDataFrame(seg_map_pdf)
    usp_pdf = user_seg_pdf.rename(columns={USER_COL: "user_id", SEGMENT_COL: "segment"})
    usp_sdf = spark.createDataFrame(usp_pdf[["user_id", "segment", "p_seg"]])
    j = (
        candidate_sdf.join(seg_map_sdf, on="item", how="left")
        .join(usp_sdf, on=["user_id", "segment"], how="left")
        .fillna({"p_seg": 0.0})
    )
    j = j.withColumn(
        "final_score",
        (1.0 - cfg.lambda_seg) * F.col("hybrid_score")
        + cfg.lambda_seg * cfg.seg_strength * F.col("p_seg"),
    )
    w = Window.partitionBy("user_id").orderBy(
        F.col("final_score").desc(),
        F.col("hybrid_score").desc(),
        F.col("als_n").desc(),
        F.col("item").asc(),
    )
    out = (
        j.withColumn("rank", F.row_number().over(w))
        .filter(F.col("rank") <= cfg.topk)
        .select("user_id", "item", "final_score", "rank")
    )
    return out
