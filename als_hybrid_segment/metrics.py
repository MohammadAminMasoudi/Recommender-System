from __future__ import annotations
import numpy as np
import pandas as pd
from pyspark.sql import functions as F, Window
from pyspark.sql.types import DoubleType
from .config import USER_COL, ITEM_COL

@F.udf(DoubleType())
def _idcg_udf(test_size: int, k: int):
    try:
        import math
        r = int(test_size or 0)
        kk = int(k or 0)
        m = max(0, min(r, kk))
        s = 0.0
        for i in range(1, m + 1):
            s += 1.0 / math.log2(i + 1.0)
        return float(s)
    except Exception:
        return 0.0

def eval_loo_metrics_basket(spark, topk_sdf, test_baskets_pdf: pd.DataFrame, k: int = 10):
    if (topk_sdf is None) or (test_baskets_pdf is None) or test_baskets_pdf.empty:
        return None
    tb = test_baskets_pdf.copy()
    tb["test_items"] = tb["test_items"].apply(lambda lst: [int(x) for x in (lst or [])])
    tb["test_size"] = tb["test_items"].apply(len)
    test_sdf = spark.createDataFrame(tb[[USER_COL, "test_items", "test_size"]]).withColumnRenamed(USER_COL, "user_id")
    j = (
        topk_sdf.join(test_sdf, on="user_id", how="inner")
        .withColumn(
            "rel",
            F.when(F.array_contains(F.col("test_items"), F.col("item")), F.lit(1.0)).otherwise(F.lit(0.0)),
        )
    )
    w_rank = Window.partitionBy("user_id").orderBy(F.col("rank").asc()).rowsBetween(
        Window.unboundedPreceding, Window.currentRow
    )
    j2 = (
        j.withColumn("cum_rel", F.sum("rel").over(w_rank))
        .withColumn("prec_at_i", F.col("cum_rel") / F.col("rank"))
        .withColumn("ap_contrib", F.col("prec_at_i") * F.col("rel"))
        .withColumn("gain", F.col("rel") / F.log2(F.col("rank") + F.lit(1.0)))
    )
    per_user = j2.groupBy("user_id").agg(
        F.sum("rel").alias("hits"),
        F.sum("gain").alias("dcg"),
        F.sum("ap_contrib").alias("ap_num"),
        F.max("test_size").alias("test_size"),
    )
    per_user = (
        per_user.withColumn("idcg", _idcg_udf(F.col("test_size"), F.lit(int(k))))
        .withColumn("precision_at_k", F.col("hits") / F.lit(float(k)))
        .withColumn(
            "recall_at_k",
            F.when(F.col("test_size") > 0, F.col("hits") / F.col("test_size")).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "hit_rate_at_k",
            F.when(F.col("hits") > 0, F.lit(1.0)).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "ndcg_at_k",
            F.when(F.col("idcg") > 0, F.col("dcg") / F.col("idcg")).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "ap_at_k",
            F.when(
                F.col("test_size") > 0,
                F.col("ap_num") / F.least(F.col("test_size"), F.lit(int(k))),
            ).otherwise(F.lit(0.0)),
        )
    )
    metrics = (
        per_user.agg(
            F.avg("precision_at_k").alias("precision_at_k"),
            F.avg("recall_at_k").alias("recall_at_k"),
            F.avg("hit_rate_at_k").alias("hit_rate_at_k"),
            F.avg("ndcg_at_k").alias("ndcg_at_k"),
            F.avg("ap_at_k").alias("map_at_k"),
        )
        .toPandas()
        .iloc[0]
        .to_dict()
    )
    metrics["n_users_eval"] = per_user.count()
    return metrics
