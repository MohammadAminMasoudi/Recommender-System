"""Run the ALS+hybrid+segment reranker per customer group.

This is a compact, script‑friendly version of your long Colab notebook.
"""
import argparse
import pandas as pd

from .config import HybridConfig, USER_COL, ITEM_COL
from .data import load_raw, split_by_custgrp
from .spark_session import get_spark
from .hybrid import (
    make_loo_split_basket_inclusive,
    train_als_scores,
    cooccurrence_summary,
    popularity_from_train,
    hybrid_topk_from_train,
    build_item_segment_map,
    user_segment_softmax_probs,
    rerank_with_segment,
)
from .metrics import eval_loo_metrics_basket

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True)
    args = ap.parse_args()
    cfg = HybridConfig()
    df = load_raw(args.data_path)
    groups = split_by_custgrp(df)
    spark = get_spark("HybridRecommender")  # one shared session
    all_rows = []
    for gname, pdf_raw in groups.items():
        print(f"\n=== Group: {gname} (rows={len(pdf_raw)}) ===")
        train_pdf_raw, test_baskets = make_loo_split_basket_inclusive(pdf_raw)
        if test_baskets.empty:
            print("[skip] empty test baskets.")
            continue
        # Build rating‑augmented train
        clean = pdf_raw[[USER_COL, ITEM_COL, "BillingDate", "BillingNetValue", "combined_rating"]].copy()
        clean["BillingDate"] = pd.to_datetime(clean["BillingDate"], errors="coerce")
        train_pdf_raw["BillingDate"] = pd.to_datetime(train_pdf_raw["BillingDate"], errors="coerce")
        train_pdf = train_pdf_raw.merge(
            clean[[USER_COL, ITEM_COL, "BillingDate", "combined_rating", "BillingNetValue"]],
            on=[USER_COL, ITEM_COL, "BillingDate"],
            how="inner",
        )
        if train_pdf.empty:
            print("[skip] empty train.")
            continue
        als_scores = train_als_scores(spark, train_pdf, topn=cfg.candidates)
        co_summary_pdf = cooccurrence_summary(train_pdf)
        pop_pdf = popularity_from_train(train_pdf, decay=cfg.decay_pop)
        item_seg_map_pdf = build_item_segment_map(pdf_raw)
        eval_users_list = test_baskets[USER_COL].dropna().unique().tolist()
        user_seg_pdf = user_segment_softmax_probs(train_pdf, item_seg_map_pdf, eval_users_list, cfg)
        cand_sdf = hybrid_topk_from_train(spark, als_scores, co_summary_pdf, pop_pdf, cfg)
        topk_final = rerank_with_segment(spark, cand_sdf, user_seg_pdf, item_seg_map_pdf, cfg)
        metrics = eval_loo_metrics_basket(spark, topk_final, test_baskets, k=cfg.topk)
        metrics["group"] = gname
        all_rows.append(metrics)
        print("Metrics:", metrics)
    if all_rows:
        import pandas as pd
        res = pd.DataFrame(all_rows).sort_values("ndcg_at_k", ascending=False)
        print("\n=== Summary ===")
        print(res.to_string(index=False))
        res.to_csv("als_hybrid_group_metrics.csv", index=False)
        print("Saved → als_hybrid_group_metrics.csv")


if __name__ == "__main__":
    main()
