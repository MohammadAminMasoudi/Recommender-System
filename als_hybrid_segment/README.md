# als_hybrid_segment – ALS + Co‑occurrence + Popularity + Segment‑aware Reranker

This module implements the PySpark pipeline that you built in Colab:

- Build per‑customer‑group dataframes
- Compute a hybrid recommendation score that mixes:
  - Implicit ALS
  - Co‑occurrence signals
  - Time‑decayed popularity
- Rerank candidates using user‑specific product‑segment preferences
- Evaluate with LOO (last‑basket) metrics and revenue metrics
- Build reusable ALS candidate pools
- Run a two‑stage hyper‑parameter search

The implementation here is slightly more compact than the original notebook but
preserves the same logic.

## Entry point

Run a single full experiment over all groups:

```bash
python -m als_hybrid_segment.run_all --data-path /path/to/Sales_Dist6_Tehran.xlsx
```

Run just Stage‑1 tuning or Stage‑2 tuning:

```bash
python -m als_hybrid_segment.tune_stage1 --data-path /path/to/Sales_Dist6_Tehran.xlsx
python -m als_hybrid_segment.tune_stage2 --data-path /path/to/Sales_Dist6_Tehran.xlsx
```

Check the module docstrings for details.
