# deepfm_ctr – CTR & Next‑basket Rerankers (DeepFM)

This module contains PyTorch / DeepCTR‑Torch implementations of several CTR
reranking scenarios on transactional sales data.

## Features

- Daily basketisation per user
- Train/validation/test chronological split
- Negative sampling with popularity‑based proposal distribution
- Feature engineering (item aggregates, recency, categorical embeddings)
- **DeepFM** model for binary CTR
- Top‑K ranking metrics: Precision@K, Recall@K, Hit‑rate@K, NDCG@K, MAP@K
- LOO (last‑basket) evaluation with revenue‑aware metrics
- Optional per‑customer‑group models (e.g. supermarket, restaurant, etc.)

## Entry points

- `python -m deepfm_ctr.train_global --data-path /path/to/Sales_Dist6_Tehran.xlsx`
- `python -m deepfm_ctr.train_loo --data-path /path/to/Sales_Dist6_Tehran.xlsx`
- `python -m deepfm_ctr.train_per_group --data-path /path/to/Sales_Dist6_Tehran.xlsx`

See the docstrings inside the modules for details.
