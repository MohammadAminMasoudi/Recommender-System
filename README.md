# Recommender-System

A complete collection of recommendation system models built on a real B2B sales dataset.

This repository contains two main projects:

1. **`deepfm_ctr/` – DeepFM CTR & next-basket rerankers (PyTorch + DeepCTR‑Torch)**  
- Next‑basket CTR reranking
- Last‑day (LOO) CTR reranking with Top‑K metrics
- Revenue‑aware evaluation
- Per‑customer‑group DeepFM models

2. **`als_hybrid_segment/` – ALS + Co‑occurrence + Popularity + Product‑segment reranker (PySpark)**  
   - Matrix factorization with implicit ALS
   - Co‑occurrence and time‑decayed popularity signals
   - Segment‑aware reranking using product semantic groups
   - Two‑stage hyper‑parameter tuning
   - Building reusable ALS candidate pools

The code is organised in small, reusable Python modules so that you can:

- Run end‑to‑end experiments from the command line
- Reuse the data preparation and metric code in notebooks
- Plug in new models while keeping the same evaluation pipeline

> **Note**: This repo assumes your raw data is stored in an Excel file like
> `Sales_Dist6_Tehran.xlsx` on Google Drive (or a local path) with columns such as
> `ShipToNum`, `MatNum`, `BillingDate`, `CustGrpName`, `MatName`, `BillingNetValue`,
> `BillingNetWeight`, etc. See the per‑project READMEs for details.

