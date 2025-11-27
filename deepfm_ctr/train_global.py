"""Train a global DeepFM next‑basket CTR reranker.

Usage:
    python -m deepfm_ctr.train_global --data-path /path/to/Sales_Dist6_Tehran.xlsx
"""
import argparse
import numpy as np
import pandas as pd

from .config import DeepFMConfig
from .data import load_raw, make_next_basket_split
from .impressions import expand_baskets_to_impressions
from .features import add_features_hist, fit_label_encoders, apply_encode
from .model import create_model, to_model_inputs, evaluate_topk

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True, help="Path to Excel file")
    args = ap.parse_args()
    cfg = DeepFMConfig()

    df = load_raw(args.data_path)
    train_b, val_b, test_b = make_next_basket_split(df)

    # history = all baskets up to test split
    hist_keys = set(map(tuple, train_b[["user_id", "BasketTime"]].values)) |                 set(map(tuple, val_b[["user_id", "BasketTime"]].values))
    df_hist = df.copy()
    df_hist["__key"] = list(zip(df_hist["user_id"], df_hist["BasketTime"]))
    df_hist = df_hist[df_hist["__key"].isin(hist_keys)].drop(columns="__key")

    train_imp = expand_baskets_to_impressions(train_b, df_hist, cfg.neg_per_pos_train, cfg.seed)
    val_imp   = expand_baskets_to_impressions(val_b,   df_hist, cfg.neg_per_pos_train, cfg.seed)

    train_x = add_features_hist(df_hist, train_imp)
    val_x   = add_features_hist(df_hist, val_imp)

    encoders, index_maps, vocab_sizes = fit_label_encoders(train_x, val_x)
    train_x = apply_encode(train_x, index_maps)
    val_x   = apply_encode(val_x, index_maps)

    model, feature_names, device = create_model(vocab_sizes, cfg.embed_dim, cfg.lr)

    X_train = to_model_inputs(train_x, feature_names)
    X_val   = to_model_inputs(val_x, feature_names)
    y_train = train_x["label"].values
    y_val   = val_x["label"].values

    history = model.fit(X_train, y_train,
                        batch_size=cfg.batch_size,
                        epochs=cfg.epochs,
                        verbose=2,
                        validation_data=(X_val, y_val))

    # Build eval candidates from test baskets using history stats only
    from .impressions import make_item_pop, sample_negatives
    from .data import USER_COL, ITEM_COL
    all_items = sorted(df_hist[ITEM_COL].dropna().unique().tolist())
    pop_items, pop_probs = make_item_pop(df_hist)
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for _, row in test_b.iterrows():
        u, t, pos_items = row[USER_COL], row["BasketTime"], list(set(row[ITEM_COL]))
        if not pos_items:
            continue
        exclude = set(pos_items)
        from .impressions import sample_negatives
        negs = sample_negatives(rng, all_items, pop_items, pop_probs,
                                exclude, max(len(pos_items)*cfg.neg_per_pos_test, 100))
        cand = set(pos_items) | set(negs)
        for it in cand:
            rows.append((u, int(it), t, 1 if it in pos_items else 0))
    test_imp = pd.DataFrame(rows, columns=[USER_COL, ITEM_COL, "BasketTime", "label"])
    test_x = add_features_hist(df_hist, test_imp)
    test_x = apply_encode(test_x, index_maps)
    X_test = to_model_inputs(test_x, feature_names)
    scores = model.predict(X_test, batch_size=8192).reshape(-1)
    scored = test_x.loc[:, [USER_COL, ITEM_COL, "BasketTime", "label"]].copy()
    scored["score"] = scores
    res = evaluate_topk(scored, k=cfg.topk)
    print(f"\n=== Global next‑basket Top-{cfg.topk} metrics ===")
    for k, v in res.items():
        print(f"{k:>10}: {v:.4f}")

if __name__ == "__main__":
    main()
