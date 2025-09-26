import argparse
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold

from .data_processing import CustomColumnScaler, DataFrameTransformer

TARGET = "popularity"
GENRE = "track_genre"
DATA_DIR = "./src/data"


def y_deciles(y: pd.Series, q: int = 10) -> np.ndarray:
    """Integer decile labels (0..q-1) for stratification."""
    bins = pd.qcut(y, q=q, duplicates="drop")
    return bins.cat.codes.to_numpy()


def make_folds(y: pd.Series, n_splits: int, seed: int):
    """Return list of (tr_idx, va_idx) using stratified deciles."""
    y_bins = y_deciles(y, q=10)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy_X = np.zeros(len(y), dtype=int)
    return list(skf.split(dummy_X, y_bins))


def oof_genre_mean_with_folds(genre: pd.Series, y: pd.Series, folds, m: float = 150.0):
    """
    Compute leakage-free out-of-fold (OOF) smoothed genre means and a full-data
    smoothed mapping for inference.
    Smoothing formula: (genre_size * genre_mean + m * global_mean) / (genre_size + m)

    Behaviour:
      - For each CV fold, compute per-genre empirical means and counts using
        only the training rows for that fold (prevents leakage).
      - Shrink those per-genre means toward the global mean using a simple
        additive-smoothing formula controlled by hyperparameter m.
      - Assign the fold-specific smoothed genre means to the fold's validation
        rows (these are the OOF predictions).
      - Finally compute the same smoothed mapping on the full dataset for
        production/inference use.

    Returns:
        oof_array: numpy array of shape (n_rows,) of OOF predictions for the
            estimated mean per genre, smoothed
        smooth_full_dict: dict mapping genre -> smoothed mean computed on FULL data (not CV)
        global_mean: float global mean of y
    """
    idx = y.index
    oof = pd.Series(np.nan, index=idx, dtype=float)
    gmean = float(y.mean())

    for tr_idx, va_idx in folds:
        # Convert positional fold indices into original index labels
        tr_ids = idx.take(tr_idx)
        va_ids = idx.take(va_idx)

        # Build a small training-only DataFrame (genre, target) to compute stats
        gdf = pd.DataFrame({GENRE: genre.loc[tr_ids], TARGET: y.loc[tr_ids]})

        # Per-genre empirical mean and count on training data for this fold
        agg = gdf.groupby(GENRE)[TARGET].agg(["mean", "size"])

        # Shrink genre mean toward global mean gmean:
        smooth = (agg["size"] * agg["mean"] + m * gmean) / (agg["size"] + m)

        # Map smoothed means to validation rows; unseen genres -> global mean
        oof.loc[va_ids] = genre.loc[va_ids].map(smooth).fillna(gmean).values

    # Full-data mapping (useful for inference on new data)
    gdf_full = pd.DataFrame({GENRE: genre, TARGET: y})
    agg_full = gdf_full.groupby(GENRE)[TARGET].agg(["mean", "size"])
    smooth_full = (agg_full["size"] * agg_full["mean"] + m * gmean) / (
        agg_full["size"] + m
    )

    return oof.to_numpy(), smooth_full.to_dict(), gmean


def rf_ctor(seed):
    return RandomForestRegressor(
        n_estimators=500, random_state=seed, n_jobs=-1, max_features="sqrt"
    )


def oof_model_preds_with_folds(
    model_ctor, X: pd.DataFrame, y: pd.Series, folds, seed: int
):
    """
    Generic OOF predictions for a direct y->model (no residualization) on the
    data without the variable `track_genre`.
    Fits a new CustomColumnScaler per fold (no leakage).
    Also fits a full-data model + scaler for possible downstream inference.
    """
    idx = y.index
    oof = pd.Series(np.nan, index=idx, dtype=float)

    for tr_idx, va_idx in folds:
        tr_ids = idx.take(tr_idx)
        va_ids = idx.take(va_idx)

        X_tr, X_va = X.loc[tr_ids], X.loc[va_ids]
        y_tr = y.loc[tr_ids]

        scaler = CustomColumnScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        model = model_ctor(seed)
        model.fit(X_tr_s, y_tr.to_numpy())
        oof.loc[va_ids] = model.predict(X_va_s)

    # full-data fit
    full_scaler = CustomColumnScaler()
    X_s = full_scaler.fit_transform(X)
    full_model = model_ctor(seed)
    full_model.fit(X_s, y.to_numpy())

    return oof.to_numpy(), full_model, full_scaler


def oof_residual_model_preds_with_folds(
    model_ctor, X: pd.DataFrame, y: pd.Series, mu_oof: np.ndarray, folds, seed: int
):
    """
    Out-of-fold (OOF) residual modelling with leakage-free scaling.

    Fit a model to the residuals `residual = y - mu_oof` using the provided
    `folds`. For each fold a fresh `CustomColumnScaler` and a new model
    (created via `model_ctor`) are fitted on the training rows only; residual
    predictions are made on the validation rows and then the corresponding
    `mu_oof` values are added back to form the final OOF predictions.

    Returns a tuple `(oof_final, full_model, full_scaler)` where `oof_final`
    are the OOF predictions aligned with `y`, and `full_model`/`full_scaler`
    are fitted on the entire dataset for downstream inference.
    """
    idx = y.index
    residual = y.to_numpy() - mu_oof
    oof_final = pd.Series(np.nan, index=idx, dtype=float)

    for tr_idx, va_idx in folds:
        tr_ids = idx.take(tr_idx)
        va_ids = idx.take(va_idx)

        X_tr, X_va = X.loc[tr_ids], X.loc[va_ids]
        res_tr = residual[tr_idx]

        scaler = CustomColumnScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        model = model_ctor(seed)
        model.fit(X_tr_s, res_tr)
        res_pred_va = model.predict(X_va_s)
        oof_final.loc[va_ids] = mu_oof[va_idx] + res_pred_va

    # full-data residual fit
    full_scaler = CustomColumnScaler()
    X_s = full_scaler.fit_transform(X)
    full_model = model_ctor(seed)
    full_model.fit(X_s, residual)

    return oof_final.to_numpy(), full_model, full_scaler


def best_blend_weight(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray, step=0.01):
    """Grid-search convex weight w in [0,1] maximizing R² for w*p2 + (1-w)*p1."""
    ws = np.arange(0.0, 1.0 + 1e-12, step)
    best_w, best_r2 = 0.5, -1e9
    for w in ws:
        pred = w * p2 + (1 - w) * p1
        r2 = r2_score(y_true, pred)
        if r2 > best_r2:
            best_r2, best_w = r2, w
    return float(best_w), float(best_r2)


def per_fold_scores(y: np.ndarray, preds: np.ndarray, folds):
    scores = []
    for _, va_idx in folds:
        scores.append(r2_score(y[va_idx], preds[va_idx]))
    return np.array(scores)


def print_feature_importance(model, feature_names, model_name):
    """Prints sorted feature importances for a fitted model (if available)."""
    if not hasattr(model, "feature_importances_"):
        print(f"\n{model_name} has no feature_importances_.")
        return
    importances = model.feature_importances_
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_df = importance_df.sort_values("importance", ascending=False)
    print(f"\nTop 20 feature importances for {model_name}:")
    print(importance_df.head(20).to_string(index=False))


def main(args):
    df_transformer = DataFrameTransformer()
    df = df_transformer.transform(pd.read_csv(f"{DATA_DIR}/train_data.csv"))

    y = df[TARGET]
    genres = df[GENRE]
    X = df.drop(columns=[TARGET, GENRE])

    folds = make_folds(y, n_splits=args.folds, seed=args.seed)

    # Train
    if args.mode in ("m2", "ensemble"):
        mu_oof, genre_map_full, gmean = oof_genre_mean_with_folds(
            genre=genres, y=y, folds=folds, m=args.m2_m
        )
        # genre-only baseline CV score
        r2_mu_only = r2_score(y, mu_oof)
        # residual model OOF preds
        oof_m2, m2_full, m2_scaler = oof_residual_model_preds_with_folds(
            rf_ctor, X, y, mu_oof=mu_oof, folds=folds, seed=args.seed
        )
        r2_m2 = r2_score(y, oof_m2)
        pf_m2 = per_fold_scores(y.to_numpy(), oof_m2, folds)

    if args.mode in ("m1", "ensemble"):
        oof_m1, m1_full, m1_scaler = oof_model_preds_with_folds(
            rf_ctor, X, y, folds=folds, seed=args.seed
        )
        r2_m1 = r2_score(y, oof_m1)
        pf_m1 = per_fold_scores(y.to_numpy(), oof_m1, folds)

    # Report
    if args.mode == "m1":
        print("Mode: M1 (no-genre) — Cross-validated (OOF) results")
        print(
            f"OOF R2 M1: {r2_m1:.5f}  | folds mean={pf_m1.mean():.5f} std={pf_m1.std():.5f}"
        )
        if args.show_importance:
            print_feature_importance(m1_full, X.columns, model_name="M1 (full-data)")
    elif args.mode == "m2":
        print("Mode: M2 (genre-residual) — Cross-validated (OOF) results")
        print(f"OOF R2 genre-only baseline (mu_genre): {r2_mu_only:.5f}")
        print(
            f"OOF R2 M2 (mu_genre + residual RF):   {r2_m2:.5f}  | folds mean={pf_m2.mean():.5f} std={pf_m2.std():.5f}"
        )
        if args.show_importance:
            print_feature_importance(
                m2_full, X.columns, model_name="M2 residual (full-data)"
            )
    else:
        # ensemble: learn weight on OOF only, then score OOF blend
        w_best, r2_blend_train = best_blend_weight(
            y_true=y.to_numpy(), p1=oof_m1, p2=oof_m2, step=0.01
        )
        oof_blend = np.clip(w_best * oof_m2 + (1 - w_best) * oof_m1, 0.0, 1.0)
        r2_blend = r2_score(y, oof_blend)
        pf_blend = per_fold_scores(y.to_numpy(), oof_blend, folds)

        print("Mode: ENSEMBLE (M1 + M2) — Cross-validated (OOF) results")
        print(f"Learned blend weight w on OOF (p = w*M2 + (1-w)*M1): {w_best:.2f}")
        print(f"OOF R2 M1:      {r2_m1:.5f}")
        print(f"OOF R2 M2:      {r2_m2:.5f}")
        print(
            f"OOF R2 BLEND:   {r2_blend:.5f}  | folds mean={pf_blend.mean():.5f} std={pf_blend.std():.5f}"
        )
        print(f"(For reference) OOF R2 mu_genre baseline: {r2_score(y, mu_oof):.5f}")
        if args.show_importance:
            print_feature_importance(m1_full, X.columns, model_name="M1 (full-data)")
            print_feature_importance(
                m2_full, X.columns, model_name="M2 residual (full-data)"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="ensemble",
        choices=["ensemble", "m1", "m2"],
        help="Which model(s) to evaluate.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument(
        "--m2_m",
        type=float,
        default=150.0,
        help="m hyperparameter for genre-mean smoothing",
    )
    parser.add_argument(
        "--show_importance",
        action="store_true",
        help="Print top-20 feature importances from full-data models",
    )
    args = parser.parse_args()
    start_time = time.time()
    main(args)
    print(f"Done in {(time.time() - start_time) / 60.0:.2f} minutes.")
