import argparse
import time
from typing import Dict, Optional, Tuple

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
    smooth_full_dict, _ = compute_genre_smoothing(
        genre=genre, y=y, m=m, global_mean=gmean
    )

    return oof.to_numpy(), smooth_full_dict, gmean


def rf_ctor(seed):
    return RandomForestRegressor(
        n_estimators=500, random_state=seed, n_jobs=-1, max_features="sqrt"
    )


def compute_genre_smoothing(
    genre: pd.Series,
    y: pd.Series,
    m: float,
    global_mean: Optional[float] = None,
) -> Tuple[Dict[str, float], float]:
    """Compute smoothed genre means on the full dataset."""
    gmean = float(y.mean()) if global_mean is None else float(global_mean)
    gdf_full = pd.DataFrame({GENRE: genre, TARGET: y})
    agg_full = gdf_full.groupby(GENRE)[TARGET].agg(["mean", "size"])
    smooth_full = (agg_full["size"] * agg_full["mean"] + m * gmean) / (
        agg_full["size"] + m
    )
    return smooth_full.to_dict(), gmean


def fit_full_model(
    model_ctor,
    X: pd.DataFrame,
    target: np.ndarray,
    seed: int,
    start_message: str,
    end_message_template: str,
):
    """Fit a model on the full dataset with leakage-free scaling."""
    scaler = CustomColumnScaler()
    X_scaled = scaler.fit_transform(X)
    model = model_ctor(seed)
    print(start_message)
    t0 = time.time()
    model.fit(X_scaled, target)
    elapsed = time.time() - t0
    print(end_message_template.format(elapsed=elapsed))
    return model, scaler


def oof_model_preds_with_folds(
    model_ctor,
    X: pd.DataFrame,
    y: pd.Series,
    folds,
    seed: int,
    fit_full: bool = True,
):
    """
    Generic OOF predictions for a direct y->model (no residualization) on the
    data without the variable `track_genre`.
    Fits a new CustomColumnScaler per fold (no leakage).
    Also fits a full-data model + scaler for possible downstream inference.
    """
    idx = y.index
    oof = pd.Series(np.nan, index=idx, dtype=float)

    n_folds = len(folds)
    for fold_idx, (tr_idx, va_idx) in enumerate(folds, start=1):
        tr_ids = idx.take(tr_idx)
        va_ids = idx.take(va_idx)

        X_tr, X_va = X.loc[tr_ids], X.loc[va_ids]
        y_tr = y.loc[tr_ids]

        scaler = CustomColumnScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        model = model_ctor(seed)
        print(f"Starting training fold {fold_idx}/{n_folds} (seed={seed})...")
        t0 = time.time()
        model.fit(X_tr_s, y_tr.to_numpy())
        elapsed = time.time() - t0
        print(f"Finished training fold {fold_idx}/{n_folds} in {elapsed:.2f} sec.")
        oof.loc[va_ids] = model.predict(X_va_s)

    full_model = None
    full_scaler = None
    if fit_full:
        full_model, full_scaler = fit_full_model(
            model_ctor=model_ctor,
            X=X,
            target=y.to_numpy(),
            seed=seed,
            start_message="Starting full-data model training...",
            end_message_template="Finished full-data model training in {elapsed:.2f} sec.",
        )

    return oof.to_numpy(), full_model, full_scaler


def oof_residual_model_preds_with_folds(
    model_ctor,
    X: pd.DataFrame,
    y: pd.Series,
    mu_oof: np.ndarray,
    folds,
    seed: int,
    fit_full: bool = True,
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

    n_folds = len(folds)
    for fold_idx, (tr_idx, va_idx) in enumerate(folds, start=1):
        tr_ids = idx.take(tr_idx)
        va_ids = idx.take(va_idx)

        X_tr, X_va = X.loc[tr_ids], X.loc[va_ids]
        res_tr = residual[tr_idx]

        scaler = CustomColumnScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        model = model_ctor(seed)
        print(f"Starting residual training fold {fold_idx}/{n_folds} (seed={seed})...")
        t0 = time.time()
        model.fit(X_tr_s, res_tr)
        elapsed = time.time() - t0
        print(
            f"Finished residual training fold {fold_idx}/{n_folds} in {elapsed:.2f} sec."
        )
        res_pred_va = model.predict(X_va_s)
        oof_final.loc[va_ids] = mu_oof[va_idx] + res_pred_va

    full_model = None
    full_scaler = None
    if fit_full:
        full_model, full_scaler = fit_full_model(
            model_ctor=model_ctor,
            X=X,
            target=residual,
            seed=seed,
            start_message="Starting full-data residual model training...",
            end_message_template="Finished full-data residual model training in {elapsed:.2f} sec.",
        )

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

    run_cv = args.execution in ("cv", "both")
    run_train = args.execution in ("train", "both")

    folds = make_folds(y, n_splits=args.folds, seed=args.seed) if run_cv else None

    mu_oof = None
    genre_map_full: Optional[Dict[str, float]] = None
    gmean: Optional[float] = None

    oof_m1 = None
    m1_full = None
    m1_scaler = None
    r2_m1 = None
    pf_m1 = None

    oof_m2 = None
    m2_full = None
    m2_scaler = None
    r2_m2 = None
    pf_m2 = None
    r2_mu_only = None

    w_best = None
    r2_blend = None
    pf_blend = None

    if run_cv:
        if args.mode in ("m2", "ensemble"):
            mu_oof, genre_map_full_cv, gmean_cv = oof_genre_mean_with_folds(
                genre=genres, y=y, folds=folds, m=args.m2_m
            )
            genre_map_full = genre_map_full_cv
            gmean = gmean_cv
            r2_mu_only = r2_score(y, mu_oof)
            oof_m2, m2_full_cv, m2_scaler_cv = oof_residual_model_preds_with_folds(
                rf_ctor,
                X,
                y,
                mu_oof=mu_oof,
                folds=folds,
                seed=args.seed,
                fit_full=run_train,
            )
            r2_m2 = r2_score(y, oof_m2)
            pf_m2 = per_fold_scores(y.to_numpy(), oof_m2, folds)
            if run_train:
                m2_full = m2_full_cv
                m2_scaler = m2_scaler_cv
        if args.mode in ("m1", "ensemble"):
            oof_m1, m1_full_cv, m1_scaler_cv = oof_model_preds_with_folds(
                rf_ctor,
                X,
                y,
                folds=folds,
                seed=args.seed,
                fit_full=run_train,
            )
            r2_m1 = r2_score(y, oof_m1)
            pf_m1 = per_fold_scores(y.to_numpy(), oof_m1, folds)
            if run_train:
                m1_full = m1_full_cv
                m1_scaler = m1_scaler_cv
        if args.mode == "ensemble":
            if oof_m1 is None or oof_m2 is None:
                raise ValueError("Ensemble mode requires both M1 and M2 predictions.")
            w_best, _ = best_blend_weight(
                y_true=y.to_numpy(), p1=oof_m1, p2=oof_m2, step=0.01
            )
            oof_blend = np.clip(w_best * oof_m2 + (1 - w_best) * oof_m1, 0.0, 1.0)
            r2_blend = r2_score(y, oof_blend)
            pf_blend = per_fold_scores(y.to_numpy(), oof_blend, folds)

        if args.mode == "m1":
            print("Mode: M1 (no-genre) — Cross-validated (OOF) results")
            if r2_m1 is not None and pf_m1 is not None:
                print(
                    f"OOF R2 M1: {r2_m1:.5f}  | folds mean={pf_m1.mean():.5f} std={pf_m1.std():.5f}"
                )
            if run_train and args.show_importance and m1_full is not None:
                print_feature_importance(
                    m1_full, X.columns, model_name="M1 (full-data)"
                )
        elif args.mode == "m2":
            print("Mode: M2 (genre-residual) — Cross-validated (OOF) results")
            if r2_mu_only is not None:
                print(f"OOF R2 genre-only baseline (mu_genre): {r2_mu_only:.5f}")
            if r2_m2 is not None and pf_m2 is not None:
                print(
                    f"OOF R2 M2 (mu_genre + residual RF):   {r2_m2:.5f}  | folds mean={pf_m2.mean():.5f} std={pf_m2.std():.5f}"
                )
            if run_train and args.show_importance and m2_full is not None:
                print_feature_importance(
                    m2_full, X.columns, model_name="M2 residual (full-data)"
                )
        else:
            print("Mode: ENSEMBLE (M1 + M2) — Cross-validated (OOF) results")
            if w_best is not None:
                print(
                    f"Learned blend weight w on OOF (p = w*M2 + (1-w)*M1): {w_best:.2f}"
                )
            if r2_m1 is not None:
                print(f"OOF R2 M1:      {r2_m1:.5f}")
            if r2_m2 is not None:
                print(f"OOF R2 M2:      {r2_m2:.5f}")
            if r2_blend is not None and pf_blend is not None:
                print(
                    f"OOF R2 BLEND:   {r2_blend:.5f}  | folds mean={pf_blend.mean():.5f} std={pf_blend.std():.5f}"
                )
            if r2_mu_only is not None:
                print(f"(For reference) OOF R2 mu_genre baseline: {r2_mu_only:.5f}")
            if run_train and args.show_importance:
                if m1_full is not None:
                    print_feature_importance(
                        m1_full, X.columns, model_name="M1 (full-data)"
                    )
                if m2_full is not None:
                    print_feature_importance(
                        m2_full, X.columns, model_name="M2 residual (full-data)"
                    )

    if run_train:
        if not run_cv:
            if args.mode in ("m2", "ensemble"):
                genre_map_full, gmean = compute_genre_smoothing(
                    genre=genres, y=y, m=args.m2_m
                )
                mu_train = genres.map(genre_map_full).fillna(gmean).to_numpy()
                m2_full, m2_scaler = fit_full_model(
                    model_ctor=rf_ctor,
                    X=X,
                    target=y.to_numpy() - mu_train,
                    seed=args.seed,
                    start_message="Starting full-data residual model training...",
                    end_message_template=(
                        "Finished full-data residual model training in {elapsed:.2f} sec."
                    ),
                )
            else:
                mu_train = None
            if args.mode in ("m1", "ensemble"):
                m1_full, m1_scaler = fit_full_model(
                    model_ctor=rf_ctor,
                    X=X,
                    target=y.to_numpy(),
                    seed=args.seed,
                    start_message="Starting full-data model training...",
                    end_message_template=(
                        "Finished full-data model training in {elapsed:.2f} sec."
                    ),
                )
            if args.mode == "ensemble" and genre_map_full is None:
                genre_map_full, gmean = compute_genre_smoothing(
                    genre=genres, y=y, m=args.m2_m
                )
        elif args.mode in ("m2", "ensemble") and genre_map_full is None:
            genre_map_full, gmean = compute_genre_smoothing(
                genre=genres, y=y, m=args.m2_m
            )

        if args.mode == "ensemble" and w_best is None:
            train_mu = genres.map(genre_map_full).fillna(gmean).to_numpy()
            assert m1_full is not None and m1_scaler is not None
            assert m2_full is not None and m2_scaler is not None
            X_train_s_m1 = m1_scaler.transform(X)
            p1_train = m1_full.predict(X_train_s_m1)
            X_train_s_m2 = m2_scaler.transform(X)
            p2_train = train_mu + m2_full.predict(X_train_s_m2)
            w_best, _ = best_blend_weight(
                y_true=y.to_numpy(), p1=p1_train, p2=p2_train, step=0.01
            )
            print(f"Computed blend weight on training data: {w_best:.2f}")

        if args.show_importance and not run_cv:
            if args.mode in ("m1", "ensemble") and m1_full is not None:
                print_feature_importance(
                    m1_full, X.columns, model_name="M1 (full-data)"
                )
            if args.mode in ("m2", "ensemble") and m2_full is not None:
                print_feature_importance(
                    m2_full, X.columns, model_name="M2 residual (full-data)"
                )

        test_df = df_transformer.transform(pd.read_csv(f"{DATA_DIR}/test_data.csv"))
        row_ids = test_df.index.to_numpy()
        X_test = test_df.drop(columns=[TARGET, GENRE], errors="ignore")

        if args.mode == "m1":
            if m1_scaler is None or m1_full is None:
                raise ValueError("M1 full model not available for inference.")
            X_test_s = m1_scaler.transform(X_test)
            pred_norm = m1_full.predict(X_test_s)
        elif args.mode == "m2":
            if genre_map_full is None or gmean is None:
                genre_map_full, gmean = compute_genre_smoothing(
                    genre=genres, y=y, m=args.m2_m
                )
            mu_test = test_df[GENRE].map(genre_map_full).fillna(gmean).to_numpy()
            if m2_scaler is None or m2_full is None:
                raise ValueError("M2 full model not available for inference.")
            X_test_s = m2_scaler.transform(X_test)
            res_pred = m2_full.predict(X_test_s)
            pred_norm = mu_test + res_pred
        else:
            if w_best is None:
                raise ValueError("Blend weight unavailable for ensemble inference.")
            if (
                m1_scaler is None
                or m1_full is None
                or m2_scaler is None
                or m2_full is None
                or genre_map_full is None
                or gmean is None
            ):
                raise ValueError("Full models not available for ensemble inference.")
            X_test_s_m1 = m1_scaler.transform(X_test)
            p1 = m1_full.predict(X_test_s_m1)
            mu_test = test_df[GENRE].map(genre_map_full).fillna(gmean).to_numpy()
            X_test_s_m2 = m2_scaler.transform(X_test)
            p2 = mu_test + m2_full.predict(X_test_s_m2)
            pred_norm = np.clip(w_best * p2 + (1 - w_best) * p1, 0.0, 1.0)

        pred_clipped = np.clip(pred_norm, 0.0, 1.0) * 100.0

        submission = pd.DataFrame({"row_id": row_ids, "popularity": pred_clipped})
        submission.to_csv(f"{DATA_DIR}/submission.csv", index=False)
        print(f"Saved submission to {DATA_DIR}/submission.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="ensemble",
        choices=["ensemble", "m1", "m2"],
        help="Which model(s) to evaluate.",
    )
    parser.add_argument(
        "--execution",
        type=str,
        default="both",
        choices=["cv", "train", "both"],
        help="Which stages to run: cross-validation only, full training only, or both.",
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
