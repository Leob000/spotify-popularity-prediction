import argparse
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from xgboost import XGBRegressor  # ty:ignore[possibly-unbound-import]

from .data_processing import CustomColumnScaler, DataFrameTransformer

TARGET = "popularity"
GENRE = "track_genre"
DATA_DIR = "./src/data"


def parse_float_list(raw: str) -> List[float]:
    """Parse comma-separated floats (e.g. '50,100,150')."""
    if not raw:
        return []
    vals: List[float] = []
    for part in raw.split(","):
        s = part.strip()
        if not s:
            continue
        vals.append(float(s))
    return vals


def format_params_for_logging(params: Optional[Dict[str, Any]]) -> str:
    """Return a human-readable string for a hyperparameter mapping."""

    if not params:
        return "default parameters"
    parts = [f"{key}={value!r}" for key, value in sorted(params.items())]
    return ", ".join(parts)


def parse_model_names(raw: str) -> List[str]:
    """Parse a comma-separated list of model names."""

    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


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


@dataclass
class ModelSpec:
    """Container describing how to build and tune a regression model."""

    name: str
    ctor: Callable[..., Any]
    param_distributions: Dict[str, Any]
    scaler_kwargs: Dict[str, Any] = field(default_factory=dict)

    def instantiate(self, seed: int, params: Optional[Dict[str, Any]] = None) -> Any:
        """Instantiate the underlying estimator with the provided params."""
        params = params or {}
        return self.ctor(seed=seed, **params)


class PredefinedFoldCV:
    """Wrap a precomputed list of folds so it can be used by sklearn CV APIs."""

    def __init__(self, folds):
        self.folds = list(folds)

    def split(self, X, y=None, groups=None):
        return iter(self.folds)

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.folds)


def model_specs(seed: int) -> Dict[str, ModelSpec]:
    """Return the available model specifications keyed by short name."""

    specs = {
        "rf": ModelSpec(
            name="rf",
            ctor=lambda seed, **kw: RandomForestRegressor(
                random_state=seed, n_jobs=-1, **kw
            ),
            param_distributions={
                "n_estimators": [100, 300, 600],
                "max_depth": [None, 10, 16, 24],
                "max_features": ["sqrt", "log2", None],
                "min_samples_leaf": [1, 2, 4],
                "min_samples_split": [2, 5, 10],
            },
        ),
        "et": ModelSpec(
            name="et",
            ctor=lambda seed, **kw: ExtraTreesRegressor(
                random_state=seed, n_jobs=-1, **kw
            ),
            param_distributions={
                "n_estimators": [300, 600, 1000],
                "max_depth": [None, 12, 20, 28],
                "max_features": ["sqrt", "log2", None],
                "min_samples_leaf": [1, 2, 4],
                "min_samples_split": [2, 5, 10],
            },
        ),
        "hgb": ModelSpec(
            name="hgb",
            ctor=lambda seed, **kw: HistGradientBoostingRegressor(
                random_state=seed, **kw
            ),
            param_distributions={
                "max_depth": [None, 6, 10, 16],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_leaf_nodes": [None, 31, 63, 127],
                "l2_regularization": [0.0, 1e-4, 1e-3, 1e-2],
            },
        ),
        "gbr": ModelSpec(
            name="gbr",
            ctor=lambda seed, **kw: GradientBoostingRegressor(random_state=seed, **kw),
            param_distributions={
                "n_estimators": [200, 400, 800],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [2, 3, 4],
                "subsample": [0.6, 0.8, 1.0],
            },
        ),
        "enet": ModelSpec(
            name="enet",
            ctor=lambda seed, **kw: ElasticNet(random_state=seed, max_iter=5000, **kw),
            param_distributions={
                "alpha": np.logspace(-3, 1, 20),
                "l1_ratio": np.linspace(0.0, 1.0, 11),
            },
            scaler_kwargs={"extend_standard_scaling": True},
        ),
        "svr": ModelSpec(
            name="svr",
            ctor=lambda seed, **kw: SVR(**kw),
            param_distributions={
                "C": np.logspace(-2, 2, 10),
                "epsilon": np.logspace(-3, 0, 8),
                "gamma": ["scale", "auto"],
                "kernel": ["rbf"],
            },
            scaler_kwargs={"extend_standard_scaling": True},
        ),
        "knn": ModelSpec(
            name="knn",
            ctor=lambda seed, **kw: KNeighborsRegressor(**kw),
            param_distributions={
                "n_neighbors": list(range(3, 61, 2)),
                "weights": ["uniform", "distance"],
                "p": [1, 2],
            },
            scaler_kwargs={"extend_standard_scaling": True},
        ),
    }
    specs["xgb"] = ModelSpec(
        name="xgb",
        ctor=lambda seed, **kw: XGBRegressor(
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
            **kw,
        ),
        param_distributions={
            "n_estimators": [300, 600, 900],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [3, 6, 9],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 3, 5],
            "reg_lambda": [0.5, 1.0, 2.0],
            "reg_alpha": [0.0, 0.1, 0.5],
        },
    )
    specs["cat"] = ModelSpec(
        name="cat",
        ctor=lambda seed, **kw: CatBoostRegressor(
            random_seed=seed,
            verbose=0,
            allow_writing_files=False,
            **kw,
        ),
        param_distributions={
            "iterations": [500, 800, 1100],
            "learning_rate": [0.03, 0.05, 0.08],
            "depth": [6, 8, 10],
            "l2_leaf_reg": [1.0, 3.0, 5.0, 7.0],
            "bagging_temperature": [0.0, 0.25, 0.5, 1.0],
            "subsample": [0.66, 0.8, 1.0],
        },
    )
    return specs


def choose_refit_params(
    params_per_fold: List[Dict[str, Any]], fold_scores: List[float]
) -> Dict[str, Any]:
    """Aggregate per-fold best params into a single configuration for refitting.

    Preference order:
      1) Majority vote (mode) across folds for stability.
      2) Tie-breaker: params from the outer fold with the highest R².
      3) Fallback: the single most common params (even if frequency == 1).
    """

    if not params_per_fold:
        return {}

    counter: Counter = Counter(tuple(sorted(p.items())) for p in params_per_fold)
    (best_items, cnt), *_ = counter.most_common(1)
    if cnt > 1:
        return dict(best_items)

    if fold_scores and len(fold_scores) == len(params_per_fold):
        best_idx = int(np.argmax(fold_scores))
        return params_per_fold[best_idx]

    return dict(best_items)


def inner_cv_search(
    spec: ModelSpec,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    n_splits: int,
    n_iter: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Randomized hyperparameter search on the provided training split.
    If --inner-iter=0, only evaluates the defaults and skip the random search
    """

    if not spec.param_distributions:
        return {}

    inner_folds = make_folds(y_tr, n_splits=n_splits, seed=seed)
    estimator = Pipeline(
        [
            (
                "scaler",
                CustomColumnScaler(**spec.scaler_kwargs, output_as_pandas=True),
            ),
            ("model", spec.instantiate(seed=seed)),
        ]
    )
    param_distributions = {
        f"model__{key}": value for key, value in spec.param_distributions.items()
    }

    # Evaluate DEFAULT hyperparameters first on the inner CV
    default_start = time.time()
    default_scores = cross_val_score(
        estimator, X_tr, y_tr, scoring="r2", cv=PredefinedFoldCV(inner_folds), n_jobs=-1
    )
    default_mean = float(np.mean(default_scores))
    default_elapsed = time.time() - default_start
    print(
        f"    Inner CV default params: mean R2={default_mean:.5f} "
        f"(fit time={default_elapsed:.2f} sec over {n_splits} folds)",
        flush=True,
    )

    # Randomized search for the remaining budget (if any)
    best_params = {}
    best_score = default_mean

    effective_n_iter = max(0, int(n_iter))
    if effective_n_iter > 0:
        search_start = time.time()
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=effective_n_iter,
            scoring="r2",
            cv=PredefinedFoldCV(inner_folds),
            random_state=seed,
            n_jobs=-1,
            verbose=0,
            refit=True,
        )
        search.fit(X_tr, y_tr)
        elapsed = time.time() - search_start
        print(
            f"    Inner CV random search (n_iter={effective_n_iter}) done in {elapsed:.2f} sec "
            f"(best_score={search.best_score_:.5f})",
            flush=True,
        )
        # Keep the better of default vs. random-search best
        if float(search.best_score_) > best_score:
            best_score = float(search.best_score_)
            best_params = {
                key.replace("model__", "", 1): value
                for key, value in search.best_params_.items()
                if key.startswith("model__")
            }
    else:
        print(
            "    Skipping random search (n_iter=0); using default parameters.",
            flush=True,
        )

    return best_params


def nested_cv_oof(
    mode: str,
    specs: Dict[str, ModelSpec],
    X: pd.DataFrame,
    y: pd.Series,
    genres: pd.Series,
    folds,
    seed: int,
    inner_splits: int = 3,
    n_iter_small: int = 20,
    m_grid: Optional[List[float]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run nested CV and collect OOF predictions per model spec."""

    if mode not in {"m1", "m2"}:
        raise ValueError("mode must be 'm1' or 'm2'")

    results: Dict[str, Dict[str, Any]] = {}
    idx = y.index
    y_np = y.to_numpy()

    n_specs = len(specs)
    for spec_idx, (name, spec) in enumerate(specs.items(), start=1):
        print(
            f"Starting nested CV for spec '{name}' ({spec_idx}/{n_specs})",
            flush=True,
        )
        oof = pd.Series(np.nan, index=idx, dtype=float)
        best_params_fold: List[Dict[str, Any]] = []
        fold_scores: List[float] = []
        best_m_per_fold: List[float] = []
        m_fold_scores: Dict[float, List[Tuple[int, float]]] = defaultdict(list)

        for fold_idx, (tr_idx, va_idx) in enumerate(folds, start=1):
            fold_start = time.time()
            print(
                f"  Fold {fold_idx}/{len(folds)}: preparing data",
                flush=True,
            )
            tr_ids = idx.take(tr_idx)
            va_ids = idx.take(va_idx)

            X_tr, X_va = X.loc[tr_ids], X.loc[va_ids]
            y_tr, y_va = y.loc[tr_ids], y.loc[va_ids]

            if mode == "m1":
                # --- M1: standard inner-search + train on y directly
                y_inner_series = y_tr
                print(
                    "    Running inner CV search "
                    f"(n_iter={n_iter_small}, inner_splits={inner_splits})",
                    flush=True,
                )
                best = inner_cv_search(
                    spec,
                    X_tr=X_tr,
                    y_tr=y_inner_series,
                    n_splits=inner_splits,
                    n_iter=n_iter_small,
                    seed=seed + fold_idx,
                )
                best_params_fold.append(best)

                scaler = CustomColumnScaler(output_as_pandas=True, **spec.scaler_kwargs)
                X_tr_s = scaler.fit_transform(X_tr)
                X_va_s = scaler.transform(X_va)

                model = spec.instantiate(seed=seed + fold_idx, params=best)
                params_text = format_params_for_logging(best)
                print(
                    f"    Training {model.__class__.__name__} (spec '{name}') "
                    f"fold {fold_idx}/{len(folds)} with {params_text}",
                    flush=True,
                )
                model.fit(X_tr_s, y_tr.to_numpy())
                pred_va = model.predict(X_va_s)
                fold_score = r2_score(y_va.to_numpy(), pred_va)
                oof.loc[va_ids] = pred_va
                fold_scores.append(float(fold_score))
            else:
                #  we may search over multiple 'm' smoothing values
                m_candidates = m_grid if m_grid else [150.0]
                best_fold_score = -1e18
                best_fold_params: Dict[str, Any] = {}
                best_fold_pred_va: Optional[np.ndarray] = None
                best_fold_m: Optional[float] = None

                for m_val in m_candidates:
                    # Build leakage-free μ_oof on outer-train via inner folds
                    inner_folds = make_folds(
                        y_tr, n_splits=max(inner_splits, 2), seed=seed + fold_idx
                    )
                    mu_oof_train, _, _ = oof_genre_mean_with_folds(
                        genre=genres.loc[tr_ids],
                        y=y_tr,
                        folds=inner_folds,
                        m=m_val,
                    )
                    residual_tr = y_tr.to_numpy() - mu_oof_train
                    y_inner_series = pd.Series(residual_tr, index=y_tr.index)
                    genre_map_full_tr, gmean_tr = compute_genre_smoothing(
                        genre=genres.loc[tr_ids], y=y_tr, m=m_val
                    )
                    mu_va = (
                        genres.loc[va_ids]
                        .map(genre_map_full_tr)
                        .fillna(gmean_tr)
                        .to_numpy()
                    )

                    print(
                        "    Running inner CV search "
                        f"(n_iter={n_iter_small}, inner_splits={inner_splits}, m={m_val})",
                        flush=True,
                    )
                    best = inner_cv_search(
                        spec,
                        X_tr=X_tr,
                        y_tr=y_inner_series,
                        n_splits=inner_splits,
                        n_iter=n_iter_small,
                        seed=seed + fold_idx,
                    )

                    scaler = CustomColumnScaler(
                        output_as_pandas=True, **spec.scaler_kwargs
                    )
                    X_tr_s = scaler.fit_transform(X_tr)
                    X_va_s = scaler.transform(X_va)

                    model = spec.instantiate(seed=seed + fold_idx, params=best)
                    params_text = format_params_for_logging(best)
                    print(
                        f"    Training {model.__class__.__name__} (spec '{name}') "
                        f"fold {fold_idx}/{len(folds)} with {params_text}, m={m_val}",
                        flush=True,
                    )
                    model.fit(X_tr_s, residual_tr)
                    pred_va = mu_va + model.predict(X_va_s)
                    fold_score = float(r2_score(y_va.to_numpy(), pred_va))
                    m_key = float(m_val)
                    m_fold_scores[m_key].append((fold_idx, fold_score))

                    if fold_score > best_fold_score:
                        best_fold_score = fold_score
                        best_fold_params = best
                        best_fold_pred_va = pred_va
                        best_fold_m = float(m_val)

                # Commit best m and predictions for this outer fold
                assert best_fold_pred_va is not None and best_fold_m is not None
                oof.loc[va_ids] = best_fold_pred_va
                fold_scores.append(best_fold_score)
                best_params_fold.append(best_fold_params)
                best_m_per_fold.append(best_fold_m)

            fold_elapsed = time.time() - fold_start
            print(
                f"  Fold {fold_idx}/{len(folds)} completed in {fold_elapsed:.2f} sec "
                f"(R2={fold_scores[-1]:.5f})",
                flush=True,
            )

        oof_np = oof.to_numpy()
        overall_r2 = r2_score(y_np, oof_np)
        pf = per_fold_scores(y_np, oof_np, folds)
        pf_mean = float(pf.mean())
        pf_std = float(pf.std())
        refit_params = choose_refit_params(best_params_fold, fold_scores)

        refit_m_majority: Optional[float] = None
        refit_m_avg: Optional[float] = None
        refit_m_source: Optional[str] = None
        m_grid_summary: Dict[float, Dict[str, Any]] = {}

        if mode == "m2":
            if best_m_per_fold:
                counter = Counter(best_m_per_fold)
                most_common_m, count = counter.most_common(1)[0]
                if count > 1:
                    refit_m_majority = float(most_common_m)
                elif fold_scores:
                    refit_m_majority = float(
                        best_m_per_fold[int(np.argmax(fold_scores))]
                    )

            if m_fold_scores:
                print(
                    "    Recap of m-grid performance across outer folds:",
                    flush=True,
                )
                best_mean = -np.inf
                for m_val in sorted(m_fold_scores.keys()):
                    fold_records = m_fold_scores[m_val]
                    fold_values = [score for _, score in fold_records]
                    mean_r2 = float(np.mean(fold_values))
                    std_r2 = float(np.std(fold_values))
                    min_r2 = float(np.min(fold_values))
                    max_r2 = float(np.max(fold_values))
                    fold_details = ", ".join(
                        f"fold{idx}={score:.5f}" for idx, score in fold_records
                    )
                    print(
                        f"      m={m_val:g}: mean={mean_r2:.5f}, std={std_r2:.5f}, "
                        f"min={min_r2:.5f}, max={max_r2:.5f} | {fold_details}",
                        flush=True,
                    )
                    m_grid_summary[m_val] = {
                        "fold_scores": fold_values,
                        "mean_r2": mean_r2,
                        "std_r2": std_r2,
                        "min_r2": min_r2,
                        "max_r2": max_r2,
                    }
                    if mean_r2 > best_mean:
                        best_mean = mean_r2
                        refit_m_avg = float(m_val)
                if refit_m_avg is not None:
                    refit_m_source = "mean"
                    print(
                        f"      -> Best mean R2 achieved with m={refit_m_avg:g}",
                        flush=True,
                    )

            if refit_m_source is None and refit_m_majority is not None:
                refit_m_source = "majority"

        refit_m_selected: Optional[float] = None
        if mode == "m2":
            refit_m_selected = (
                refit_m_avg if refit_m_avg is not None else refit_m_majority
            )

        results[name] = {
            "oof": oof_np,
            "best_params_per_fold": best_params_fold,
            "fold_r2": fold_scores,
            "mean_r2": pf_mean,
            "std_r2": pf_std,
            "overall_r2": float(overall_r2),
            "refit_params": refit_params,
            # For M2 only (present when mode == 'm2'):
            "best_m_per_fold": best_m_per_fold if mode == "m2" else [],
            "refit_m": refit_m_selected,
            "refit_m_source": refit_m_source if mode == "m2" else None,
            "m_grid_summary": m_grid_summary if mode == "m2" else {},
        }
        extra = ""
        if mode == "m2" and refit_m_selected is not None and refit_m_source:
            extra = f" | selected m={refit_m_selected:g} ({refit_m_source})"
        print(
            f"[{mode.upper()}] {name}: OOF R2={overall_r2:.5f} | folds mean={pf_mean:.5f} std={pf_std:.5f}{extra}"
        )
        print(f"Finished nested CV for spec '{name}'.", flush=True)

    return results


def select_best_spec(
    results: Dict[str, Dict[str, Any]],
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Return the spec name and result dict with the best overall R²."""

    if not results:
        return None, None
    best_name, best_result = max(
        results.items(), key=lambda item: item[1].get("overall_r2", float("-inf"))
    )
    return best_name, best_result


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
    scaler_kwargs: Optional[Dict[str, Any]] = None,
    log_model_name: Optional[str] = None,
    log_params: Optional[Dict[str, Any]] = None,
):
    """Fit a model on the full dataset with leakage-free scaling."""
    scaler_kwargs = scaler_kwargs or {}
    scaler = CustomColumnScaler(output_as_pandas=True, **scaler_kwargs)
    X_scaled = scaler.fit_transform(X)
    model = model_ctor(seed)
    params_text = ""
    if log_model_name is not None:
        params_text = (
            f" ({log_model_name} params: {format_params_for_logging(log_params)})"
        )
    print(f"{start_message}{params_text}", flush=True)
    t0 = time.time()
    model.fit(X_scaled, target)
    elapsed = time.time() - t0
    print(end_message_template.format(elapsed=elapsed), flush=True)
    return model, scaler


def oof_model_preds_with_folds(
    model_ctor,
    X: pd.DataFrame,
    y: pd.Series,
    folds,
    seed: int,
    fit_full: bool = True,
    scaler_kwargs: Optional[Dict[str, Any]] = None,
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

        scaler = CustomColumnScaler(output_as_pandas=True, **(scaler_kwargs or {}))
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
            scaler_kwargs=scaler_kwargs,
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
    scaler_kwargs: Optional[Dict[str, Any]] = None,
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

        scaler = CustomColumnScaler(output_as_pandas=True, **(scaler_kwargs or {}))
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
            scaler_kwargs=scaler_kwargs,
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
    """Compute per-fold R² scores from the provided predictions."""
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
    """Execute the training and evaluation workflow based on parsed CLI args."""
    df_transformer = DataFrameTransformer()
    df = df_transformer.transform(pd.read_csv(f"{DATA_DIR}/train_data.csv"))

    y = df[TARGET]
    genres = df[GENRE]
    X = df.drop(columns=[TARGET, GENRE])

    run_cv = args.execution in ("cv", "both")
    run_train = args.execution in ("train", "both")

    folds = make_folds(y, n_splits=args.folds, seed=args.seed) if run_cv else None

    specs_lookup = model_specs(args.seed)
    m1_model_names = parse_model_names(args.m1_models)
    m2_model_names = parse_model_names(args.m2_models)

    if args.mode in ("m1", "ensemble") and not m1_model_names:
        raise ValueError("Provide at least one model spec via --m1-models for M1.")
    if args.mode in ("m2", "ensemble") and not m2_model_names:
        raise ValueError("Provide at least one model spec via --m2-models for M2.")

    def resolve_specs(names: List[str]) -> Dict[str, ModelSpec]:
        selected: Dict[str, ModelSpec] = {}
        for name in names:
            if name not in specs_lookup:
                available = ", ".join(sorted(specs_lookup.keys()))
                raise ValueError(
                    f"Unknown model spec '{name}'. Available specs: {available}"
                )
            selected[name] = specs_lookup[name]
        return selected

    selected_m1_specs = (
        resolve_specs(m1_model_names) if args.mode in ("m1", "ensemble") else {}
    )
    selected_m2_specs = (
        resolve_specs(m2_model_names) if args.mode in ("m2", "ensemble") else {}
    )

    mu_oof: Optional[np.ndarray] = None
    genre_map_full: Optional[Dict[str, float]] = None
    gmean: Optional[float] = None

    oof_m1: Optional[np.ndarray] = None
    oof_m2: Optional[np.ndarray] = None

    r2_m1: Optional[float] = None
    r2_m2: Optional[float] = None
    r2_blend: Optional[float] = None
    r2_mu_only: Optional[float] = None

    pf_m1_stats: Optional[Tuple[float, float]] = None
    pf_m2_stats: Optional[Tuple[float, float]] = None
    pf_blend: Optional[np.ndarray] = None

    best_m1_spec: Optional[ModelSpec] = None
    best_m1_spec_name: Optional[str] = None
    best_m1_params: Dict[str, Any] = {}

    best_m2_spec: Optional[ModelSpec] = None
    best_m2_spec_name: Optional[str] = None
    best_m2_params: Dict[str, Any] = {}
    best_m2_m: Optional[float] = None

    m1_full = None
    m1_scaler = None
    m2_full = None
    m2_scaler = None
    w_best: Optional[float] = None
    mu_train: Optional[np.ndarray] = None

    ensemble_strategy: Optional[str] = None
    if run_cv:
        y_np = y.to_numpy()

        if args.mode in ("m2", "ensemble"):
            nested_results_m2 = nested_cv_oof(
                mode="m2",
                specs=selected_m2_specs,
                X=X,
                y=y,
                genres=genres,
                folds=folds,
                seed=args.seed,
                inner_splits=args.inner_splits,
                n_iter_small=args.inner_iter,
                m_grid=(parse_float_list(args.m2_m_grid) or [args.m2_m]),
            )
            best_m2_spec_name, best_m2_result = select_best_spec(nested_results_m2)
            if best_m2_result is not None and best_m2_spec_name is not None:
                best_m2_spec = selected_m2_specs[best_m2_spec_name]
                best_m2_params = best_m2_result.get("refit_params", {})
                oof_m2 = best_m2_result["oof"]
                r2_m2 = best_m2_result["overall_r2"]
                # Selected m for full-data training
                best_m2_m = best_m2_result.get("refit_m", args.m2_m)
                # For reference baseline, compute μ-only OOF R² using selected m
                mu_oof, genre_map_full_cv, gmean_cv = oof_genre_mean_with_folds(
                    genre=genres, y=y, folds=folds, m=best_m2_m
                )
                genre_map_full = genre_map_full_cv
                gmean = gmean_cv
                r2_mu_only = r2_score(y_np, mu_oof)
                pf_m2_stats = (
                    best_m2_result["mean_r2"],
                    best_m2_result["std_r2"],
                )

        if args.mode in ("m1", "ensemble"):
            nested_results_m1 = nested_cv_oof(
                mode="m1",
                specs=selected_m1_specs,
                X=X,
                y=y,
                genres=genres,
                folds=folds,
                seed=args.seed,
                inner_splits=args.inner_splits,
                n_iter_small=args.inner_iter,
            )
            best_m1_spec_name, best_m1_result = select_best_spec(nested_results_m1)
            if best_m1_result is not None and best_m1_spec_name is not None:
                best_m1_spec = selected_m1_specs[best_m1_spec_name]
                best_m1_params = best_m1_result.get("refit_params", {})
                oof_m1 = best_m1_result["oof"]
                r2_m1 = best_m1_result["overall_r2"]
                pf_m1_stats = (
                    best_m1_result["mean_r2"],
                    best_m1_result["std_r2"],
                )

        if args.mode == "ensemble" and oof_m1 is not None and oof_m2 is not None:
            w_best, _ = best_blend_weight(y_true=y_np, p1=oof_m1, p2=oof_m2, step=0.01)
            oof_blend = np.clip(w_best * oof_m2 + (1 - w_best) * oof_m1, 0.0, 1.0)
            r2_blend = r2_score(y_np, oof_blend)
            pf_blend = per_fold_scores(y_np, oof_blend, folds)
            # Select best inference strategy based on OOF R²
            candidates = [("m1", r2_m1), ("m2", r2_m2), ("blend", r2_blend)]
            ensemble_strategy = max(
                candidates, key=lambda x: (x[1] if x[1] is not None else -np.inf)
            )[0]

        if args.mode == "m1":
            print("Mode: M1 (no-genre) — Cross-validated (OOF) results")
            if best_m1_spec_name:
                print(f"Best spec: {best_m1_spec_name}")
                if best_m1_params:
                    print(f"Refit params: {best_m1_params}")
            if r2_m1 is not None and pf_m1_stats is not None:
                mean_r2, std_r2 = pf_m1_stats
                print(
                    f"OOF R2 M1 ({best_m1_spec_name}): {r2_m1:.5f}  | folds mean={mean_r2:.5f} std={std_r2:.5f}"
                )
        elif args.mode == "m2":
            print("Mode: M2 (genre-residual) — Cross-validated (OOF) results")
            if best_m2_spec_name:
                print(f"Best spec: {best_m2_spec_name}")
                if best_m2_params:
                    print(f"Refit params: {best_m2_params}")
            if r2_mu_only is not None:
                print(f"OOF R2 genre-only baseline (mu_genre): {r2_mu_only:.5f}")
            if r2_m2 is not None and pf_m2_stats is not None:
                mean_r2, std_r2 = pf_m2_stats
                print(
                    f"OOF R2 M2 ({best_m2_spec_name}): {r2_m2:.5f}  | folds mean={mean_r2:.5f} std={std_r2:.5f}"
                )
        else:
            print("Mode: ENSEMBLE (M1 + M2) — Cross-validated (OOF) results")
            if best_m1_spec_name:
                print(f"Best M1 spec: {best_m1_spec_name}")
            if best_m2_spec_name:
                print(f"Best M2 spec: {best_m2_spec_name}")
            if w_best is not None:
                print(
                    f"Learned blend weight w on OOF (p = w*M2 + (1-w)*M1): {w_best:.2f}"
                )
            if r2_m1 is not None and pf_m1_stats is not None:
                mean_r2, std_r2 = pf_m1_stats
                print(
                    f"OOF R2 M1 ({best_m1_spec_name}): {r2_m1:.5f}  | folds mean={mean_r2:.5f} std={std_r2:.5f}"
                )
            if r2_m2 is not None and pf_m2_stats is not None:
                mean_r2, std_r2 = pf_m2_stats
                print(
                    f"OOF R2 M2 ({best_m2_spec_name}): {r2_m2:.5f}  | folds mean={mean_r2:.5f} std={std_r2:.5f}"
                )
            if r2_blend is not None and pf_blend is not None:
                print(
                    f"OOF R2 BLEND:   {r2_blend:.5f}  | folds mean={pf_blend.mean():.5f} std={pf_blend.std():.5f}"
                )
            if r2_mu_only is not None:
                print(f"(For reference) OOF R2 mu_genre baseline: {r2_mu_only:.5f}")
            if ensemble_strategy is not None:
                print(f"Selected inference strategy: {ensemble_strategy}")

    if run_train:
        if args.mode in ("m2", "ensemble"):
            if best_m2_spec is None and selected_m2_specs:
                fallback_name = m2_model_names[0]
                best_m2_spec = selected_m2_specs[fallback_name]
                best_m2_spec_name = fallback_name
                best_m2_params = {}
                if not run_cv:
                    print(
                        f"Train-only execution: defaulting M2 to '{fallback_name}' with baseline params."
                    )
            if best_m2_spec is None:
                raise ValueError("No model spec available for M2 training.")
            # Always compute genre smoothing with the selected M2 'm'
            chosen_m = best_m2_m or args.m2_m
            genre_map_full, gmean = compute_genre_smoothing(
                genre=genres, y=y, m=chosen_m
            )
            mu_train = genres.map(genre_map_full).fillna(gmean).to_numpy()

            def m2_ctor(seed: int, spec=best_m2_spec, params=best_m2_params):
                return spec.instantiate(seed=seed, params=params)

            m2_full, m2_scaler = fit_full_model(
                model_ctor=m2_ctor,
                X=X,
                target=y.to_numpy() - mu_train,
                seed=args.seed,
                start_message="Starting full-data residual model training...",
                end_message_template=(
                    "Finished full-data residual model training in {elapsed:.2f} sec."
                ),
                scaler_kwargs=best_m2_spec.scaler_kwargs,
                log_model_name=best_m2_spec_name or best_m2_spec.name,
                log_params=best_m2_params,
            )
        else:
            mu_train = None

        if args.mode in ("m1", "ensemble"):
            if best_m1_spec is None and selected_m1_specs:
                fallback_name = m1_model_names[0]
                best_m1_spec = selected_m1_specs[fallback_name]
                best_m1_spec_name = fallback_name
                best_m1_params = {}
                if not run_cv:
                    print(
                        f"Train-only execution: defaulting M1 to '{fallback_name}' with baseline params."
                    )
            if best_m1_spec is None:
                raise ValueError("No model spec available for M1 training.")

            def m1_ctor(seed: int, spec=best_m1_spec, params=best_m1_params):
                return spec.instantiate(seed=seed, params=params)

            m1_full, m1_scaler = fit_full_model(
                model_ctor=m1_ctor,
                X=X,
                target=y.to_numpy(),
                seed=args.seed,
                start_message="Starting full-data model training...",
                end_message_template=(
                    "Finished full-data model training in {elapsed:.2f} sec."
                ),
                scaler_kwargs=best_m1_spec.scaler_kwargs,
                log_model_name=best_m1_spec_name or best_m1_spec.name,
                log_params=best_m1_params,
            )

        if args.mode == "ensemble" and w_best is None:
            # Ensure mapping uses the selected M2 'm'
            if genre_map_full is None or gmean is None:
                chosen_m = best_m2_m or args.m2_m
                genre_map_full, gmean = compute_genre_smoothing(
                    genre=genres, y=y, m=chosen_m
                )
            train_mu = genres.map(genre_map_full).fillna(gmean).to_numpy()
            if m1_full is None or m2_full is None:
                raise ValueError("Full models unavailable for computing blend weight.")
            X_train_m1 = m1_scaler.transform(X) if m1_scaler is not None else X
            p1_train = m1_full.predict(X_train_m1)
            X_train_m2 = m2_scaler.transform(X) if m2_scaler is not None else X
            p2_train = train_mu + m2_full.predict(X_train_m2)
            w_best, _ = best_blend_weight(
                y_true=y.to_numpy(), p1=p1_train, p2=p2_train, step=0.01
            )
            print(f"Computed blend weight on training data: {w_best:.2f}")

        if args.show_importance:
            if args.mode in ("m1", "ensemble") and m1_full is not None:
                model_name = "M1"
                if best_m1_spec_name:
                    model_name += f" ({best_m1_spec_name}, full-data)"
                print_feature_importance(m1_full, X.columns, model_name=model_name)
            if args.mode in ("m2", "ensemble") and m2_full is not None:
                model_name = "M2 residual"
                if best_m2_spec_name:
                    model_name += f" ({best_m2_spec_name}, full-data)"
                print_feature_importance(m2_full, X.columns, model_name=model_name)

        test_df = df_transformer.transform(pd.read_csv(f"{DATA_DIR}/test_data.csv"))
        row_ids = test_df.index.to_numpy()
        X_test = test_df.drop(columns=[TARGET, GENRE], errors="ignore")

        if args.mode == "m1":
            if m1_full is None:
                raise ValueError("M1 full model not available for inference.")
            X_test_s = m1_scaler.transform(X_test) if m1_scaler is not None else X_test
            pred_norm = m1_full.predict(X_test_s)
        elif args.mode == "m2":
            if genre_map_full is None or gmean is None:
                genre_map_full, gmean = compute_genre_smoothing(
                    genre=genres,
                    y=y,
                    m=(best_m2_m or args.m2_m),
                )
            mu_test = test_df[GENRE].map(genre_map_full).fillna(gmean).to_numpy()
            if m2_full is None:
                raise ValueError("M2 full model not available for inference.")
            X_test_s = m2_scaler.transform(X_test) if m2_scaler is not None else X_test
            res_pred = m2_full.predict(X_test_s)
            pred_norm = mu_test + res_pred
        else:
            # In ENSEMBLE mode, use the strategy selected on OOF (if available).
            if ensemble_strategy == "m1":
                if m1_full is None:
                    raise ValueError(
                        "M1 full model not available for ensemble inference."
                    )
                X_test_s_m1 = (
                    m1_scaler.transform(X_test) if m1_scaler is not None else X_test
                )
                pred_norm = m1_full.predict(X_test_s_m1)
            elif ensemble_strategy == "m2":
                if m2_full is None or genre_map_full is None or gmean is None:
                    raise ValueError(
                        "M2 full model or genre mapping unavailable for ensemble inference."
                    )
                # genre_map_full already computed with the selected 'm'
                mu_test = test_df[GENRE].map(genre_map_full).fillna(gmean).to_numpy()
                X_test_s_m2 = (
                    m2_scaler.transform(X_test) if m2_scaler is not None else X_test
                )
                res_pred = m2_full.predict(X_test_s_m2)
                pred_norm = mu_test + res_pred
            else:
                # Default to blending when no strategy was selected via CV
                if (
                    w_best is None
                    or m1_full is None
                    or m2_full is None
                    or genre_map_full is None
                    or gmean is None
                ):
                    raise ValueError(
                        "Ensemble inference requires fitted models and blend weight."
                    )
                X_test_s_m1 = (
                    m1_scaler.transform(X_test) if m1_scaler is not None else X_test
                )
                p1 = m1_full.predict(X_test_s_m1)
                mu_test = test_df[GENRE].map(genre_map_full).fillna(gmean).to_numpy()
                X_test_s_m2 = (
                    m2_scaler.transform(X_test) if m2_scaler is not None else X_test
                )
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
        "--m1-models",
        type=str,
        default="rf",
        help=(
            "Comma-separated model specs to evaluate for M1 (no-genre). "
            "Available options: rf, et, hgb, gbr, enet, svr, knn, xgb, cat."
        ),
    )
    parser.add_argument(
        "--m2-models",
        type=str,
        default="rf",
        help=(
            "Comma-separated model specs to evaluate for M2 (genre-residual). "
            "Available options: rf, et, hgb, gbr, enet, svr, knn, xgb, cat."
        ),
    )
    parser.add_argument(
        "--m2-m-grid",
        type=str,
        default="",
        help="Comma-separated values for M2 smoothing m to search (e.g. '50,100,150,300'). "
        "If provided, overrides --m2-m during CV and the best m is used for full training/inference.",
    )
    parser.add_argument(
        "--inner-splits",
        type=int,
        default=3,
        help="Number of inner CV splits for hyperparameter search.",
    )
    parser.add_argument(
        "--inner-iter",
        type=int,
        default=20,
        help="Number of RandomizedSearch iterations per model spec.",
    )
    parser.add_argument(
        "--m2-m",
        type=float,
        default=150.0,
        help="m hyperparameter for genre-mean smoothing",
    )
    parser.add_argument(
        "--show-importance",
        action="store_true",
        help="Print top-20 feature importances from full-data models",
    )
    args = parser.parse_args()
    start_time = time.time()
    main(args)
    print(f"Done in {(time.time() - start_time) / 60.0:.2f} minutes.")
