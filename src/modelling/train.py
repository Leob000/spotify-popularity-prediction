import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from .constants import BAGGING_MODELS, BOOSTING_MODELS, DATA_DIR, GENRE, TARGET
from .data_processing import DataFrameTransformer
from .train_utils import (
    ModelSpec,
    best_blend_weight,
    compute_genre_smoothing,
    fit_full_model,
    make_folds,
    model_specs,
    nested_cv_oof,
    oof_genre_mean_with_folds,
    parse_float_list,
    parse_model_names,
    per_fold_scores,
    print_feature_importance,
    select_best_spec,
)


def main(args):
    """Execute the training and evaluation workflow based on parsed CLI args."""
    df_transformer = DataFrameTransformer()

    train_path = os.path.join(DATA_DIR, "train_data.csv")
    if not os.path.exists(train_path):
        print(f"Train data not found. Please place `train_data.csv` at `{train_path}`")
        sys.exit(1)
    df = df_transformer.transform(pd.read_csv(train_path))

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
    m2_selection_type = "single"
    m2_blend_weight: Optional[float] = None
    best_m2_components: Dict[str, Dict[str, Any]] = {}

    m1_full = None
    m1_scaler = None
    m2_full = None
    m2_scaler = None
    w_best: Optional[float] = None
    mu_train: Optional[np.ndarray] = None

    candidate_params_from_m2: Dict[str, List[Dict[str, Any]]] = {}
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
            for spec_name, spec_result in nested_results_m2.items():
                params = spec_result.get("refit_params") or {}
                if params:
                    candidate_params_from_m2[spec_name] = [dict(params)]
            m2_blend_candidate: Optional[Dict[str, Any]] = None
            if args.m2_blend_bag_boost:
                bagging_best: Optional[Tuple[str, Dict[str, Any]]] = None
                boosting_best: Optional[Tuple[str, Dict[str, Any]]] = None
                for candidate in sorted(BAGGING_MODELS):
                    if candidate in nested_results_m2:
                        candidate_result = nested_results_m2[candidate]
                        if (
                            bagging_best is None
                            or candidate_result["overall_r2"]
                            > bagging_best[1]["overall_r2"]
                        ):
                            bagging_best = (candidate, candidate_result)
                for candidate in sorted(BOOSTING_MODELS):
                    if candidate in nested_results_m2:
                        candidate_result = nested_results_m2[candidate]
                        if (
                            boosting_best is None
                            or candidate_result["overall_r2"]
                            > boosting_best[1]["overall_r2"]
                        ):
                            boosting_best = (candidate, candidate_result)
                if bagging_best and boosting_best:
                    bag_name, bag_res = bagging_best
                    boost_name, boost_res = boosting_best
                    bag_oof = bag_res["oof"]
                    boost_oof = boost_res["oof"]
                    weight_candidate, r2_candidate = best_blend_weight(
                        y_true=y_np, p1=bag_oof, p2=boost_oof, step=0.01
                    )
                    blend_oof = np.clip(
                        weight_candidate * boost_oof
                        + (1.0 - weight_candidate) * bag_oof,
                        0.0,
                        1.0,
                    )
                    blend_pf = per_fold_scores(y_np, blend_oof, folds)
                    bag_spec = selected_m2_specs[bag_name]
                    boost_spec = selected_m2_specs[boost_name]
                    bag_refit_m = bag_res.get("refit_m", args.m2_m)
                    boost_refit_m = boost_res.get("refit_m", args.m2_m)
                    bag_params = dict(bag_res.get("refit_params") or {})
                    boost_params = dict(boost_res.get("refit_params") or {})
                    baseline_component = (
                        "bagging"
                        if bag_res["overall_r2"] >= boost_res["overall_r2"]
                        else "boosting"
                    )
                    m2_blend_candidate = {
                        "label": f"blend[{bag_name}+{boost_name}]",
                        "overall_r2": r2_candidate,
                        "mean_r2": float(blend_pf.mean()),
                        "std_r2": float(blend_pf.std()),
                        "oof": blend_oof,
                        "weight": weight_candidate,
                        "per_fold_scores": blend_pf,
                        "components": {
                            "bagging": {
                                "name": bag_name,
                                "spec": bag_spec,
                                "params": bag_params,
                                "refit_m": bag_refit_m,
                                "result": bag_res,
                            },
                            "boosting": {
                                "name": boost_name,
                                "spec": boost_spec,
                                "params": boost_params,
                                "refit_m": boost_refit_m,
                                "result": boost_res,
                            },
                        },
                        "baseline_component": baseline_component,
                    }
                    print(
                        f"[M2] Bagging/boosting blend candidate {bag_name}+{boost_name}: "
                        f"OOF R2={r2_candidate:.5f}, weight={weight_candidate:.2f}",
                        flush=True,
                    )

            best_m2_spec_name, best_m2_result = select_best_spec(nested_results_m2)
            if m2_blend_candidate is not None and (
                best_m2_result is None
                or m2_blend_candidate["overall_r2"]
                > best_m2_result.get("overall_r2", float("-inf"))
            ):
                m2_selection_type = "blend"
                best_m2_spec = None
                best_m2_params = {}
                best_m2_m = None
                oof_m2 = m2_blend_candidate["oof"]
                r2_m2 = m2_blend_candidate["overall_r2"]
                pf_m2_stats = (
                    m2_blend_candidate["mean_r2"],
                    m2_blend_candidate["std_r2"],
                )
                m2_blend_weight = float(m2_blend_candidate["weight"])
                best_m2_components = m2_blend_candidate["components"]
                best_m2_spec_name = m2_blend_candidate["label"]
                baseline_key = m2_blend_candidate["baseline_component"]
                baseline_component = best_m2_components[baseline_key]
                baseline_m = baseline_component.get("refit_m")
                if baseline_m is None:
                    baseline_m = args.m2_m
                mu_oof, genre_map_full_cv, gmean_cv = oof_genre_mean_with_folds(
                    genre=genres, y=y, folds=folds, m=float(baseline_m)
                )
                genre_map_full = genre_map_full_cv
                gmean = gmean_cv
                r2_mu_only = r2_score(y_np, mu_oof)
            elif best_m2_result is not None and best_m2_spec_name is not None:
                m2_selection_type = "single"
                best_m2_spec = selected_m2_specs[best_m2_spec_name]
                best_m2_params = best_m2_result.get("refit_params", {})
                oof_m2 = best_m2_result["oof"]
                r2_m2 = best_m2_result["overall_r2"]
                best_m2_m = best_m2_result.get("refit_m", args.m2_m)
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
                candidate_param_sets=candidate_params_from_m2,
                enable_random_search=False,
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
                if m2_selection_type == "blend":
                    bag_component = best_m2_components.get("bagging")
                    boost_component = best_m2_components.get("boosting")
                    if bag_component:
                        print(
                            f"  bagging ({bag_component['name']}): params={bag_component['params']} "
                            f"refit_m={bag_component['refit_m']}"
                        )
                    if boost_component:
                        print(
                            f"  boosting ({boost_component['name']}): params={boost_component['params']} "
                            f"refit_m={boost_component['refit_m']}"
                        )
                    if m2_blend_weight is not None:
                        print(f"  blend weight (boost share)={m2_blend_weight:.2f}")
                elif best_m2_params:
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
                if m2_selection_type == "blend":
                    bag_component = best_m2_components.get("bagging")
                    boost_component = best_m2_components.get("boosting")
                    if bag_component:
                        print(
                            f"  bagging ({bag_component['name']}): params={bag_component['params']} "
                            f"refit_m={bag_component['refit_m']}"
                        )
                    if boost_component:
                        print(
                            f"  boosting ({boost_component['name']}): params={boost_component['params']} "
                            f"refit_m={boost_component['refit_m']}"
                        )
                    if m2_blend_weight is not None:
                        print(f"  blend weight (boost share)={m2_blend_weight:.2f}")
                elif best_m2_params:
                    print(f"  Refit params: {best_m2_params}")
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
            if m2_selection_type == "blend":
                if not best_m2_components:
                    raise ValueError(
                        "Blend requested for M2 but component specifications are unavailable."
                    )

                def train_component(comp_key: str) -> None:
                    component = best_m2_components[comp_key]
                    refit_m = component.get("refit_m")
                    if refit_m is None:
                        refit_m = args.m2_m
                    genre_map, comp_gmean = compute_genre_smoothing(
                        genre=genres, y=y, m=float(refit_m)
                    )
                    component["genre_map_full"] = genre_map
                    component["gmean"] = comp_gmean
                    mu_train_comp = genres.map(genre_map).fillna(comp_gmean).to_numpy()
                    component["mu_train"] = mu_train_comp

                    def ctor(
                        seed: int, spec=component["spec"], params=component["params"]
                    ):
                        return spec.instantiate(seed=seed, params=params)

                    start_msg = f"Starting full-data residual model training ({comp_key}: {component['name']})..."
                    end_msg = (
                        "Finished full-data residual model training "
                        f"({comp_key}: {component['name']}) in {{elapsed:.2f}} sec."
                    )
                    model, scaler_obj = fit_full_model(
                        model_ctor=ctor,
                        X=X,
                        target=y.to_numpy() - mu_train_comp,
                        seed=args.seed,
                        start_message=start_msg,
                        end_message_template=end_msg,
                        scaler_kwargs=component["spec"].scaler_kwargs,
                        log_model_name=f"{component['name']} ({comp_key})",
                        log_params=component["params"],
                    )
                    component["model"] = model
                    component["scaler"] = scaler_obj

                train_component("bagging")
                train_component("boosting")

                def predict_component(
                    comp_key: str, X_frame: pd.DataFrame, genre_series: pd.Series
                ) -> np.ndarray:
                    component = best_m2_components[comp_key]
                    scaler_obj = component.get("scaler")
                    X_scaled = (
                        scaler_obj.transform(X_frame)
                        if scaler_obj is not None
                        else X_frame
                    )
                    mu_vals = (
                        genre_series.map(component["genre_map_full"])
                        .fillna(component["gmean"])
                        .to_numpy()
                    )
                    residual_pred = component["model"].predict(X_scaled)
                    return mu_vals + residual_pred

                bag_pred_train = predict_component("bagging", X, genres)
                boost_pred_train = predict_component("boosting", X, genres)
                m2_blend_weight, _ = best_blend_weight(
                    y_true=y.to_numpy(),
                    p1=bag_pred_train,
                    p2=boost_pred_train,
                    step=0.01,
                )
                print(
                    f"Computed M2 blend weight on training data (boost share): {m2_blend_weight:.2f}",
                    flush=True,
                )
            else:
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

        def predict_m2_dataframe(
            X_frame: pd.DataFrame, genre_series: pd.Series
        ) -> np.ndarray:
            if m2_selection_type == "blend":
                if not best_m2_components or m2_blend_weight is None:
                    raise ValueError(
                        "M2 blend components or weight unavailable for prediction."
                    )

                def component_prediction(comp_key: str) -> np.ndarray:
                    component = best_m2_components[comp_key]
                    if "model" not in component or "genre_map_full" not in component:
                        raise ValueError(
                            f"Component '{comp_key}' is not fitted for M2 blending."
                        )
                    scaler_obj = component.get("scaler")
                    X_scaled = (
                        scaler_obj.transform(X_frame)
                        if scaler_obj is not None
                        else X_frame
                    )
                    mu_vals = (
                        genre_series.map(component["genre_map_full"])
                        .fillna(component["gmean"])
                        .to_numpy()
                    )
                    residual_pred = component["model"].predict(X_scaled)
                    return mu_vals + residual_pred

                bag_pred = component_prediction("bagging")
                boost_pred = component_prediction("boosting")
                return m2_blend_weight * boost_pred + (1.0 - m2_blend_weight) * bag_pred

            if genre_map_full is None or gmean is None or m2_full is None:
                raise ValueError("M2 model unavailable for prediction.")
            mu_vals = genre_series.map(genre_map_full).fillna(gmean).to_numpy()
            X_scaled = (
                m2_scaler.transform(X_frame) if m2_scaler is not None else X_frame
            )
            residual_pred = m2_full.predict(X_scaled)
            return mu_vals + residual_pred

        if args.mode == "ensemble" and w_best is None:
            if m1_full is None:
                raise ValueError(
                    "M1 full model unavailable for computing blend weight."
                )
            X_train_m1 = m1_scaler.transform(X) if m1_scaler is not None else X
            p1_train = m1_full.predict(X_train_m1)

            if m2_selection_type == "blend":
                if not best_m2_components or m2_blend_weight is None:
                    raise ValueError(
                        "M2 blend components unavailable for computing blend weight."
                    )
                p2_train = predict_m2_dataframe(X, genres)
            else:
                if genre_map_full is None or gmean is None:
                    chosen_m = best_m2_m or args.m2_m
                    genre_map_full, gmean = compute_genre_smoothing(
                        genre=genres, y=y, m=chosen_m
                    )
                if m2_full is None:
                    raise ValueError(
                        "M2 full model unavailable for computing blend weight."
                    )
                train_mu = genres.map(genre_map_full).fillna(gmean).to_numpy()
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
            if args.mode in ("m2", "ensemble"):
                if m2_selection_type == "blend":
                    for comp_key, component in best_m2_components.items():
                        model_obj = component.get("model")
                        if model_obj is None:
                            continue
                        model_name = "M2 residual"
                        if component.get("name"):
                            model_name += (
                                f" ({component['name']}, full-data, {comp_key})"
                            )
                        print_feature_importance(
                            model_obj, X.columns, model_name=model_name
                        )
                elif m2_full is not None:
                    model_name = "M2 residual"
                    if best_m2_spec_name:
                        model_name += f" ({best_m2_spec_name}, full-data)"
                    print_feature_importance(m2_full, X.columns, model_name=model_name)

        test_path = os.path.join(DATA_DIR, "test_data.csv")
        if not os.path.exists(test_path):
            print(f"Test data not found. Please place `test_data.csv` at `{test_path}`")
            sys.exit(1)
        test_df = df_transformer.transform(pd.read_csv(test_path))
        row_ids = test_df.index.to_numpy()
        X_test = test_df.drop(columns=[TARGET, GENRE], errors="ignore")

        if args.mode == "m1":
            if m1_full is None:
                raise ValueError("M1 full model not available for inference.")
            X_test_s = m1_scaler.transform(X_test) if m1_scaler is not None else X_test
            pred_norm = m1_full.predict(X_test_s)
        elif args.mode == "m2":
            pred_norm = predict_m2_dataframe(X_test, test_df[GENRE])
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
                pred_norm = predict_m2_dataframe(X_test, test_df[GENRE])
            else:
                # Default to blending when no strategy was selected via CV
                if (
                    w_best is None
                    or m1_full is None
                    or (m2_selection_type != "blend" and m2_full is None)
                ):
                    raise ValueError(
                        "Ensemble inference requires fitted models and blend weight."
                    )
                X_test_s_m1 = (
                    m1_scaler.transform(X_test) if m1_scaler is not None else X_test
                )
                p1 = m1_full.predict(X_test_s_m1)
                p2 = predict_m2_dataframe(X_test, test_df[GENRE])
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
        "--m2-blend-bag-boost",
        action="store_true",
        help=(
            "When evaluating M2, try blending the best bagging-type (rf/et) and "
            "boosting-type (xgb/cat) models using convex non-negative weights."
        ),
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
