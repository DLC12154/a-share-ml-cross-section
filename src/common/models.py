"""Leakage-aware model fitting and prediction generation."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from .evaluation import signal_summary
from .features import ResearchPanel


@dataclass(frozen=True)
class ModelRun:
    predictions: dict[str, pd.DataFrame]
    diagnostics: dict[str, Any]


def _xy(
    panel: ResearchPanel,
    mask: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    data = panel.frame.loc[mask]
    features = data.loc[:, panel.rank_feature_columns].astype(float)
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    target = data["rank_label"].astype(float)
    if target.isna().any():
        raise ValueError("Rank target contains missing values in a fit mask")
    return features, target


def _prediction_frame(
    panel: ResearchPanel,
    mask: pd.Series,
    prediction: np.ndarray,
    *,
    model: str,
    split: str,
) -> pd.DataFrame:
    rows = panel.frame.loc[mask].copy()
    if len(rows) != len(prediction):
        raise ValueError("Prediction length does not match selected rows")
    output = rows[
        [
            "signal_date",
            "code",
            "eligible",
            "future_return",
            "label_end_date",
        ]
    ].copy()
    output["prediction"] = np.asarray(prediction, dtype=float)
    output["model"] = model
    output["split"] = split
    return output


def _mean_prediction(predictions: list[np.ndarray]) -> np.ndarray:
    if not predictions:
        raise ValueError("At least one prediction array is required")
    return np.mean(np.vstack(predictions), axis=0)


def _fit_mlp(
    features: pd.DataFrame,
    target: pd.Series,
    model_config: dict[str, Any],
    seed: int,
) -> MLPRegressor:
    model = MLPRegressor(
        hidden_layer_sizes=tuple(model_config["mlp_hidden_layers"]),
        activation="relu",
        solver="adam",
        alpha=float(model_config["mlp_alpha"]),
        batch_size=int(model_config["mlp_batch_size"]),
        learning_rate_init=float(model_config["mlp_learning_rate"]),
        max_iter=int(model_config["mlp_max_iter"]),
        shuffle=True,
        random_state=seed,
        early_stopping=False,
        tol=1e-4,
        n_iter_no_change=12,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(features, target)
    return model


def _fit_lightgbm(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    model_config: dict[str, Any],
):
    import lightgbm as lgb
    params = {
        "objective": "regression",
        "metric": "l2",
        "boosting_type": "gbdt",
        "num_leaves": int(model_config["lgbm_num_leaves"]),
        "learning_rate": float(model_config["lgbm_learning_rate"]),
        "min_child_samples": int(model_config["lgbm_min_child_samples"]),
        "subsample": float(model_config["lgbm_subsample"]),
        "subsample_freq": int(model_config["lgbm_subsample_freq"]),
        "colsample_bytree": float(model_config["lgbm_colsample_bytree"]),
        "reg_alpha": float(model_config["lgbm_reg_alpha"]),
        "reg_lambda": float(model_config["lgbm_reg_lambda"]),
        "verbosity": -1,
        "seed": 20261029,
        "deterministic": True,
        "force_col_wise": True,
    }
    train_set = lgb.Dataset(train_x, label=train_y, free_raw_data=False)
    valid_set = lgb.Dataset(
        valid_x, label=valid_y, reference=train_set, free_raw_data=False
    )
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=int(model_config["lgbm_max_rounds"]),
        valid_sets=[valid_set],
        callbacks=[
            lgb.early_stopping(
                int(model_config["lgbm_early_stopping_rounds"]),
                verbose=False,
            ),
            lgb.log_evaluation(0),
        ],
    )
    best_iteration = int(
        booster.best_iteration
        if booster.best_iteration and booster.best_iteration > 0
        else min(100, int(model_config["lgbm_max_rounds"]))
    )
    return booster, best_iteration


def _fit_lightgbm_fixed(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    model_config: dict[str, Any],
    num_rounds: int,
):
    import lightgbm as lgb
    params = {
        "objective": "regression",
        "metric": "l2",
        "boosting_type": "gbdt",
        "num_leaves": int(model_config["lgbm_num_leaves"]),
        "learning_rate": float(model_config["lgbm_learning_rate"]),
        "min_child_samples": int(model_config["lgbm_min_child_samples"]),
        "subsample": float(model_config["lgbm_subsample"]),
        "subsample_freq": int(model_config["lgbm_subsample_freq"]),
        "colsample_bytree": float(model_config["lgbm_colsample_bytree"]),
        "reg_alpha": float(model_config["lgbm_reg_alpha"]),
        "reg_lambda": float(model_config["lgbm_reg_lambda"]),
        "verbosity": -1,
        "seed": 20261029,
        "deterministic": True,
        "force_col_wise": True,
    }
    train_set = lgb.Dataset(train_x, label=train_y, free_raw_data=False)
    return lgb.train(
        params,
        train_set,
        num_boost_round=max(1, int(num_rounds)),
    )


def _select_ridge(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    panel: ResearchPanel,
    validation_mask: pd.Series,
    alphas: list[float],
) -> tuple[float, np.ndarray, dict[str, float]]:
    scores: dict[str, float] = {}
    best_alpha = float(alphas[0])
    best_score = -np.inf
    for alpha in alphas:
        model = Ridge(alpha=float(alpha))
        model.fit(train_x, train_y)
        prediction = model.predict(valid_x)
        valid_frame = _prediction_frame(
            panel,
            validation_mask,
            prediction,
            model="ridge",
            split="validation",
        )
        score = float(
            signal_summary(valid_frame)["mean_rank_ic"]
        )
        scores[str(alpha)] = score
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_alpha = float(alpha)
    selected = Ridge(alpha=best_alpha).fit(train_x, train_y)
    return best_alpha, selected.predict(valid_x), scores


def run_models(
    panel: ResearchPanel,
    splits: dict[str, pd.Series],
    model_config: dict[str, Any],
) -> ModelRun:
    frame = panel.frame
    prediction_sets: list[pd.DataFrame] = []
    diagnostics: dict[str, Any] = {
        "features": list(panel.rank_feature_columns),
        "model_backend": "scikit-learn MLPRegressor",
        "seeds": [int(seed) for seed in model_config["seeds"]],
    }

    for split_name in ("validation", "test"):
        split_mask = splits[split_name]
        for predictor, label in (
            ("cs_ret_20", "momentum_20"),
            ("factor_composite", "factor_composite"),
        ):
            if predictor == "factor_composite":
                score = frame.loc[split_mask, list(panel.rank_feature_columns)].mean(
                    axis=1
                )
            else:
                score = frame.loc[split_mask, predictor]
            prediction_sets.append(
                _prediction_frame(
                    panel,
                    split_mask,
                    score.to_numpy(dtype=float),
                    model=label,
                    split=split_name,
                )
            )

    train_mask = splits["development"]
    validation_mask = splits["validation"]
    test_mask = splits["test"]
    final_fit_mask = splits["train_plus_validation"]
    train_x, train_y = _xy(panel, train_mask)
    valid_x, valid_y = _xy(panel, validation_mask)
    final_x, final_y = _xy(panel, final_fit_mask)
    test_x = (
        frame.loc[test_mask, panel.rank_feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    diagnostics["fit_rows"] = {
        "development": int(train_mask.sum()),
        "validation": int(validation_mask.sum()),
        "train_plus_validation": int(final_fit_mask.sum()),
        "test": int(test_mask.sum()),
    }

    ridge_alpha, ridge_valid, ridge_scores = _select_ridge(
        train_x,
        train_y,
        valid_x,
        valid_y,
        panel,
        validation_mask,
        [float(value) for value in model_config["ridge_alphas"]],
    )
    ridge_final = Ridge(alpha=ridge_alpha).fit(final_x, final_y)
    diagnostics["ridge"] = {
        "selected_alpha": ridge_alpha,
        "validation_rank_ic_by_alpha": ridge_scores,
    }
    prediction_sets.extend(
        [
            _prediction_frame(
                panel,
                validation_mask,
                ridge_valid,
                model="ridge",
                split="validation",
            ),
            _prediction_frame(
                panel,
                test_mask,
                ridge_final.predict(test_x),
                model="ridge",
                split="test",
            ),
        ]
    )

    validation_mlp_predictions: list[np.ndarray] = []
    test_mlp_predictions: list[np.ndarray] = []
    for seed in model_config["seeds"]:
        seed = int(seed)
        validation_model = _fit_mlp(train_x, train_y, model_config, seed)
        validation_mlp_predictions.append(validation_model.predict(valid_x))
        final_model = _fit_mlp(final_x, final_y, model_config, seed)
        test_mlp_predictions.append(final_model.predict(test_x))
    prediction_sets.extend(
        [
            _prediction_frame(
                panel,
                validation_mask,
                _mean_prediction(validation_mlp_predictions),
                model="mlp",
                split="validation",
            ),
            _prediction_frame(
                panel,
                test_mask,
                _mean_prediction(test_mlp_predictions),
                model="mlp",
                split="test",
            ),
        ]
    )

    try:
        booster, best_iteration = _fit_lightgbm(
            train_x,
            train_y,
            valid_x,
            valid_y,
            model_config,
        )
        lgb_validation = booster.predict(valid_x, num_iteration=best_iteration)
        final_booster = _fit_lightgbm_fixed(
            final_x,
            final_y,
            model_config,
            best_iteration,
        )
        lgb_test = final_booster.predict(test_x, num_iteration=best_iteration)
        prediction_sets.extend(
            [
                _prediction_frame(
                    panel,
                    validation_mask,
                    lgb_validation,
                    model="lightgbm",
                    split="validation",
                ),
                _prediction_frame(
                    panel,
                    test_mask,
                    lgb_test,
                    model="lightgbm",
                    split="test",
                ),
            ]
        )
        diagnostics["lightgbm"] = {
            "best_iteration": best_iteration,
            "status": "trained",
            "subsample": float(model_config["lgbm_subsample"]),
            "subsample_freq": int(model_config["lgbm_subsample_freq"]),
        }
    except ImportError:
        diagnostics["lightgbm"] = {"status": "unavailable"}

    predictions = pd.concat(prediction_sets, ignore_index=True)
    return ModelRun(predictions=predictions, diagnostics=diagnostics)
