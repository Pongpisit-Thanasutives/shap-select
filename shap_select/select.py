from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
import shap
import statsmodels.api as sm
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, Ridge


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_linear_model(model: Any) -> bool:
    """Return True if *model* is a scikit-learn LinearRegression or Ridge."""
    return isinstance(model, (LinearRegression, Ridge))


def _extract_summary(result) -> pd.DataFrame:
    """Pull coefficient table from a statsmodels fit result."""
    return result.summary2().tables[1]


def is_fitted(model):
    try:
        check_is_fitted(model)
        return True
    except NotFittedError:
        print("the model is NOT fitted.")
        return False


# ---------------------------------------------------------------------------
# Linear-model SHAP utilities
# ---------------------------------------------------------------------------

def shap_linear_importance(
    X: np.ndarray,
    y: np.ndarray,
    scale: bool = True,
    full: bool = False,
    alpha: float = 0,
) -> np.ndarray:
    """
    Compute SHAP values for a linear model via ``shap.explainers.Linear``.

    Fits a ``LinearRegression`` (when ``alpha == 0``) or ``Ridge``
    (when ``alpha > 0``) internally, then uses the SHAP Linear explainer
    for exact Shapley values.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,)
        Target vector.
    scale : bool, default True
        When ``full=False``, divide mean |SHAP| values by their sum so the
        result is a normalised importance fraction (sums to 1).
        Has no effect when ``full=True``.
    full : bool, default False
        If False return the mean absolute SHAP value per feature, shape
        ``(n_features,)``.  If True return the full SHAP matrix, shape
        ``(n_samples, n_features)``.
    alpha : float, default 0
        Ridge regularisation strength.  ``alpha=0`` uses plain
        ``LinearRegression``; any positive value uses ``Ridge``.

    Returns
    -------
    np.ndarray
        Mean |SHAP| vector (``full=False``) or full SHAP matrix
        (``full=True``).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    alpha = max(alpha, 0)

    if alpha == 0:
        lm: Union[LinearRegression, Ridge] = LinearRegression(fit_intercept=False)
    else:
        lm = Ridge(fit_intercept=False, alpha=alpha)
    lm.fit(X, y)

    explainer = shap.explainers.Linear(lm, X)
    feature_importance: np.ndarray = explainer(X).values  # (n_samples, n_features)

    if not full:
        feature_importance = np.abs(feature_importance).mean(axis=0)
        if scale:
            total = feature_importance.sum()
            if total > 0:
                feature_importance = feature_importance / total
    return feature_importance


def shap_feature_elimination(
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.01,
    s_max: Optional[int] = None,
    alpha: float = 0,
) -> Tuple[List[int], List[int]]:
    """
    Iteratively remove features whose relative SHAP importance falls below
    *threshold* until all remaining features contribute at least that fraction
    of the total importance.

    An optional *s_max* pre-selection step caps the candidate set before the
    iterative loop: the top-``s_max`` features (by mean |SHAP|) that
    individually exceed ``threshold`` are kept; the rest go straight to
    *remove*.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,)
        Target vector.
    threshold : float, default 0.01
        Minimum fraction of total mean |SHAP| a feature must contribute to be
        kept.  Clipped to [0, 1].
    s_max : int or None, default None
        Pre-select at most *s_max* features before the elimination loop.
        ``None`` disables the pre-selection step.

    Returns
    -------
    keep : list of int
        Column indices (into the original feature matrix) of retained features.
    remove : list of int
        Column indices of eliminated features, in removal order.
    """
    threshold = min(max(threshold, 0.0), 1.0)
    X = np.asarray(X, dtype=float)
    n_features = X.shape[1]
    keep = np.arange(n_features)
    remove: List[int] = []

    # -- optional s_max pre-selection pass ----------------------------------
    if s_max is not None and s_max < n_features: 
        abs_shap = shap_linear_importance(X, y, scale=False, full=True, alpha=alpha)
        abs_shap = np.abs(abs_shap).mean(axis=0)
        abs_shap = abs_shap / abs_shap.sum()
        order = np.argsort(abs_shap)[::-1]
        keep = np.sort(order[abs_shap[order] > threshold][:s_max])
        remove = np.setdiff1d(np.arange(n_features), keep, assume_unique=True).tolist()

    # -- iterative elimination loop ----------------------------------------
    while len(keep) > 0:
        abs_shap = shap_linear_importance(X[:, keep], y, scale=False, full=True, alpha=alpha)
        abs_shap = np.abs(abs_shap).mean(axis=0)
        total = abs_shap.sum()
        if total == 0:
            break
        abs_shap_pct = abs_shap / total
        keep_mask = abs_shap_pct >= threshold
        if keep_mask.all():
            break
        remove.extend(keep[~keep_mask].tolist())
        keep = keep[keep_mask]

    return keep.tolist(), remove


# ---------------------------------------------------------------------------
# SHAP feature matrix builders
# ---------------------------------------------------------------------------

def create_shap_features_linear(
    model: Union[LinearRegression, Ridge],
    validation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a SHAP-value DataFrame for a fitted ``LinearRegression`` or
    ``Ridge`` model using ``shap.explainers.Linear``.

    Parameters
    ----------
    model : LinearRegression or Ridge
        A fitted scikit-learn linear model.
    validation_df : pd.DataFrame
        Validation features (same columns used during training).

    Returns
    -------
    pd.DataFrame
        SHAP values with the same index and columns as *validation_df*.
    """
    assert is_fitted(model)
    X = validation_df.values.astype(float)
    explainer = shap.explainers.Linear(model, X)
    shap_matrix = explainer(X).values
    return pd.DataFrame(shap_matrix, columns=validation_df.columns, index=validation_df.index)


def create_shap_features(
    model: Any,
    validation_df: pd.DataFrame,
    classes: Optional[List] = None,
) -> Union[pd.DataFrame, Dict[Any, pd.DataFrame]]:
    """
    Generate SHAP values for a tree-based (or any ``shap.Explainer``-
    compatible) model on a validation dataset.

    Parameters
    ----------
    model : Any
        A trained model compatible with ``shap.Explainer`` (XGBoost,
        LightGBM, scikit-learn trees/forests, …).
    validation_df : pd.DataFrame
        Validation features.
    classes : list or None
        Class labels for multiclass classification.  ``None`` for regression
        or binary classification.

    Returns
    -------
    pd.DataFrame or dict of pd.DataFrame
        SHAP values.  For multiclass, a dict keyed by class label.
    """
    explainer = shap.Explainer(model, model_output="raw")(validation_df)
    shap_values = explainer.values

    if shap_values.ndim == 2:
        assert classes is None, (
            "Don't specify classes for binary classification or regression"
        )
        return pd.DataFrame(
            shap_values, columns=validation_df.columns, index=validation_df.index
        )
    elif shap_values.ndim == 3:  # multiclass
        return {
            c: pd.DataFrame(
                shap_values[:, :, i],
                columns=validation_df.columns,
                index=validation_df.index,
            )
            for i, c in enumerate(classes)
        }
    raise ValueError(f"Unexpected SHAP value shape: {shap_values.shape}")


# ---------------------------------------------------------------------------
# Significance functions
# ---------------------------------------------------------------------------

def binary_classifier_significance(
    shap_features: pd.DataFrame,
    target: pd.Series,
    alpha: float,
) -> pd.DataFrame:
    """
    Fit a regularised logistic regression on *shap_features* and return
    per-feature significance statistics.
    """
    shap_features_with_constant = sm.add_constant(shap_features)

    alpha_in_loop = alpha
    for _ in range(10):
        try:
            logit_model = sm.Logit(target, shap_features_with_constant)
            result = logit_model.fit_regularized(disp=False, alpha=alpha_in_loop)
            break
        except np.linalg.LinAlgError:
            alpha_in_loop *= 5
        except Exception as exc:
            raise RuntimeError(exc) from exc
    else:
        raise RuntimeError("Logistic regression failed to converge after maximum retries.")

    summary_frame = _extract_summary(result)
    result_df = pd.DataFrame(
        {
            "feature name":      summary_frame.index,
            "coefficient":       summary_frame["Coef."],
            "stderr":            summary_frame["Std.Err."],
            "stat.significance": summary_frame["P>|z|"],
            "t-value":           summary_frame["Coef."] / summary_frame["Std.Err."],
        }
    ).reset_index(drop=True)
    result_df["closeness to 1.0"] = (result_df["coefficient"] - 1.0).abs()
    return result_df.loc[result_df["feature name"] != "const"]


def multi_classifier_significance(
    shap_features: Dict[Any, pd.DataFrame],
    target: pd.Series,
    alpha: float,
    return_individual_significances: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[pd.DataFrame]]]:
    """
    One-vs-all logistic regression significance for multiclass targets.
    Applies Bonferroni correction across classes.
    """
    significance_dfs = [
        binary_classifier_significance(feature_df, (target == cls).astype(int), alpha)
        for cls, feature_df in shap_features.items()
    ]

    combined_df = pd.concat(significance_dfs)
    max_significance_df = (
        combined_df.groupby("feature name", as_index=False)
        .agg({"t-value": "max", "closeness to 1.0": "min", "coefficient": "max"})
        .reset_index(drop=True)
    )
    n_classes = len(shap_features)
    max_significance_df["stat.significance"] = max_significance_df["t-value"].apply(
        lambda x: n_classes * (1 - stats.norm.cdf(x))
    )

    if return_individual_significances:
        return max_significance_df, significance_dfs
    return max_significance_df


def regression_significance(
    shap_features: pd.DataFrame,
    target: pd.Series,
    alpha: float,
) -> pd.DataFrame:
    """
    Fit a regularised OLS on *shap_features* and return per-feature
    significance statistics.

    Retries with progressively smaller regularisation (up to 12 attempts)
    to avoid NaN-filled summary tables caused by near-singular designs.
    On the final attempt, ``alpha`` is forced to 0 (plain OLS) as a
    last-resort fallback.
    """
    summary_frame = None
    for i in range(12):
        effective_alpha = 0.0 if i == 11 else alpha * (10 ** -i)
        ols_model = sm.OLS(target, shap_features)
        result = ols_model.fit_regularized(alpha=effective_alpha, refit=True)
        summary_frame = _extract_summary(result)
        if not summary_frame.isna().any().any():
            break

    result_df = pd.DataFrame(
        {
            "feature name":      summary_frame.index,
            "coefficient":       summary_frame["Coef."],
            "stderr":            summary_frame["Std.Err."],
            "stat.significance": summary_frame["P>|t|"],
            "t-value":           summary_frame["Coef."] / summary_frame["Std.Err."],
        }
    ).reset_index(drop=True)
    result_df["closeness to 1.0"] = (result_df["coefficient"] - 1.0).abs()
    return result_df


# ---------------------------------------------------------------------------
# Significance pipeline
# ---------------------------------------------------------------------------

def shap_features_to_significance(
    shap_features: Union[pd.DataFrame, Dict[Any, pd.DataFrame]],
    target: pd.Series,
    task: str,
    alpha: float,
) -> pd.DataFrame:
    """
    Dispatch to the appropriate significance function and return results
    sorted by t-value descending (most significant first).
    """
    if task == "regression":
        result_df = regression_significance(shap_features, target, alpha)
    elif task == "binary":
        result_df = binary_classifier_significance(shap_features, target, alpha)
    elif task == "multiclass":
        result_df = multi_classifier_significance(shap_features, target, alpha)
    else:
        raise ValueError("`task` must be 'regression', 'binary', or 'multiclass'.")

    return result_df.sort_values(by="t-value", ascending=False).reset_index(drop=True)


def iterative_shap_feature_reduction(
    shap_features: Union[pd.DataFrame, Dict[Any, pd.DataFrame]],
    target: pd.Series,
    task: str,
    alpha: float = 1e-6,
) -> pd.DataFrame:
    """
    Repeatedly ablate the weakest feature (lowest t-value) until none
    remain, collecting each row for a final ranked DataFrame.
    """
    collected_rows: List[dict] = []

    while True:
        significance_df = shap_features_to_significance(shap_features, target, task, alpha)

        if significance_df["t-value"].isna().all():
            collected_rows.extend(significance_df.to_dict("records"))
            break

        min_row = significance_df.loc[significance_df["t-value"].idxmin()]
        collected_rows.append(min_row.to_dict())

        feature_to_remove = min_row["feature name"]
        if isinstance(shap_features, pd.DataFrame):
            shap_features = shap_features.drop(columns=[feature_to_remove])
            if len(shap_features.columns) == 0:
                break
        else:
            shap_features = {k: v.drop(columns=[feature_to_remove]) for k, v in shap_features.items()}
            if len(next(iter(shap_features.values())).columns) == 0:
                break

    return (
        pd.DataFrame(collected_rows)
        .sort_values(by="t-value", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def shap_select(
    model: Any,
    validation_df: pd.DataFrame,
    target: Union[pd.Series, str],
    feature_names: Optional[List[str]] = None,
    task: Optional[str] = None,
    threshold: float = 0.05,
    return_extended_data: bool = False,
    alpha: float = 1e-6,
    # backward-compat alias
    tree_model: Any = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Select features based on their SHAP values and statistical significance.

    Parameters
    ----------
    model : Any
        A trained model.  Supported types:

        * Any model compatible with ``shap.Explainer`` (XGBoost, LightGBM,
          scikit-learn trees/forests, …).
        * ``sklearn.linear_model.LinearRegression`` or ``Ridge`` — SHAP
          values are computed via ``shap.explainers.Linear``; task is
          automatically set to ``"regression"``.
    validation_df : pd.DataFrame
        Validation dataset containing the features.
    target : pd.Series or str
        Target values, or the name of the target column in *validation_df*.
    feature_names : list of str or None
        Feature columns to use.  Defaults to all columns of *validation_df*.
    task : str or None
        ``'regression'``, ``'binary'``, or ``'multiclass'``.  Inferred
        automatically when ``None``.
    threshold : float, default 0.05
        Significance threshold (p-value) for feature selection.
    return_extended_data : bool, default False
        If True, also return the SHAP feature DataFrame(s).
    alpha : float, default 1e-6
        Regularisation strength for the significance regression.
    tree_model : Any
        Deprecated alias for *model*.  Will be removed in a future release.

    Returns
    -------
    pd.DataFrame
        Feature names, t-values, p-values, coefficients, and a ``selected``
        flag (1 = selected, 0 = not significant, -1 = negative t-value).
    (pd.DataFrame, pd.DataFrame)
        As above, plus the SHAP feature DataFrame when
        ``return_extended_data=True``.
    """
    if tree_model is not None:
        warnings.warn(
            "The 'tree_model' argument is deprecated; use 'model' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if model is None:
            model = tree_model

    if isinstance(target, str):
        target = validation_df[target]

    if feature_names is None:
        feature_names = validation_df.columns.tolist()

    feat_df = validation_df[feature_names]

    if _is_linear_model(model):
        task = "regression"
        shap_features = create_shap_features_linear(model, feat_df)
    else:
        if task is None:
            if pd.api.types.is_numeric_dtype(target) and target.nunique() > 10:
                task = "regression"
            elif target.nunique() == 2:
                task = "binary"
            else:
                task = "multiclass"

        if task == "multiclass":
            unique_classes = sorted(target.unique().tolist())
            shap_features = create_shap_features(model, feat_df, unique_classes)
        else:
            shap_features = create_shap_features(model, feat_df)

    significance_df = iterative_shap_feature_reduction(shap_features, target, task, alpha)

    significance_df["selected"] = (significance_df["stat.significance"] < threshold).astype(int)
    significance_df.loc[significance_df["t-value"] < 0, "selected"] = -1

    output_cols = ["feature name", "t-value", "stat.significance", "coefficient", "selected"]
    if return_extended_data:
        return significance_df, shap_features
    return significance_df[output_cols]
