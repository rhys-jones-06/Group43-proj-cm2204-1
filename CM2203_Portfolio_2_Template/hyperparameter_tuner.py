import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# 30 candidates log-spaced from 1e-12 to 1.0
VAR_SMOOTHING_GRID = np.logspace(-12, 0, 30)

FAIRNESS_LAMBDA = 0.5
INNER_FOLDS = 5


def _fairness_score(
    y_true: pd.Series,
    y_pred: np.ndarray,
    sensitive: pd.Series,
    lam: float,
) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
    groups = sensitive.unique()
    if len(groups) < 2:
        return bal_acc

    group_accs = []
    for g in groups:
        mask = sensitive == g
        if mask.sum() > 0:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                group_accs.append(balanced_accuracy_score(y_true[mask], y_pred[mask]))

    disparity = max(group_accs) - min(group_accs)
    return bal_acc - lam * disparity


def tune_var_smoothing(
    training_data: pd.DataFrame,
    class_name: str,
    sensitive_feature: str = 'sex',
    var_smoothing_grid: np.ndarray = VAR_SMOOTHING_GRID,
    lam: float = FAIRNESS_LAMBDA,
    inner_folds: int = INNER_FOLDS,
) -> float:

    X = training_data.drop(class_name, axis=1)
    y = training_data[class_name]
    sensitive = X[sensitive_feature] if sensitive_feature in X.columns else None

    skf = StratifiedKFold(n_splits=inner_folds, shuffle=True)

    best_vs, best_score = var_smoothing_grid[0], -np.inf

    for vs in var_smoothing_grid:
        fold_scores = []

        for tr_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X_tr_enc = enc.fit_transform(X_tr).astype(float)
            X_val_enc = enc.transform(X_val).astype(float)

            scaler = StandardScaler()
            X_tr_enc = scaler.fit_transform(X_tr_enc)
            X_val_enc = scaler.transform(X_val_enc)

            model = GaussianNB(var_smoothing=vs)
            model.fit(X_tr_enc, y_tr)
            y_pred = model.predict(X_val_enc)

            if sensitive is not None:
                score = _fairness_score(y_val, y_pred, sensitive.iloc[val_idx], lam)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    score = balanced_accuracy_score(y_val, y_pred)

            fold_scores.append(score)

        mean_score = np.mean(fold_scores)
        if mean_score > best_score:
            best_score = mean_score
            best_vs = vs

    return best_vs