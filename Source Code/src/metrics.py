"""6 项指标 (ACC, SEN, SPE, PRE, F1, AUC) 和分层 10 折交叉验证。"""
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve)
METRIC_NAMES = ['ACC', 'SEN(Recall)', 'SPE', 'PRE', 'F1', 'AUC']
def specificity_score(y_true, y_pred):
    """特异度 SPE = TN / (TN + FP)"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) else 0.0

def _compute_metrics(y_true, y_pred, y_score):
    return [
        accuracy_score(y_true, y_pred),
        recall_score(y_true, y_pred),          # SEN
        specificity_score(y_true, y_pred),     # SPE
        precision_score(y_true, y_pred),       # PRE
        f1_score(y_true, y_pred),              # F1
        roc_auc_score(y_true, y_score),        # AUC
    ]

def evaluate_cv(estimator, X, y, n_splits=10, seed=42, return_details=False):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    details = {'y_true_list': [], 'y_score_list': [], 'y_pred_list': []}

    for tr_idx, te_idx in skf.split(X, y):
        est = clone(estimator)
        est.fit(X[tr_idx], y[tr_idx])
        y_pred = est.predict(X[te_idx])

        if hasattr(est, 'predict_proba'):
            y_score = est.predict_proba(X[te_idx])[:, 1]
        elif hasattr(est, 'decision_function'):
            y_score = est.decision_function(X[te_idx])
        else:
            y_score = y_pred.astype(float)

        rows.append(_compute_metrics(y[te_idx], y_pred, y_score))
        if return_details:
            details['y_true_list'].append(y[te_idx])
            details['y_score_list'].append(y_score)
            details['y_pred_list'].append(y_pred)

    rows = np.array(rows)
    result = (rows.mean(axis=0), rows.std(axis=0))
    if return_details:
        return result + (details,)
    return result

def compute_roc_data(estimator, X, y, n_splits=10, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)

    for tr_idx, te_idx in skf.split(X, y):
        est = clone(estimator)
        est.fit(X[tr_idx], y[tr_idx])
        if hasattr(est, 'predict_proba'):
            y_score = est.predict_proba(X[te_idx])[:, 1]
        else:
            y_score = est.decision_function(X[te_idx])
        fpr, tpr, _ = roc_curve(y[te_idx], y_score)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    return mean_fpr, tprs, mean_tpr, std_tpr