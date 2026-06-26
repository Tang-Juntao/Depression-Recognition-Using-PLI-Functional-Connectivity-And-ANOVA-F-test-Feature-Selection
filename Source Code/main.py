import os
import sys
import time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from src.data_loader import load_data
from src.metrics import evaluate_cv, compute_roc_data, METRIC_NAMES
from src.model1_svm import build_svm
from src.model2_mlp import build_mlp
from src.model3_fs_svm import build_fs_svm, build_fs_svm_fast, build_tuned_svm
from src.plot_utils import (plot_comparison_bar, plot_roc_curves,
                            plot_confusion_matrix, plot_k_sensitivity)
N_SPLITS = 10
SEED = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), 'coh_ind_new20hb')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

def print_separator(title):
    print(f'\n{"=" * 65}')
    print(f'  {title}')
    print(f'{"=" * 65}')
def print_result_row(name, means, stds):
    row = f'{name:<24}'
    for m, s in zip(means, stds):
        row += f'{m * 100:>7.2f}'
    row += '  (均值)'
    print(row)
    row2 = f'{"":<24}'
    for m, s in zip(means, stds):
        row2 += f'  ±{s * 100:>4.2f}'
    print(row2)
def run_k_sensitivity(X, y):
    """    测试不同特征数 k 对改进模型性能的影响。
    使用分层 10 折 CV。    """
    print_separator('特征数 k 敏感性分析')
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.svm import SVC

    k_list = [80, 160, 320, 560, 800, 1200, 1600]
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    scores = []
    for k in k_list:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('fs', SelectKBest(score_func=f_classif, k=k)),
            ('clf', SVC(kernel='rbf', gamma='scale', probability=True,
                        random_state=SEED)),
        ])
        s = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        scores.append(s.mean())
        print(f'  k={k:>5}: ACC={s.mean() * 100:.2f}% (±{s.std() * 100:.2f}%)')

    plot_k_sensitivity(k_list, scores,
                       os.path.join(RESULTS_DIR, 'k_sensitivity.png'))
    return k_list, scores


def main():
    t_start = time.time()
    print('=' * 65)
    print('  基于 EEG 功能连接特征的抑郁症识别实验')
    print(f'  交叉验证: {N_SPLITS} 折 | 随机种子: {SEED}')
    print('=' * 65)

    # ---- 1. 加载数据 ----
    print_separator('1. 数据加载')
    data_path = os.path.join(DATA_DIR, 'data_coh20.mat')
    X, y = load_data(DATA_DIR)
    print(f'   正样本 (MDD=1): {y.sum()}   负样本 (HC=0): {len(y) - y.sum()}')
    print(f'   特征维度: {X.shape[1]}   样本数: {X.shape[0]}')

    # ---- 2. 定义三个模型 ----
    print_separator('2. 构建模型')
    builders = [
        ('SVM (模型1)',              build_svm),
        ('MLP (模型2)',              build_mlp),
        ('特征选择+SVM (改进模型3)', build_fs_svm_fast),
    ]
    all_means = []
    all_stds = []
    roc_data_list = []
    auc_values = []
    cv_details_list = []

    # ---- 3. 评估 ----
    for model_name, build_fn in builders:
        print(f'\n  >>> 训练与评估: {model_name}')
        t0 = time.time()

        estimator = build_fn()
        means, stds, details = evaluate_cv(estimator, X, y,
                                           n_splits=N_SPLITS, seed=SEED,
                                           return_details=True)
        elapsed = time.time() - t0

        all_means.append(means)
        all_stds.append(stds)
        cv_details_list.append(details)

        print(f'  [耗时 {elapsed:.1f}s]')
        for i, name in enumerate(METRIC_NAMES):
            print(f'    {name:<12}: {means[i] * 100:>6.2f}%  ± {stds[i] * 100:.2f}%')
        # 打印模型超参数信息
        from sklearn.model_selection import GridSearchCV as _GSCV
        if isinstance(estimator, _GSCV):
            fitted = estimator.fit(X, y)
            print(f'    最佳参数: {fitted.best_params_}')
            print(f'    最佳内层 CV 分数: {fitted.best_score_ * 100:.2f}%')
        elif hasattr(estimator, 'named_steps'):
            clf = estimator.named_steps.get('clf', None)
            if clf:
                params = clf.get_params()
                print(f'    模型参数: C={params.get("C")}, gamma={params.get("gamma")}')
    print_separator('4. 绘制图表')
    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone
    from sklearn.metrics import roc_curve, auc

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for model_name, build_fn in builders:
        estimator = build_fn()
        tprs = []
        mean_fpr = np.linspace(0, 1, 100)

        for tr_idx, te_idx in skf.split(X, y):
            est = clone(estimator)
            est.fit(X[tr_idx], y[tr_idx])
            if hasattr(est, 'predict_proba'):
                y_score = est.predict_proba(X[te_idx])[:, 1]
            elif hasattr(est, 'decision_function'):
                y_score = est.decision_function(X[te_idx])
            else:
                y_score = est.predict(X[te_idx]).astype(float)
            fpr, tpr, _ = roc_curve(y[te_idx], y_score)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(tprs, axis=0)
        roc_data_list.append((mean_fpr, mean_tpr, std_tpr))
        roc_auc = auc(mean_fpr, mean_tpr)
        auc_values.append(roc_auc)
    # ---- 5. 结果汇总表 ----
    print_separator('5. 最终结果汇总')
    header = f'{"模型":<24}' + ''.join(f'{m:>8}' for m in METRIC_NAMES)
    print(header)
    print('-' * (24 + 8 * len(METRIC_NAMES)))
    model_names_display = [b[0] for b in builders]
    for i, (name, means, stds) in enumerate(zip(model_names_display,
                                                 all_means, all_stds)):
        print_result_row(name, means, stds)

    # ---- 6. 绘制与保存图表 ----
    plot_comparison_bar(all_means, all_stds,
                        os.path.join(RESULTS_DIR, 'comparison_bar.png'),
                        model_names=model_names_display)

    plot_roc_curves(roc_data_list, auc_values,
                    os.path.join(RESULTS_DIR, 'roc_curves.png'),
                    model_names=model_names_display)

    # 改进模型的混淆矩阵 (聚合所有折预测)
    y_true_all = np.concatenate(cv_details_list[-1]['y_true_list'])
    y_pred_all = np.concatenate(cv_details_list[-1]['y_pred_list'])
    plot_confusion_matrix(y_true_all, y_pred_all,
                          os.path.join(RESULTS_DIR, 'confusion_matrix_model3.png'),
                          title='改进模型3 (特征选择+SVM) 混淆矩阵')

    # ---- 7. k 敏感性 ----
    run_k_sensitivity(X, y)

    # ---- END ----
    print_separator('实验完成')
    print(f'  总耗时: {time.time() - t_start:.0f}s')
    print(f'  图表保存在: {RESULTS_DIR}/')
    print(f'    - comparison_bar.png     (三模型柱状图)')
    print(f'    - roc_curves.png         (ROC 曲线叠加)')
    print(f'    - confusion_matrix_model3.png (改进模型混淆矩阵)')
    print(f'    - k_sensitivity.png      (特征数敏感性曲线)')
if __name__ == '__main__':
    main()