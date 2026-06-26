"""改进模型 3: 特征选择 + 超参数寻优 SVM。
在模型 1 基础上加入:
  1. ANOVA F 检验特征选择 (SelectKBest + f_classif)
  2. 网格搜索 (GridSearchCV) 同时优化 k、C、gamma
外层 10 折 CV 使用 GridSearchCV 作为 estimator = 嵌套交叉验证。"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
def build_fs_svm(k_values=None, C_values=None, gamma_values=None,
                 inner_cv=5, n_jobs=-1, seed=42):
    if k_values is None:
        k_values = [320, 800, 1600, 2400, 3200]
    if C_values is None:
        C_values = [0.1, 1, 10]
    if gamma_values is None:
        gamma_values = ['scale', 0.01, 0.001]

    base = Pipeline([
        ('scaler', StandardScaler()),
        ('fs', SelectKBest(score_func=f_classif, k=320)),
        ('clf', SVC(kernel='rbf', probability=True, random_state=seed)),
    ])

    param_grid = {
        'fs__k': k_values,
        'clf__C': C_values,
        'clf__gamma': gamma_values,
    }
    inner_skf = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=seed)
    return GridSearchCV(base, param_grid, cv=inner_skf,
                        scoring='accuracy', n_jobs=n_jobs, verbose=0)
def build_tuned_svm(C_values=None, gamma_values=None,
                    inner_cv=5, n_jobs=-1, seed=42):
    if C_values is None:
        C_values = [0.01, 0.1, 1, 10, 100]
    if gamma_values is None:
        gamma_values = ['scale', 'auto', 0.1, 0.01, 0.001]

    base = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', probability=True, random_state=seed)),
    ])

    param_grid = {
        'clf__C': C_values,
        'clf__gamma': gamma_values,
    }
    inner_skf = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=seed)
    return GridSearchCV(base, param_grid, cv=inner_skf,
                        scoring='accuracy', n_jobs=n_jobs, verbose=0)
def build_fs_svm_fast():
    return build_fs_svm(
        k_values=[320, 800, 1600, 2400],
        C_values=[0.1, 1, 10],
        gamma_values=['scale'],
        inner_cv=5
    )