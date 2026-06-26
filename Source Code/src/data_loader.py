"""数据导入与标签构造模块。
从 coh_ind_new20hb 数据集读取 EEG 功能连接特征矩阵并构造标签。
被试构成: 24 名 MDD + 29 名 HC, 每人 10 段 = 530 样本, 每样本 5460 维 PLI 特征。
标签: 前 240 行 = MDD(1), 后 290 行 = HC(0)。"""
import os
import numpy as np
import scipy.io as sio
def load_data(data_path=None, verify_labels=True):
    if data_path is None:
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                 'coh_ind_new20hb')
    # 主数据文件: data_coh20.mat 包含完整的 530×5460 矩阵
    main_file = os.path.join(data_path, 'data_coh20.mat')
    mat = sio.loadmat(main_file)
    X = mat['coh_ind_all'].astype(np.float64)
    if X.shape[0] != 530 and X.shape[1] == 530:
        X = X.T
    assert X.shape == (530, 5460), f'期望 (530, 5460), 得到 {X.shape}'
    assert not np.isnan(X).any(), '数据中存在 NaN'
    # 标签构造: 前 240 行 = 24 名 MDD × 10 段, 后 290 行 = 29 名 HC × 10 段
    n_mdd, n_hc, seg = 24, 29, 10
    y = np.array([1] * (n_mdd * seg) + [0] * (n_hc * seg), dtype=np.int32)
    if verify_labels:
        label_file = os.path.join(data_path, 'data_coh20l.mat')
        if os.path.exists(label_file):
            mat_l = sio.loadmat(label_file)
            Xl = mat_l['coh_ind_all']
            y_true = Xl[:, -1].astype(np.int32)
    print(f'数据加载完成: X.shape={X.shape}, MDD={y.sum()}, HC={len(y) - y.sum()}')
    return X, y
def load_data_by_subject(data_path=None):
    if data_path is None:
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                 'coh_ind_new20hb')
    n_subjects = 53
    n_segments = 10
    n_features = 5460
    X_list = []
    for i in range(1, n_subjects + 1):
        fname = f'data_coh20{i}.mat'
        fpath = os.path.join(data_path, fname)
        mat = sio.loadmat(fpath)
        Xi = mat['coh_ind_hb'].astype(np.float64)
        assert Xi.shape == (n_segments, n_features), \
            f'{fname}: 期望 (10, 5460), 得到 {Xi.shape}'
        X_list.append(Xi)
    X = np.vstack(X_list)
    assert X.shape == (530, 5460)
    y = np.array([1] * (24 * 10) + [0] * (29 * 10), dtype=np.int32)
    groups = np.repeat(np.arange(n_subjects), n_segments)
    return X, y, groups