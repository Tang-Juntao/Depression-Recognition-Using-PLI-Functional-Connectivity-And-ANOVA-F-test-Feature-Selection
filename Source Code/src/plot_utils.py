"""可视化工具模块。
输出: 分组柱状图 (6 指标对比)、ROC 曲线叠加图、混淆矩阵热力图。"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import confusion_matrix, roc_curve, auc

_CJK_CANDIDATES = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'STSong', 'FangSong']
_available = {f.name for f in fm.fontManager.ttflist}
_cjk = [f for f in _CJK_CANDIDATES if f in _available]
if _cjk:
    plt.rcParams['font.sans-serif'] = _cjk + ['DejaVu Sans', 'Arial']
    plt.rcParams['font.family'] = 'sans-serif'
else:
    plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams.update({
    'font.size': 10,
    'axes.unicode_minus': False,
    'figure.dpi': 150,
})

METRIC_NAMES = ['ACC', 'SEN\n(Recall)', 'SPE', 'PRE', 'F1', 'AUC']
MODEL_NAMES = ['SVM\n(模型1)', 'MLP\n(模型2)', '特征选择+SVM\n(改进模型3)']
def plot_comparison_bar(means_list, stds_list, save_path, model_names=None):
    if model_names is None:
        model_names = MODEL_NAMES

    n_metrics = len(METRIC_NAMES)
    n_models = len(means_list)
    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ['#4472C4', '#ED7D31', '#70AD47']

    for i, (means, stds) in enumerate(zip(means_list, stds_list)):
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, means * 100, width, yerr=stds * 100,
                      capsize=3, color=colors[i % len(colors)],
                      edgecolor='white', linewidth=0.5, label=model_names[i])
        for bar, val in zip(bars, means * 100):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_NAMES, fontsize=9)
    ax.set_ylabel('百分比 (%)')
    ax.set_ylim(0, 110)
    ax.legend(loc='lower right', fontsize=8)
    ax.set_title('三模型性能对比 (10折交叉验证, 均值±标准差)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f'[图表] 柱状图已保存: {save_path}')


def plot_roc_curves(roc_data_list, auc_values, save_path, model_names=None):
    if model_names is None:
        model_names = MODEL_NAMES

    fig, ax = plt.subplots(figsize=(6, 5.5))
    colors = ['#4472C4', '#ED7D31', '#70AD47']

    for i, ((mean_fpr, mean_tpr, std_tpr), auc_val) in \
            enumerate(zip(roc_data_list, auc_values)):
        ax.plot(mean_fpr, mean_tpr, color=colors[i], lw=1.8,
                label=f'{model_names[i]} (AUC={auc_val:.3f})')
        ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                        color=colors[i], alpha=0.12)

    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='随机猜测 (AUC=0.500)')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('假阳性率 (FPR)', fontsize=10)
    ax.set_ylabel('真阳性率 (TPR / Recall)', fontsize=10)
    ax.set_title('ROC 曲线对比 (10折交叉验证)', fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f'[图表] ROC 曲线已保存: {save_path}')
def plot_confusion_matrix(y_true, y_pred, save_path, title=None):
    """绘制混淆矩阵热力图。"""
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    # cm[0,0]=TP(MDD), cm[0,1]=FN(MDD→HC), cm[1,0]=FP(HC→MDD), cm[1,1]=TN(HC)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap='Blues', vmin=0)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                    fontsize=18, fontweight='bold',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['MDD (预测)', 'HC (预测)'], fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['MDD (真实)', 'HC (真实)'], fontsize=10)
    if title:
        ax.set_title(title, fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.82)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f'[图表] 混淆矩阵已保存: {save_path}')

def plot_k_sensitivity(k_list, scores_list, save_path):
    """绘制特征数 k 敏感性曲线。"""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_list, [s * 100 for s in scores_list], 'o-',
            color='#70AD47', lw=1.5, markersize=6)
    ax.set_xlabel('保留特征数 k')
    ax.set_ylabel('准确率 ACC (%)')
    ax.set_title('特征数 k 对准确率的影响')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f'[图表] k 敏感性曲线已保存: {save_path}')