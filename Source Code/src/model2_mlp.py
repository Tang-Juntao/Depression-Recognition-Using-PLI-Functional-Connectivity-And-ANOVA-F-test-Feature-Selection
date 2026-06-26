"""
模型 2: MLP 多层感知机 (深度学习代表)。
小隐层 + L2 正则 + 早停, 适配 530 样本 / 5460 维的高维小样本场景。
基于 sklearn MLPClassifier, 不依赖 PyTorch/TensorFlow。
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
def build_mlp(hidden_layer_sizes=(256, 64), alpha=1e-3, max_iter=500):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            alpha=alpha,
            max_iter=max_iter,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.1,
            random_state=42,
        )),
    ])
