"""模型 1: SVM 支持向量机 (基线模型)。
RBF 核 + StandardScaler 管道, 高维小样本经典方案。"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
def build_svm(kernel='rbf', C=1.0, gamma='scale'):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel=kernel, C=C, gamma=gamma,
                    probability=True, random_state=42)),
    ])