import numpy as np

class PCA():
    def __init__(self, k):
        self.k = k
    
    def fit(self, X):
        """PCA数据降维

        Args:
            X (np.array (n_samples, n_features)): _description_

        Returns:
            feat_space (n_samples, k)： 降维后的训练数据
        """
        n_samples, n_features = X.shape
        assert n_features >= self.k
        self.train_num = X.shape[1]
        self.X_mean = np.mean(X, axis=0)
        X_std = (X-self.X_mean)
        
        # faster
        cov = np.cov(X.T, rowvar=False)
        # u1:(n_samples, n_samples)
        w1, u1 = np.linalg.eig(cov)
        # u1: (n_samples, n_samples)
        u1 = np.dot(X_std.T, u1)
        # project_matrix:(n_features, k)
        project_matrix = np.real(u1[:, np.argsort(w1)[::-1][:self.k]])
        # feat_space:(n_samples, k)
        feat_space = np.dot(X_std, project_matrix)
        self.project_matrix = project_matrix
        return feat_space

    def transform(self, X, project_matrix=None):
        """对数据进行降维

        Args:
            X (n_samples, n_features): 数据
            project_matrix (n_features, k): 投影矩阵
        """
        if project_matrix is None:
            return np.dot((X-self.X_mean), self.project_matrix)
        else:
            return np.dot((X.T-self.X_mean), project_matrix)

    