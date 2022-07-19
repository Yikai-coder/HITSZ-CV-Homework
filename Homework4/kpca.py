import numpy as np

class KPCA():
    def __init__(self, k, sigma):
        self.k = k
        self.sigma = sigma
    
    @staticmethod
    def kernel_filter_gaussian(X, sigma):
        """计算K矩阵
        Args:
            X (np.array (n_samples, n_features)): _description_
            sigma: 高斯核参数
            
        Returns:
            K (n_samples, n_samples)
        """
        K = np.zeros((X.shape[0], X.shape[0]))
        for i in range(K.shape[0]):
            for j in range(K.shape[1]):
                K[i, j] = np.exp(-np.linalg.norm(X[i, :]-X[j, :])**2 / (2*sigma**2))                
        return K
    
    @staticmethod
    def kernel_filter_gausian_xi(X, xi, sigma):
        k = np.zeros(X.shape[0])
        for i in range(k.shape[0]):
            k[i] = np.exp(-np.linalg.norm(X[i, :]-xi)**2 / (2*sigma**2))   
        
        return k

    def kernel_filter_multinominal(self, X, a, c, d):
        """𝑘(𝑥,𝑦)=(𝑎𝑥^𝑇𝑦+𝑐)^𝑑

        Args:
            X (_type_): _description_
            a (_type_): _description_
            c (_type_): _description_
            d (_type_): _description_

        Returns:
            _type_: _description_
        """
        K = np.zeros((X.shape[1], X.shape[1]))
        for i in range(K.shape[0]):
            # for j in range(K.shape[1]):
            #     K[i, j] = np.dot(X[:, i], X[:, j]) + c 
            K[i] = self.multinominal_kernel(X, X[:, i], a, c, d)            
        return K
    @staticmethod
    def multinominal_kernel(X, xi, a, c, d):
        k = np.zeros(X.shape[1])
        for i in range(k.shape[0]):
            k[i] = (np.dot(X[:, i], xi)*a + c)**d
        
        return k       
    
    def fit(self, X):
        """KPCA数据降维

        Args:
            X (np.array (n_samples, n_features)): _description_
            sigma: 高斯核参数

        Returns:
            feat_space (n_samples, k)： 降维后的训练数据
        """
        n_samples, n_features = X.shape
        assert n_features >= self.k
        self.train_num = X.shape[0]
        self.X_mean = np.mean(X, axis=0)
        X_std = (X - self.X_mean)
        self.X = X_std
        # K_matrix = self.kernel_filter_multinominal(X, 1/40, 0, 2)
        # K_matrix: (n_samples. n_samples)
        K_matrix = self.kernel_filter_gaussian(X_std, self.sigma)
        # faster
        # u1:(sample_num, sample_num)
        # print("K_matrix:", K_matrix)
        w1, u1 = np.linalg.eig(K_matrix)
        # project_matrix:(n_samples, k)
        project_matrix = np.real(u1[:, np.argsort(w1)[::-1][:self.k]])
        self.project_matrix = project_matrix
        # print(w1, u1)
        # feat_space:(n_samples, k)
        feat_space = np.dot(K_matrix, project_matrix)
        return feat_space

    def transform(self, X, project_matrix=None):
        """利用训练好的KPCA对数据进行降维

        Args:
            X ((n_test_samples, n_features)): _description_
            project_matrix (project_matrix (n_samples, k), optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        K = np.zeros([X.shape[0], self.k])
        X = (X-self.X_mean)
        for i in range(X.shape[0]):
            K[i, :] = np.dot(self.kernel_filter_gausian_xi(self.X, X[i, :], self.sigma), self.project_matrix)
        return K

