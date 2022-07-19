import scipy
import numpy as np

class LDA():
    def __init__(self, n_components=5):
        self.n_components = n_components
    
    def fit(self, train_data, train_labels, solver="eigen"):
        """使用线性判别分析对train_data进行降维，
        维度为self.n_components

        Args:
            train_data (np.array(n_samples, features)): _description_
            train_labels (n_samples): _description_
        """
        if solver == "eigen":
            self.project_matrix = self._solver_eigen(train_data, train_labels)
        return self.transform(train_data)

    def _solver_eigen(self, train_data, train_labels):
        """使用特征值求解的方式进行LDA

        Args:
            train_data (_type_): _description_
            train_labels (_type_): _description_

        Returns:
            _type_: _description_
        """
        n_samples = train_data.shape[0]
        n_features = train_data.shape[1]
        labels_unique = np.unique(train_labels)
        # 首先计算各个类的均值
        # 然后计算类内散度矩阵Sw
        # 两个循环合并
        means = []
        Swi = []
        for label in labels_unique:
            Xg = train_data[train_labels==label, :]          # 取出相同标签的数据
            means.append(np.mean(Xg, axis=0))
            Swi.append(np.cov(Xg, rowvar=False))        # cov默认为列向量，这里设置rowvar从而以行向量为单个样本
        train_data_mean = np.average(means, axis=0)
        Sw = np.average(Swi, axis=0)
        # 计算类间散度矩阵
        # 可以通过总体散度矩阵St-Sw得到Sb
        St = np.cov(train_data, rowvar=False)
        Sb = St - Sw
        # 计算特征值和特征向量
        evals, evecs = np.linalg.eig(np.dot(np.linalg.inv(Sw), Sb))
        # eig输出并不严格有序，需要根据特征值对特征向量进行排序
        evecs = evecs[:, np.argsort(evals)[::-1]]
        project_matrix = np.real(evecs[:, :self.n_components])
        return project_matrix

    def transform(self, test_data):
        """使用训练好的LDA对test_data进行降维

        Args:
            test_data (np.array(n_samples, features)): _description_
        Return:
            X_new (np.array(n_samples, n_components))
        """
        return np.dot(test_data, self.project_matrix)

