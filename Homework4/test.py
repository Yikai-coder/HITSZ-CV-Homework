import numpy as np
import os 
import cv2
import random
import gc
import time
from pca import PCA
from kpca import KPCA
from lda import LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import decomposition

def load_data(path):
    imgs = []
    labels = []
    os.chdir(path)
    for dir in os.listdir():
        # print(root.split("Person"))
        label = int(dir.split("Person")[1])
        for file in os.listdir(dir):
            img = cv2.imread(os.path.join(os.getcwd(), dir, file), cv2.IMREAD_GRAYSCALE).flatten() # 读取数据并转换为1维长向量
            imgs.append(img)
            labels.append(label)
    print(len(imgs))
    print(len(labels))
    return imgs, labels

def split_train_test(data, labels, rate):
    assert len(data) == len(labels)
    total_num = len(data)
    train_num = int(rate * len(data))
    test_num = len(data) - train_num
    train_data = []
    train_labels = []
    test_labels = []
    test_data = []
    
    test_selected = []
    train_selected = []
    num = 0
    while num < test_num:
        idx = random.randint(0, total_num-1)
        if not idx in test_selected:
            test_selected.append(idx)
            num +=1
            
    train_selected = list(set(list(range(total_num))).difference(set(test_selected)))
    
    for idx in test_selected:
        test_data.append(data[idx])
        test_labels.append(labels[idx])
    
    for idx in train_selected:
        train_data.append(data[idx])
        train_labels.append(labels[idx])
        
    return np.array(train_data).T, np.array(train_labels), np.array(test_data).T, np.array(test_labels)

def split_data_in_labels(data, labels):
    assert len(data) == len(labels)
    classes = len(labels) / 10
    test_data = []
    train_data = []
    test_labels = []
    train_labels = []
    test_selected = []
    train_selected = []
    for i in range(int(classes)):
        test_selected.append(i*10 + random.randint(0, 9))
        test_selected.append(i*10 + random.randint(0, 9))

    train_selected = list(set(list(range(len(data)))).difference(set(test_selected)))
    
    for idx in test_selected:
        test_data.append(data[idx])
        test_labels.append(labels[idx])
    
    for idx in train_selected:
        train_data.append(data[idx])
        train_labels.append(labels[idx])
    return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)


import matplotlib as mpl
import matplotlib.pyplot as plt

# 读取数据，划分训练集
data, labels = load_data("./data")
# train_data, train_labels, test_data, test_labels = split_train_test(data, labels, 0.8)
train_data, train_labels, test_data, test_labels = split_data_in_labels(data, labels)

del data, labels
gc.collect()
print(train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)

# 统一分类器
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=1)

acc1 = []
acc2 = []
acc3 = []
k = list(range(1, 11))
# 使用PCA对训练数据进行降维，用于LDA
pca_tmp = PCA(40) 
train_data_new = pca_tmp.fit(train_data)
test_data_new = pca_tmp.transform(test_data)
# clf.fit(train_data_new, train_labels)
# baseline = clf.score(test_data_new, test_labels)
# print(baseline)

for i in k:
    pca = PCA(i)
    train_feats = pca.fit(train_data)
    test_feats = pca.transform(test_data)
    clf.fit(train_feats, train_labels)
    acc1.append(clf.score(test_feats, test_labels))

    kpca = KPCA(i, 10000)    # magic_num for gausian_kernel_filter
    train_feats = kpca.fit(train_data)
    test_feats = kpca.transform(test_data)
    clf.fit(train_feats, train_labels)
    acc2.append(clf.score(test_feats, test_labels))

    lda = LDA(i)
    train_feats = pca.fit(train_data_new)
    test_feats = pca.transform(test_data_new)
    clf.fit(train_feats, train_labels)
    acc3.append(clf.score(test_feats, test_labels))

with open("./acc.txt", 'w') as f:
    f.write("PCA:\n")
    for i in range(len(acc1)):
        f.write("k=%d, acc=%.5f\n" % (k[i], acc1[i]))
    f.write("KPCA:\n")
    for i in range(len(acc2)):
        f.write("k=%d, acc=%.5f\n" % (k[i], acc2[i]))
    f.write("LDA:\n")
    for i in range(len(acc2)):
        f.write("k=%d, acc=%.5f\n" % (k[i], acc3[i]))


fig = plt.figure()
ax = plt.subplot()
ax.plot(k, acc1, label="PCA")
ax.plot(k, acc2, label="KPCA")
ax.plot(k, acc3, label="LDA")
# ax.plot(k, [baseline] * len(k), label="LDA baseline")
plt.legend(loc=4)
plt.show()
