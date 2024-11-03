import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 加载数据集
data = pd.read_csv("train/train.csv")
X_train = data.iloc[:,2:]
y_train = data.iloc[:,1]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # 获取平均值 然后进行标准化。
y_train = list(y_train)

# 降维
pca = PCA(n_components = 15)
X_train = pca.fit_transform(X_train)

# 加载测试数据
test_data = pd.read_csv('test/test.csv')
X_test = test_data.iloc[:,1:]

X_test = scaler.transform(X_test) # 直接获取训练数据集的平均值进行标准化。

# 降维
X_test = pca.transform(X_test)

# 建立训练模型
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)

# 预测
y_pred = knn.predict(X_test)
df = pd.DataFrame(data=test_data["ID"],columns=['ID'])
df["Diagnosis"] = y_pred
df.to_csv('result.csv', index=False)