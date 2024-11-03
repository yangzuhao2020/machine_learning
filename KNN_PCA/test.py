import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据集
data = pd.read_csv("train/train.csv")
X = data.iloc[:, 2:]
y = data.iloc[:, 1]

# 将数据集分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42) # 20% 用于验证

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# PCA降维
pca = PCA(n_components=20)
X_train = pca.fit_transform(X_train)
X_val = pca.transform(X_val)

# 建立和训练模型
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 在验证集上预测
y_val_pred = knn.predict(X_val)

# 计算验证集的准确度
accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {accuracy:.4f}")
