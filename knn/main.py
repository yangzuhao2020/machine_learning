# 导入所需的库
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 加载训练数据
data = pd.read_csv('train/train.csv')

# 分割特征和结果
X = data.iloc[:, 1:5]  # 保留特征
y = data.iloc[:, -1]  # 保留结果

# 数据预处理，化为标准正态分布，以提高运行效率
scaler = StandardScaler()
X_train = scaler.fit_transform(X)
y_train = list(y)

# 初始化并训练KNN分类器
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 加载测试数据
test_data = pd.read_csv('test/test.csv')

# 获取特征
X = test_data.iloc[:, 1:5]  # 特征

# 数据预处理，化为标准正态分布
scaler = StandardScaler()
X_test = scaler.fit_transform(X)

# 预测
y_pred = knn.predict(X_test)

df = pd.DataFrame(data=test_data['Id'], columns=['Id'])
df['Species'] = y_pred
df.to_csv('result.csv', index=False)

