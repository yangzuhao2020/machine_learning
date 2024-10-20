import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
from sklearn.preprocessing import LabelEncoder
import xgboost

# load train data
train_data = pd.read_csv('train/train.csv')
# 加载测试数据集
test_data = pd.read_csv('test/test.csv')

X_train = train_data.iloc[:,:-1] # 特征
X_train.replace('?', np.nan, inplace=True)  # 替换问号为缺失值

X_test = test_data.iloc[:,:] # 获取特征
X_test.replace('?', np.nan, inplace=True) # 替换问号为缺失值

y_train = train_data.iloc[:,-1] # income label
# print(type(X_train))

# 定义需要编码的分类特征
categorical_columns = ['workplace', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']

le = LabelEncoder() # 初始化 LabelEncoder

# 对每一个分类进行编码 
for col in categorical_columns:
    X_train[col] = X_train[col].fillna(X_train[col].mode()[0]) # 用众数替换缺失的地方
    X_train[col] = le.fit_transform(X_train[col]) # 使用 LabelEncoder 对当前列进行编码。
    
for col in categorical_columns:
    X_test[col] = X_test[col].fillna(X_test[col].mode()[0]) # 用众数替换缺失值
    X_test[col] = le.fit_transform(X_test[col]) # 对测试数据集进行编码

model = xgboost.XGBClassifier(n_estimators=200, eval_metric='logloss')
# xgboost.XGBClassifier 是 XGBoost 库中用于分类任务的类
# n_estimators=200 指明决策树的数量
# eval_metric='logloss' 损失函数采用对数损失。

# 划分验证集合 和 训练集合
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=60)

# 模型训练
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# 在验证集合上进行预测
val_predictions = model.predict(X_val)

# 对测试数据集进行预测
test_predictions = model.predict(X_test)

# 生成输出文件
output = pd.DataFrame({'income':test_predictions})
output.to_csv('result.csv', index=False)
