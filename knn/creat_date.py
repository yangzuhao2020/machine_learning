from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

# 加载鸢尾花数据集
iris_dataset = load_iris()
# 划分训练集和测试集，一共150份数据，按9:1划分数据集
X_train, X_test, y_train, y_test = train_test_split(iris_dataset.data, iris_dataset.target, test_size=0.1)

df1 = pd.DataFrame(data=X_train,
                   columns=['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width'])
df1['Species'] = y_train

df2 = pd.DataFrame(data=X_test,
                   columns=['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width'])
df2['Species'] = y_test

df1.to_csv('train/train.csv', index_label='Id')  # 训练集
df2.to_csv('test/test.csv', index_label='Id')  # 测试集