import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np

# 读取训练数据
train_data = pd.read_csv('train/train.csv')
X_train = train_data.iloc[:, :-1] # 特征
y_train = train_data.iloc[:,-1] # 标签

# 将数据转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
# view 方法用于改变张量的形状（类似于 NumPy 的 reshape 方法）。-1 表示该维度的大小由其他维度自动推断。在这种情况下，-1 会根据 y_train 的总元素数量来计算合适的值。
# 1 表示第二个维度的大小为 1。

# 定义自定义数据集类
class FishDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    #__len__：这个方法返回数据集的长度，即数据集中样本的数量。这里返回 self.X 的长度，因为 self.X 包含了所有的输入数据。
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]
    # 通过索引返回数据集中的样本

# 实例化数据集和数据加载器对象
train_dataset = FishDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 定义模型
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x) # 这里返回的是预测值，而不是实际值。

# 超参数
input_dim = X_train.shape[1] # X_train.shape[1] 表示 X_train 的特征数量。
learning_rate = 0.01
num_epochs = 100 # 如果 num_epochs 设置为 100，那么在整个训练过程中，数据集将被完整遍历 100 次。

# 初始化模型、损失函数和优化器
model = LinearRegressionModel(input_dim)
criterion = nn.MSELoss()  # 定义了损失函数即均方误差损失函数。
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # 优化器使用随机梯度下降（SGD）算法，学习率设为 0.01。

# 训练模型
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader): # i 前批次的索引（从 0 开始）。(inputs, labels)：当前批次的数据和标签。
        # 前向传播
        outputs = model(inputs) # 前向传播，计算预测值。
        loss = criterion(outputs, labels) # 计算损失,outputs和y之间的均方误差。

        # 反向传播和优化
        optimizer.zero_grad() # 梯度清零
        loss.backward() # 计算梯度
        optimizer.step() # 更新参数

    # if (epoch+1) % 10 == 0:
    #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
# 外层循环，迭代 num_epochs 次，内层循环遍历 train_loader，每次迭代计算损失并更新参数。

# 读取测试数据
test_data = pd.read_csv('test/test.csv')
X_test = test_data.iloc[:, :]  # 测试集只有特征没有标签

# 将数据转换为 PyTorch 张量
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

# 创建数据加载器
test_dataset = FishDataset(X_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 对测试集进行预测
model.eval()  # 设置模型为评估模式
predictions = [] # 初始化一个空列表 predictions，用于存储模型的预测结果。

with torch.no_grad(): # 禁用梯度计算，以加速预测。
    for inputs in test_loader:
        outputs = model(inputs)
        predictions.extend(outputs.numpy().flatten())
# 将outputs转换为 NumPy 数组，最终展开为一维数组，并将其添加到 predictions 列表中。

predictions = np.array(predictions)

# 将预测结果保存到新的CSV文件
result_df = pd.DataFrame(predictions, columns=['age'])
result_df.to_csv('result.csv', index=False)
