import csv
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torch.optim as optim

# 读取训练数据集
class train_CustomDataLoader(Dataset):
    def __init__(self, dirctory:str):
        self.data = []
        self.labels = []
        for filename in os.listdir(dirctory):
            if filename.endswith('.txt'):
                filepath = os.path.join(dirctory, filename)
                # 读取文件名
                label = int(filename.split('_')[0])
                self.labels.append(label) # 将标签添加到列表中

                # 获取标签
                with open(filepath, 'r',) as file:
                    content = file.readlines()[:32]  # 只读取前32行
                    my_matrix = np.array([list(map(int, line.strip())) for line in content], dtype=int)
                    # 字符串也是可迭代的对象，将每一个字符转化为int 类型，然后再转化为numpy 数组。
                    self.data.append(my_matrix.flatten()) # 将矩阵转为一维向量

        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.int32)
        # 转化为张量统一数据格式，是深度学习常用的数据格式，最终转化为一个numpy数组。

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 读取测试数据集
class test_CustomDataLoader(Dataset):
    def __init__(self, dirctory:str):
        self.data = []
        self.labels = []
        for filename in os.listdir(dirctory):
            if filename.endswith('.txt'):
                filepath = os.path.join(dirctory, filename)
                # 读取文件名
                with open(filepath, 'r',) as file:
                    content = file.readlines()[:32]  # 只读取前32行
                    my_matrix = np.array([list(map(int, line.strip())) for line in content], dtype=int)
                    # 字符串也是可迭代的对象，将每一个字符转化为int 类型，然后再转化为numpy 数组。
                    self.data.append(my_matrix.flatten()) # 将矩阵转为一维向量
        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
        # 转化为张量统一数据格式，是深度学习常用的数据格式，最终转化为一个numpy数组。

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class MulticLinearSVM(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MulticLinearSVM, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        # 定义全连接线性层 input_dim 输入维度 output_dim 输出维度。

    def forward(self, x):
        return self.linear(x)
    

def hinge_loss(output, target, model, C):
    batch_size = target.size(0) # 获取批次的大小
    hinge_loss_value = 0.0 # 循环前重置为0
    for i in range(batch_size):
        target_class = target[i]
        loss_i = torch.sum(torch.clamp(1- output[i][target_class] + output[i], min=0))
        # 这样做的目的是鼓励模型在正确类别上的预测值比其他类别的预测值至少高出 1 个单位，从而提高分类的置信度。
        # torch.clamp 将张量中所有的元素小于负数设置为0， 这也说明元素分类没有问题。如果介于 0 到 1 说明存在误差，如果为大于1的数，则存在较大的误差。
        # torch.sum 表示将张量求和
        hinge_loss_value += loss_i

    hinge_loss_value /= batch_size # 平均化
    l2_regularization = 0.5 * torch.norm(model.linear.weight)**2
    # 正则化
    total_loss = l2_regularization + C*hinge_loss_value
    return total_loss

input_size = 32 * 32 # 输入维度
output_size = 10 # 输出维度
batch_size = 64  # 批次大小

model = MulticLinearSVM(input_size, output_size)
optimizer = optim.SGD(model.parameters(), lr=0.01)

train_dataset = train_CustomDataLoader('train/') 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

test_dataset = test_CustomDataLoader('test/') 
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 训练过程
epochs = 10
def train_svm(model, train_loader):
    model.train()
    for epoch in range(epochs):
        for X_train, y_train in train_loader:
            optimizer.zero_grad()
            # 梯度化为零 
            # X_train = X_train.view(X_train.size(0), -1)  # 将数据展平成 [batch_size, 1024]
            outputs = model(X_train)
            # 代入模型中, 前向传播
            # print("outputs shape:", outputs.shape)
            # print("labels shape", y_train.shape)
            # print(outputs)
            # outputs 最终的输出结果，并不是一个数字结果，而是一个张量。

            loss = hinge_loss(outputs, y_train, model, 1)
            # loss = hinge_loss(outputs, y_train)

            # 计算损失函数
            # print(f"Loss: {loss.item()}")

            loss.backward()
            # 反响传播
            optimizer.step()
            # 进一步优化
            # print(epoch)

train_svm(model, train_loader)

# 预测测试集的结果
def test_svm(model, test_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for X_test in test_loader:
            X_test = X_test.view(X_test.size(0), -1)
            # 让每一个样本都展开成一个向量。构成一个二维张量。 torch.Size([1, 1024])
            test_outputs = model(X_test)
            # test_outputs 也是一个张量 torch.Size([1, 10])
            _, pred = torch.max(test_outputs.data, 1)
            # 返回每一列的最大值所在的索引，_ 表示最大值。 
            # torch.max 的返回形式是 torch形式，如torch([4])
            predictions.extend(pred.tolist())

    return predictions
    
predictions = test_svm(model, test_loader)

with open('result.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['num'])
    for pred in predictions:
        writer.writerow([pred])

weights = model.linear.weight.data
bias = model.linear.bias.data
# print(type(weights), weights)
print(weights.shape) # torch.Size([10, 1024])
# print(type(bias), bias)
