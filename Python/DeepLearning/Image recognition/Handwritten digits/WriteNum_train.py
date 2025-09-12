import torch
import torch.nn as nn
import pandas as pd

raw_df = pd.read_csv('../data/train.csv')
# print(raw_df)
# 特征
# 标签
label = raw_df['label'].values  # DataFarme -> numpy.ndarray 类型转换
raw_df = raw_df.drop(['label'], axis=1)
feature = raw_df.values / 255.0

# 划分数据集，测试集 + 训练集
train_feature = feature[:int(len(feature) * 0.8)]
train_label = label[:int(len(label) * 0.8)]
test_feature = feature[int(len(feature) * 0.8):]
test_label = label[int(len(label) * 0.8):]

train_feature = torch.tensor(train_feature).to(torch.float).cuda()
train_label = torch.tensor(train_label).cuda()
test_feature = torch.tensor(test_feature).to(torch.float).cuda()
test_label = torch.tensor(test_label).cuda()

# 序列化 在模型创建时，W和b都会被PyTorch自动初始化
model = nn.Sequential(
    nn.Linear(784, 444),  # 全连接 计算Wx+b
    nn.ReLU(),  # 激活函数
    nn.Linear(444, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
    # nn.Softmax()  # 激活函数 将输出转化为概率
).cuda()  # 放入显存

# 梯度下降
lossfunction = nn.CrossEntropyLoss()  # 交叉熵损失函数
# 优化器 Adam
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)  # 优化哪里，学习率
# 训练轮数
for i in range(100):
    # 清空优化器的梯度(偏导)
    optimizer.zero_grad()
    predict = model(train_feature)
    result = torch.argmax(predict, axis=1)
    train_acc = torch.mean((result == train_label).to(torch.float))
    loss = lossfunction(predict, train_label)
    loss.backward()  # 反向传播 计算每个 W 和 b 对误差的 “贡献度”（梯度）
    optimizer.step()  # 根据梯度调整 W 和 b 的值
    print("train loss:{} train acc:{}".format(loss.item(), train_acc.item()))

    optimizer.zero_grad()
    predict = model(test_feature)
    result = torch.argmax(predict, axis=1)
    test_acc = torch.mean((result == test_label).to(torch.float))
    loss = lossfunction(predict, test_label)
    print("test loss:{} test acc:{}".format(loss.item(), test_acc.item()))

torch.save(model.state_dict(), '..\data\mymodel.pt')  # 保存w,b

# # 加载模型文件
# params = torch.load('E:\maybe_use\mymodel.pt')
# # 把参数塞进模型
# model.load_state_dict(params)
#
# new_test_data = test_feature[100:111]
# new_test_label = test_label[100:111]
# predict = model(new_test_data)
# result = torch.argmax(predict, axis=1)
# print(new_test_label)
# print(result)
