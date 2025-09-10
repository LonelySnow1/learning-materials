import os

import numpy as np
from torchvision import transforms, datasets

# 定义transforms
transforms = transforms.Compose(
    [

        transforms.RandomResizedCrop(150),  # 随机裁剪 把图像变成（150,150）
        transforms.ToTensor(),  # 归一化，转tensor张量 -> [3,150,150]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化 (像素值 - 通道均值[mean]) / 通道标准差[std]
                             std=[0.229, 0.224, 0.225])  # ImageNet 数据集的通用参数，也是pytorch官方与预训练模型的规定参数
    ]
)

file_path = r"D:\下载\data"
tr = "train"
te = "test"

train_data = datasets.ImageFolder(os.path.join(file_path, tr), transforms)
test_data = datasets.ImageFolder(os.path.join(file_path, te), transforms)

from torch.utils import data

batch_size = 32
train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = data.DataLoader(test_data, batch_size=batch_size)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# 架构会有很大的不同，因为28*28-》150*150,变化挺大的，这个步长应该快一点。
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积块1：Conv + BatchNorm + ReLU + MaxPool
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 3→32通道，padding=1保持尺寸
            nn.BatchNorm2d(32),  # BatchNorm：加速收敛，稳定训练
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 尺寸减半：150→75
        )
        # 卷积块2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 32→64通道
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 75→37（整数除法）
        )
        # 卷积块3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 64→128通道
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 37→18
        )
        # 卷积块4
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 128→256通道
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 18→9
        )
        # 全连接层：先计算输入维度（256通道×9×9尺寸）
        self.fc1 = nn.Linear(256 * 9 * 9, 512)  # 高维特征压缩
        self.dropout = nn.Dropout(0.5)  # Dropout：防止过拟合，提升泛化
        self.fc2 = nn.Linear(512, 2)  # 最终2分类

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = x.view(-1, 256 * 9 * 9)  # 展平：(batch, 256*9*9)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # 应用Dropout
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)


def train(model, device, train_loader, optimizer, epoch, losses):
    model.train()
    for idx, (t_data, t_target) in enumerate(train_loader):
        t_data, t_target = t_data.to(device), t_target.to(device)
        pred = model(t_data)  # batch_size*2
        loss = F.nll_loss(pred, t_target)

        # Adam
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 10 == 0:
            print("epoch:{},iteration:{},loss:{}".format(epoch, idx, loss.item()))
            losses.append(loss.item())


def test(model, device, test_loader):
    model.eval()
    correct = 0  # 预测对了几个。
    total_test_loss = 0.0  # 新增：统计测试集总损失
    with torch.no_grad():
        for idx, (t_data, t_target) in enumerate(test_loader):
            t_data, t_target = t_data.to(device), t_target.to(device)
            pred = model(t_data)  # batch_size*2
            total_test_loss += F.nll_loss(pred, t_target, reduction='sum').item()  # 累加损失
            pred_class = pred.argmax(dim=1)  # batch_size*2->batch_size*1
            correct += pred_class.eq(t_target.view_as(pred_class)).sum().item()
    acc = correct / len(test_data)
    average_loss = total_test_loss / len(test_data)  # 计算平均损失
    print(f"accuracy:{acc:.4f}, average_loss:{average_loss:.4f}")


num_epochs = 10
losses = []
from time import *

begin_time = time()
for epoch in range(num_epochs):
    train(model, device, train_loader, optimizer, epoch, losses)
# test(model,device,test_loader)
end_time = time()
print(f"总训练时间：{end_time - begin_time:.2f}秒")  # 打印总耗时

# torch.save(model.state_dict(), '.\data\CatDog.pt')  # 保存w,b

torch.save(model, '.\data\CatDog.pt')  # 保存完整模型
