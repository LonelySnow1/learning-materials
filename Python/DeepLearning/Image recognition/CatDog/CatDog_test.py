import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os


# ---------------------- 1. 关键：先重新定义与训练时完全一致的CNN模型结构 ----------------------
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


# ---------------------- 2. 加载“模型参数”到模型中 ----------------------
# def load_model_with_params(param_path):
#     # 1. 初始化模型对象
#     model = CNN()
#     # 2. 加载参数字典（OrderedDict）
#     param_dict = torch.load(param_path)
#     # 3. 将参数加载到模型中
#     model.load_state_dict(param_dict)
#     # 4. 选择设备并移动模型
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     model.eval()  # 切换到评估模式
#     print(f"参数加载成功！模型类型：{type(model)}，使用设备：{device}")
#     return model, device


def load_complete_model(model_path):
    # 1. 直接加载完整模型（包含结构和参数）
    # 处理设备兼容性：若模型保存时在GPU，加载时自动映射到可用设备（CPU/GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)  # 直接加载完整模型

    # 2. 切换到评估模式（关闭dropout/batchnorm等训练相关层）
    model.eval()

    print(f"完整模型加载成功！模型类型：{type(model)}，使用设备：{device}")
    return model, device


# ---------------------- 3. 后续预测代码（和之前一致） ----------------------
def get_transform():
    return transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def predict_single_image(model, device, image_path, transform):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return f"图片读取失败：{str(e)}"
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_prob = torch.exp(model(image_tensor))
        pred_idx = pred_prob.argmax(dim=1).item()
        pred_conf = pred_prob[0][pred_idx].item()
    return {"预测类别": ["猫", "狗"][pred_idx], "置信度": round(pred_conf, 4)}


# ---------------------- 4. 执行预测 ----------------------
if __name__ == "__main__":
    # 注意：这里传入的是“参数文件路径”（best_model_params.pth）
    PARAM_PATH = r"../data/CatDog.pt"
    # model, device = load_model_with_params(PARAM_PATH)
    model, device = load_complete_model(PARAM_PATH)
    transform = get_transform()

    # 预测单张图片
    TEST_IMAGE = r"C:\Users\H1441400335\Desktop\2.jpg"
    result = predict_single_image(model, device, TEST_IMAGE, transform)
    print("预测结果：", result)
