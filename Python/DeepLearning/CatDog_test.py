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
        self.conv1 = nn.Conv2d(3, 20, 5, 5)
        self.conv2 = nn.Conv2d(20, 50, 4, 1)
        self.fc1 = nn.Linear(50 * 6 * 6, 200)
        self.fc2 = nn.Linear(200, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 50 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# ---------------------- 2. 加载“模型参数”到模型中（而非直接加载完整模型） ----------------------
def load_model_with_params(param_path):
    # 1. 初始化模型对象
    model = CNN()
    # 2. 加载参数字典（OrderedDict）
    param_dict = torch.load(param_path)
    # 3. 将参数加载到模型中
    model.load_state_dict(param_dict)
    # 4. 选择设备并移动模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # 切换到评估模式
    print(f"参数加载成功！模型类型：{type(model)}，使用设备：{device}")
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
    PARAM_PATH = r"./data/CatDog.pt"
    model, device = load_model_with_params(PARAM_PATH)
    transform = get_transform()

    # 预测单张图片
    TEST_IMAGE = r"C:\Users\H1441400335\Desktop\2.jpg"
    result = predict_single_image(model, device, TEST_IMAGE, transform)
    print("预测结果：", result)
