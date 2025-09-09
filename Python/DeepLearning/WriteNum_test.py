import torch
import torch.nn as nn
import numpy as np
from PIL import Image

import os
import matplotlib.pyplot as plt  # 用于图像可视化（确认预处理效果）


# -------------------------- 1. 定义与训练时完全一致的模型（必须和训练代码匹配） --------------------------
def create_model():
    """
    注意：此模型结构必须和训练时完全相同！
    训练时已删除最后一层Softmax（因CrossEntropyLoss内置），预测时需保留Softmax输出概率
    """
    model = nn.Sequential(
        nn.Linear(784, 444),  # 28×28=784个输入特征
        nn.ReLU(),  # 激活函数
        nn.Linear(444, 512),  # 隐藏层1
        nn.ReLU(),
        nn.Linear(512, 512),  # 隐藏层2
        nn.ReLU(),
        nn.Linear(512, 10),  # 10个输出（对应0-9）
        nn.Softmax(dim=1)  # 预测时保留，将输出转为概率（dim=1确保按行归一化）
    )
    return model


# -------------------------- 2. 图像预处理（核心优化：反色+二值化，确保匹配MNIST格式） --------------------------
def preprocess_image(image_path, need_invert=True, threshold=150):
    """
    图像预处理：读取→灰度→缩放→反色→二值化→归一化→展平
    参数说明：
        - image_path: 手写照片路径
        - need_invert: 是否需要反色（True=黑底白字→白底黑字；False=已白底黑字，无需反色）
        - threshold: 二值化阈值（100-200可调，数字模糊则减小，背景杂则增大）
    """
    # 1. 验证路径是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ 图片文件不存在：{image_path}")

    # 2. 读取图像并转灰度图（MNIST是单通道灰度图）
    img = Image.open(image_path).convert('L')  # 'L'模式=灰度图

    # 3. 缩放为28×28（MNIST标准尺寸），抗锯齿插值避免失真
    img_resized = img.resize((28, 28), Image.Resampling.LANCZOS)

    # 4. 转为numpy数组，便于后续处理
    img_np = np.array(img_resized, dtype=np.float32)

    # 5. 反色处理（关键：MNIST要求“白底黑字”）
    if need_invert:
        img_np = 255.0 - img_np  # 黑底白字 → 白底黑字（像素值反转：255→0，0→255）

    # 6. 二值化（核心：强制分离数字和背景，解决模糊问题）
    # 大于阈值→纯白背景（255），小于等于阈值→纯黑数字（0）
    img_np[img_np > threshold] = 255.0
    img_np[img_np <= threshold] = 0.0

    # 7. 归一化（MNIST训练时像素值已归一化到0-1，保持一致）
    img_normalized = img_np / 255.0  # 此时：背景=1，数字=0

    # 8. 展平为784维向量（匹配模型输入：(1, 784)，1个样本+784个特征）
    img_flatten = img_normalized.reshape(1, 784)

    # 9. 可视化预处理过程（关键：确认每一步是否正确）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 解决中文显示
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # 1行3列子图

    # 原始缩放图
    axes[0].imshow(img_resized, cmap='gray')
    axes[0].set_title(f'1. 原始缩放图\n(28×28像素)')
    axes[0].axis('off')  # 隐藏坐标轴

    # 反色后图
    axes[1].imshow(img_np, cmap='gray')
    axes[1].set_title(f'2. 反色+二值化后\n(阈值={threshold})')
    axes[1].axis('off')

    # 最终输入模型图（归一化后）
    axes[2].imshow(img_normalized, cmap='gray')
    axes[2].set_title(f'3. 最终输入模型图\n(像素值0-1)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()  # 弹出窗口显示预处理效果

    # 10. 转为Tensor并适配设备（GPU/CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = torch.tensor(img_flatten).to(device)

    return img_tensor


# -------------------------- 3. 加载训练好的模型权重 --------------------------
def load_trained_model(model_path):
    """
    加载模型权重：初始化模型→加载权重→切换为评估模式
    """
    # 1. 确认模型路径存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 模型文件不存在：{model_path}")

    # 2. 适配设备（优先GPU，无则CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ 使用设备：{device}")

    # 3. 初始化模型并加载权重
    model = create_model().to(device)
    # map_location=device：自动适配当前设备（避免训练用GPU、预测用CPU的设备不匹配问题）
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 4. 切换为评估模式（关闭Dropout等训练特有的层，不影响当前模型，但必须养成习惯）
    model.eval()
    print(f"✅ 模型加载成功：{model_path}")

    return model, device


# -------------------------- 4. 核心预测函数 --------------------------
def predict_handwritten_digit(image_path, model_path, need_invert=True, threshold=150):
    """
    完整预测流程：加载模型→预处理图像→推理预测→输出结果
    """
    try:
        # 1. 加载模型
        model, device = load_trained_model(model_path)

        # 2. 预处理图像（得到模型可识别的Tensor）
        img_tensor = preprocess_image(image_path, need_invert, threshold)

        # 3. 推理预测（关闭梯度计算，节省显存和时间）
        with torch.no_grad():
            output = model(img_tensor)  # 模型输出：(1, 10)，1个样本的10个数字概率
            predicted_prob, predicted_label = torch.max(output, dim=1)  # 取概率最大的标签

        # 4. 输出结果
        print("\n" + "=" * 50)
        print(f"📊 识别结果：数字 {predicted_label.item()}")
        print(f"📈 置信度：{predicted_prob.item():.4f}")  # 置信度=该数字的概率（越高越可信）
        print(f"📋 所有数字概率分布：")
        for i in range(10):
            print(f"   数字{i}：{output[0][i].item():.4f}")
        print("=" * 50)

        return predicted_label.item(), predicted_prob.item()

    except Exception as e:
        print(f"\n❌ 识别失败：{str(e)}")
        return None, None


# -------------------------- 5. 一键执行识别（修改路径即可用） --------------------------
if __name__ == "__main__":
    # -------------------------- 请根据你的实际情况修改以下2个路径！ --------------------------
    # 1. 手写数字照片路径（支持png/jpg/jpeg格式，用r前缀避免转义问题）
    HANDWRITTEN_IMAGE_PATH = r"D:/下载/QQ Flies/4.png"  # 你的手写照片路径
    # 2. 训练好的模型权重路径（训练时保存的mymodel.pt路径）
    TRAINED_MODEL_PATH = r"./data/mymodel.pt"  # 你的模型路径

    # -------------------------- 可选参数调整（根据预处理图像效果修改） --------------------------
    NEED_INVERT = True  # 是否需要反色：看预处理图2是否为“白底黑字”，不是则改True
    THRESHOLD = 120  # 二值化阈值：数字模糊→减小（如120），背景有杂点→增大（如180）

    # 执行识别
    predict_handwritten_digit(
        image_path=HANDWRITTEN_IMAGE_PATH,
        model_path=TRAINED_MODEL_PATH,
        need_invert=NEED_INVERT,
        threshold=THRESHOLD
    )
