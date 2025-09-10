import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont  # 导入PIL库
import platform


# ---------------------- 1. 模型结构（与之前一致，省略重复代码） ----------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(256 * 9 * 9, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = x.view(-1, 256 * 9 * 9)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# ---------------------- 2. 加载模型并导出ONNX（与之前一致，省略重复代码） ----------------------
model = torch.load('./data/CatDog.pt')
model.eval()
device = next(model.parameters()).device
dummy_input = torch.randn(1, 3, 150, 150, device=device)
torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    verbose=False,
    input_names=['input'],
    output_names=['output'],
    opset_version=11
)
net = cv2.dnn.readNetFromONNX('model.onnx')


# ---------------------- 3. 新增：用PIL加载中文字体（替代cv2.freetype） ----------------------
def load_pil_chinese_font():
    """用PIL加载中文字体，返回字体对象"""
    os_type = platform.system()
    font_path = ""

    # 1. 适配不同操作系统的中文字体路径（与之前一致）
    if os_type == "Windows":
        font_path = "C:/Windows/Fonts/msyh.ttc"  # 微软雅黑
        if not cv2.os.path.exists(font_path):
            font_path = "C:/Windows/Fonts/simsun.ttc"  # 宋体备选
    elif os_type == "Linux":
        font_path = "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc"  # 文泉驿正黑
    elif os_type == "Darwin":  # macOS
        font_path = "/Library/Fonts/PingFang.ttc"  # 苹方
    else:
        raise Exception(f"不支持的操作系统：{os_type}，请手动指定中文字体路径")

    # 2. 检查字体文件是否存在
    if not cv2.os.path.exists(font_path):
        raise FileNotFoundError(f"中文字体文件不存在！路径：{font_path}\n请手动修改font_path")

    # 3. 用PIL加载字体（字体大小30，可调整）
    try:
        font = ImageFont.truetype(font_path, 30)  # 30是字体大小
    except IOError:
        raise IOError(f"PIL无法加载字体文件，请检查字体路径是否正确：{font_path}")
    return font


# 加载PIL中文字体（程序启动时执行一次）
pil_chinese_font = load_pil_chinese_font()


# ---------------------- 4. 预处理函数（与之前一致，省略重复代码） ----------------------
def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    frame_tensor = transform(frame_rgb).unsqueeze(0)
    frame_np = frame_tensor.numpy().squeeze(0).transpose(1, 2, 0)
    blob = cv2.dnn.blobFromImage(
        frame_np,
        swapRB=False,
        scalefactor=1.0,
        mean=None,
        size=(150, 150)
    )
    return blob


# ---------------------- 5. 新增：用PIL绘制中文，再转回OpenCV帧 ----------------------
def draw_chinese_with_pil(frame, text, org, color=(0, 255, 0)):
    """
    用PIL在OpenCV帧上绘制中文
    :param frame: OpenCV的BGR帧（numpy数组）
    :param text: 要绘制的中文文本
    :param org: 文本左上角坐标 (x, y)
    :param color: 文本颜色（OpenCV的BGR格式，默认绿色）
    :return: 绘制中文后的OpenCV帧
    """
    # 1. OpenCV帧（BGR）→ PIL图像（RGB）
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # 2. 用PIL绘制中文
    draw = ImageDraw.Draw(pil_image)
    x, y = org
    # 注意：PIL的颜色是RGB格式，需将OpenCV的BGR转为RGB
    draw.text(
        (x, y),
        text,
        font=pil_chinese_font,
        fill=(color[2], color[1], color[0])  # BGR→RGB（OpenCV是B，G，R；PIL是R，G，B）
    )

    # 3. PIL图像（RGB）→ OpenCV帧（BGR）
    frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return frame_bgr


# ---------------------- 6. 摄像头实时识别（修改为PIL绘制中文） ----------------------
def camera_real_time_detect():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头！")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("警告：无法读取摄像头帧！")
            break

        # 预处理+推理（与之前一致）
        blob = preprocess_frame(frame)
        net.setInput(blob)
        output = net.forward()
        output_tensor = torch.from_numpy(output)
        confidences = torch.exp(output_tensor)
        max_conf, pred_idx = torch.max(confidences, 1)
        class_names = ['猫', '狗']
        pred_class = class_names[pred_idx.item()]
        result_text = f"类别：{pred_class} | 置信度：{max_conf.item():.2f}"  # 中文文本

        # ---------------------- 关键修改：用PIL绘制中文 ----------------------
        # 替代原cv2.freetype绘制，调用draw_chinese_with_pil函数
        frame = draw_chinese_with_pil(
            frame,
            result_text,
            org=(20, 60),  # 文本左上角坐标
            color=(0, 255, 0)  # 绿色（OpenCV的BGR格式）
        )

        # 绘制边框（与之前一致）
        cv2.rectangle(frame, (10, 10), (frame.shape[1] - 10, frame.shape[0] - 10), (0, 255, 0), 2)

        # 显示画面
        cv2.imshow("Cat/Dog Real-Time Detect（中文支持）", frame)

        # 退出逻辑
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("结束实时识别。")
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------- 7. 启动程序 ----------------------
if __name__ == "__main__":
    camera_real_time_detect()
