import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import platform
import time


# ---------------------- 1. 模型结构 ----------------------
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
        return x


# ---------------------- 2. 加载模型并导出ONNX ----------------------
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


# ---------------------- 3. 加载中文字体 ----------------------
def load_pil_chinese_font():
    os_type = platform.system()
    font_path = ""
    if os_type == "Windows":
        font_path = "C:/Windows/Fonts/msyh.ttc"
        if not cv2.os.path.exists(font_path):
            font_path = "C:/Windows/Fonts/simsun.ttc"
    elif os_type == "Linux":
        font_path = "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc"
    elif os_type == "Darwin":
        font_path = "/Library/Fonts/PingFang.ttc"
    else:
        raise Exception(f"不支持的操作系统：{os_type}")

    if not cv2.os.path.exists(font_path):
        raise FileNotFoundError(f"字体文件不存在：{font_path}")
    try:
        return ImageFont.truetype(font_path, 30)
    except IOError:
        raise IOError(f"无法加载字体：{font_path}")


pil_chinese_font = load_pil_chinese_font()


# ---------------------- 4. 预处理函数（支持批处理） ----------------------
def preprocess_batch(windows):
    """批量预处理窗口，减少内存分配次数"""
    batch = []
    for window in windows:
        frame_rgb = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        tensor = transform(frame_rgb).numpy()
        batch.append(tensor)

    # 转换为批处理格式 (N, C, H, W)
    batch_np = np.stack(batch, axis=0)
    blob = cv2.dnn.blobFromImages(
        batch_np.transpose(0, 2, 3, 1),  # (N,C,H,W)→(N,H,W,C)
        swapRB=False,
        scalefactor=1.0,
        size=(150, 150)
    )
    return blob


# ---------------------- 5. 绘制中文函数 ----------------------
def draw_chinese_with_pil(frame, text, org, color=(0, 255, 0)):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_image)
    draw.text(org, text, font=pil_chinese_font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


# ---------------------- 6. 优化的滑动窗口检测函数（减少内存消耗） ----------------------
def detect_objects(frame, confidence_threshold=0.5):
    objects = []
    h, w = frame.shape[:2]

    # 优化1：减少窗口尺寸种类（从3种→2种），保留最常用的尺寸
    window_sizes = [(180, 180), (240, 240)]  # 适合猫狗的中等尺寸
    # 优化2：增大滑动步长（从50→80），减少窗口数量
    step = 80
    # 优化3：批处理窗口数量（每次处理8个窗口，减少推理次数）
    batch_size = 8
    batch_windows = []
    batch_metadata = []  # 存储窗口坐标信息

    for (win_w, win_h) in window_sizes:
        if win_w > w or win_h > h:
            continue

        # 滑动窗口
        for y in range(0, h - win_h + 1, step):
            for x in range(0, w - win_w + 1, step):
                window = frame[y:y + win_h, x:x + win_w].copy()  # 复制窗口避免引用冲突
                batch_windows.append(window)
                batch_metadata.append((x, y, x + win_w, y + win_h))

                # 当批次满了或遍历结束时，执行批处理推理
                if len(batch_windows) >= batch_size or (y == h - win_h and x == w - win_w):
                    # 批量预处理
                    blob = preprocess_batch(batch_windows)
                    net.setInput(blob)
                    outputs = net.forward()  # 一次推理处理多个窗口

                    # 解析每个窗口的结果
                    for i in range(len(batch_windows)):
                        output = outputs[i:i + 1]  # 取单个窗口的输出
                        probabilities = torch.sigmoid(torch.from_numpy(output)).numpy()[0]
                        max_prob = np.max(probabilities)
                        class_idx = np.argmax(probabilities)

                        if max_prob >= confidence_threshold:
                            x1, y1, x2, y2 = batch_metadata[i]
                            class_name = ['猫', '狗'][class_idx]
                            objects.append((x1, y1, x2, y2, class_name, max_prob))

                    # 清空批次，释放内存
                    batch_windows.clear()
                    batch_metadata.clear()

    # 非极大值抑制+保留最高置信度结果
    if len(objects) > 0:
        boxes = np.array([[x1, y1, x2, y2] for x1, y1, x2, y2, _, _ in objects])
        scores = np.array([conf for _, _, _, _, _, conf in objects])
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), confidence_threshold, 0.3)

        if len(indices) > 0:
            indices = indices.flatten()
            objects = [objects[i] for i in indices]
            objects.sort(key=lambda x: x[5], reverse=True)
            return objects[0]

    return None


# ---------------------- 7. 摄像头间隔识别 ----------------------
def camera_interval_detect():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头！")
        return

    CONFIDENCE_THRESHOLD = 0.5
    DETECT_INTERVAL = 0.5
    last_detect_time = time.time()
    last_detected_object = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("警告：无法读取摄像头帧！")
            break

        current_time = time.time()
        if current_time - last_detect_time >= DETECT_INTERVAL:
            last_detected_object = detect_objects(frame, CONFIDENCE_THRESHOLD)
            last_detect_time = current_time

        # 绘制结果
        if last_detected_object:
            x1, y1, x2, y2, class_name, confidence = last_detected_object
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            frame = draw_chinese_with_pil(
                frame, f"{class_name}：{confidence:.2f}",
                (x1, y1 - 35 if y1 > 35 else y1 + 5)
            )
        else:
            frame = draw_chinese_with_pil(frame, "未检测到猫/狗", (20, 60), (0, 0, 255))

        cv2.imshow("优化内存的猫狗检测", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_interval_detect()
