# 第一步：安装依赖
# pip install ultralytics opencv-python pillow torch langchain

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import ImageDraw, ImageFont, Image
import platform
import time  # 用于控制检测间隔

# ---------------------- 1. 加载轻量级检测模型（YOLOv5n，n= nano，最小模型） ----------------------
model = YOLO(r'../data/yolov5n.pt')  # 可选：yolov5s.pt（稍大，精度更高）、yolov8n.pt（更新版本）


# 加载中文字体
def load_pil_chinese_font():
    try:
        os_type = platform.system()
        # 提供多种字体备选
        font_paths = {
            "Windows": ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simsun.ttc"],
            "Linux": ["/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc",
                      "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"],
            "Darwin": ["/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/PingFang.ttc"]
        }

        # 尝试加载第一个可用的字体
        for path in font_paths.get(os_type, []):
            try:
                return ImageFont.truetype(path, 30)
            except:
                continue

        # 如果所有字体都加载失败，使用默认字体
        return ImageFont.load_default()
    except Exception as e:
        print(f"加载字体时出错: {e}")
        return ImageFont.load_default()


pil_chinese_font = load_pil_chinese_font()


def draw_chinese_with_pil(frame, text, org, color=(0, 255, 0)):
    try:
        # 确保文本位置在图像范围内
        h, w = frame.shape[:2]
        x, y = org
        x = max(0, min(x, w - 100))  # 限制x坐标范围
        y = max(30, min(y, h - 10))  # 限制y坐标范围，确保不超出顶部

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_img)
        draw.text((x, y), text, font=pil_chinese_font, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"绘制文本时出错: {e}")
        return frame


# ---------------------- 2. 定时检测（每0.5秒一次，降低内存消耗） ----------------------
def camera_timed_detect():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头！")
        return

    # 配置：仅检测猫（15）、狗（16），置信度阈值0.5
    target_classes = {0: '人', 15: '猫', 16: '狗'}
    confidence_threshold = 0.5
    detection_interval = 0.5  # 检测间隔时间（秒）
    last_detection_time = 0  # 上次检测时间戳
    detected_objs = []  # 存储最新检测结果

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 获取当前时间
        current_time = time.time()

        # 每隔指定时间执行一次检测
        if current_time - last_detection_time >= detection_interval:
            # 执行模型推理
            results = model(frame, classes=[0, 15, 16], conf=confidence_threshold, verbose=False)

            # 更新检测结果
            detected_objs = []
            if results[0].boxes:
                boxes = results[0].boxes.data.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2, conf, cls_id = box
                    detected_objs.append((int(x1), int(y1), int(x2), int(y2), target_classes[cls_id], conf))

            # 更新上次检测时间
            last_detection_time = current_time
            # 可选：显示检测时间点，方便调试
            # print(f"执行检测: {time.strftime('%H:%M:%S', time.localtime(current_time))}")

        # 绘制最新检测结果（即使未执行新检测，也显示上一次的结果）
        for obj in detected_objs:
            x1, y1, x2, y2, cls, conf = obj
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 更可靠的文本位置计算
            text_y = y1 - 10 if y1 > 40 else y2 + 30
            frame = draw_chinese_with_pil(frame, f"{cls}：{conf:.2f}", (x1, text_y))

        # 显示检测间隔信息
        frame = draw_chinese_with_pil(frame, f"检测间隔: {detection_interval}秒", (10, 30), (255, 0, 0))

        cv2.imshow("SI - YOLOv5n", frame)  # Scheduled inspection 定时检测
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_timed_detect()
