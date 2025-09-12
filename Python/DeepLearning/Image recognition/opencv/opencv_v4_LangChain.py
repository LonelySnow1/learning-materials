# 使用Langchain集成llm，为识别信息输出一段简短的语句

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import ImageDraw, ImageFont, Image
import time
import os
import threading  # 用于多线程监听终端输入
import init
# LangChain相关导入
from langchain.schema import HumanMessage, SystemMessage

# 线程通信信号量：need_regenerate为True时，主线程重新生成介绍文字
need_regenerate = threading.Event()
need_regenerate.clear()  # 初始状态：不触发重新生成

# ---------------------- 2. 初始化LLM与YOLO模型 ----------------------
# 初始化LLM
llm = init.chat

# 加载YOLO模型（本地路径，避免重复下载）
model = YOLO(r'../data/yolov5n.pt')


# ---------------------- 3. 中文字体加载与绘制工具 ----------------------


def draw_chinese_with_pil(frame, text, org, color=(0, 255, 0)):
    try:
        h, w = frame.shape[:2]
        x, y = org
        text_width = len(text) * 16  # 估算中文字符宽度，避免超出画面
        x = max(0, min(x, w - text_width))
        y = max(30, min(y, h - 30))

        # BGR转RGB（PIL格式）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', size=18)
        draw.text((x, y), text, font=font, fill=color)
        # 转回BGR（OpenCV格式）
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"文本绘制警告：{e}")
        return frame


# ---------------------- 4. LLM介绍文字生成函数 ----------------------
def generate_detection_intro(detected_objs, llm):
    """根据检测结果生成自然语言介绍"""
    try:
        if not detected_objs:
            return "当前画面未检测到人、猫或狗，持续监控中..."

        # 整理检测结果（简化位置描述，更易读）
        obj_details = []
        for idx, obj in enumerate(detected_objs, 1):
            x1, y1, x2, y2, cls, conf = obj
            # 按画面宽度判断位置（假设默认摄像头宽度640，可根据实际调整）
            position = f"{'左侧' if x1 < 220 else '中间' if x1 < 420 else '右侧'}"
            # 按目标面积判断体型
            area = (x2 - x1) * (y2 - y1)
            size = f"{'较小' if area < 10000 else '中等' if area < 30000 else '较大'}"
            obj_details.append(f"{idx}. {cls}（可信度{conf:.2f}）：位于画面{position}，体型{size}")

        # 构建LLM提示词（明确角色和输出要求）
        messages = [
            SystemMessage(content="""你是智能监控助手，根据检测结果生成简洁介绍，要求：
1. 口语化，无专业术语；
2. 包含目标类型、数量、大致位置和可信度；
3. 不超过3句话，适合监控画面显示；
4. 多目标按可信度从高到低排序。"""),
            HumanMessage(content=f"检测结果：{'; '.join(obj_details)}，生成介绍")
        ]

        # 调用LLM（适配新版本的invoke方法）
        response = llm(messages)
        return response.content

    except Exception as e:
        error_msg = f"介绍生成失败：{str(e)[:20]}..."
        print(f"LLM调用错误：{e}")
        return error_msg


# ---------------------- 5. 子线程：监听终端输入（检测restart指令） ----------------------
def listen_terminal_input():
    """子线程：持续监听终端输入，输入restart则触发重新生成"""
    print("\n=== 终端指令说明 ===")
    print("1. 输入 'restart' 并回车：重新生成介绍文字")
    print("2. 输入 'q' 并回车：退出程序")
    print("===================\n")

    while True:
        # 监听终端输入（阻塞式，但在子线程中不影响主线程）
        user_input = input().strip().lower()  # 忽略大小写，去除空格

        if user_input == "restart":
            print(f"[{time.strftime('%H:%M:%S')}] 收到restart指令，将重新生成介绍")
            need_regenerate.set()  # 触发主线程重新生成
        elif user_input == "q":
            print(f"[{time.strftime('%H:%M:%S')}] 收到退出指令，程序即将关闭")
            # 关闭所有OpenCV窗口，终止主线程
            cv2.destroyAllWindows()
            os._exit(0)  # 强制退出整个程序（包括主线程）
        else:
            print(f"[{time.strftime('%H:%M:%S')}] 未知指令，请输入'restart'或'q'")


# ---------------------- 6. 主线程：摄像头定时检测与画面渲染 ----------------------
def camera_timed_detect_with_llm():
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头！请检查连接/权限/是否被占用")
        return

    # 基础配置
    target_classes = {0: '人', 15: '猫', 16: '狗'}
    confidence_threshold = 0.5  # 过滤低可信度结果
    detection_interval = 0.5  # 每0.5秒检测一次YOLO
    last_detection_time = 0  # 上次YOLO检测时间戳
    detected_objs = []  # 最新检测结果
    current_intro = "初始化中，等待首次检测..."  # 当前显示的介绍文字

    # 启动子线程：监听终端输入（ daemon=True 确保主线程退出时子线程也退出）
    input_thread = threading.Thread(target=listen_terminal_input, daemon=True)
    input_thread.start()

    # 主线程循环：摄像头检测与画面渲染
    while True:
        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            print("警告：无法读取摄像头帧，重试中...")
            time.sleep(1)
            continue

        current_time = time.time()

        # ---------------------- 定时执行YOLO检测 ----------------------
        if current_time - last_detection_time >= detection_interval:
            # 执行YOLO检测（仅检测指定类别，减少计算）
            results = model(
                frame,
                classes=[0, 15, 16],
                conf=confidence_threshold,
                verbose=False  # 关闭YOLO日志
            )

            # 提取检测结果
            detected_objs = []
            if results[0].boxes:
                boxes = results[0].boxes.data.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2, conf, cls_id = box
                    detected_objs.append((
                        int(x1), int(y1), int(x2), int(y2),
                        target_classes[cls_id], round(conf, 2)
                    ))

            # 更新检测时间戳
            last_detection_time = current_time
            # 首次检测自动生成介绍
            if current_intro == "初始化中，等待首次检测...":
                current_intro = generate_detection_intro(detected_objs, llm)

        # ---------------------- 检测restart指令，重新生成介绍 ----------------------
        if need_regenerate.is_set():
            current_intro = generate_detection_intro(detected_objs, llm)
            need_regenerate.clear()  # 重置信号量，避免重复触发

        # ---------------------- 绘制画面元素 ----------------------
        # 1. 绘制目标检测框和标签
        for obj in detected_objs:
            x1, y1, x2, y2, cls, conf = obj
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框
            # 标签位置：框上方（避免遮挡）
            text_y = y1 - 10 if y1 > 40 else y2 + 30
            frame = draw_chinese_with_pil(frame, f"{cls}：{conf:.2f}", (x1, text_y), (0, 255, 0))

        # 2. 绘制LLM介绍文字（顶部居中，红色醒目）
        frame = draw_chinese_with_pil(
            frame,
            current_intro,
            org=((frame.shape[1] - len(current_intro) * 16) // 2, 10),  # 水平居中
            color=(179, 31, 69)  # 红色
        )

        # 3. 绘制检测间隔提示（底部左侧，蓝色）
        interval_text = f"检测间隔：{detection_interval}秒 | 终端输入'restart'重新生成"
        frame = draw_chinese_with_pil(
            frame,
            interval_text,
            org=(10, frame.shape[0] - 35),  # 底部留出空间
            color=(255, 0, 0)  # 蓝色
        )

        # 显示画面
        cv2.imshow("SI - YOLOv5n + LLM", frame)

        # 处理OpenCV窗口关闭事件（点击窗口X也能退出）
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("检测到窗口关闭，程序退出")
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("程序退出，资源已释放")


# ---------------------- 启动程序 ----------------------
if __name__ == "__main__":
    try:
        camera_timed_detect_with_llm()
    except Exception as e:
        print(f"程序启动失败：{str(e)}")
