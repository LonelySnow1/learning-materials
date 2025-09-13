# 使用Langchain集成llm，为识别信息输出一段简短的语句
# 使用Langchain结合日志系统进行轻量时序分析

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import ImageDraw, ImageFont, Image
import init
import time
import os
import threading
from langchain.schema import HumanMessage, SystemMessage

# 线程通信信号量
need_regenerate = threading.Event()
need_regenerate.clear()

# ---------------------- 新增：异常分析相关全局变量 ----------------------
behavior_history = []  # 行为历史记录：(时间戳, 目标列表)
HISTORY_MAX_LENGTH = 60  # 历史记录最大长度 0.5s检测一次，60就是存半分钟，方便时序模型进行分析
is_abnormal = False  # 异常状态标记
abnormal_desc = ""  # 异常结果描述
last_abnormal_analysis_time = 0  # 上次异常分析时间戳
abnormal_analysis_interval = 31  # 时序分析间隔（秒） moonshot接口限频 3 RPM/min 所以只能默认半分钟一次（在不进行restart的情况下）

# ---------------------- 初始化LLM与YOLO模型 ----------------------
llm = init.chat  # 假设init模块已正确初始化LLM
model = YOLO(r'../data/yolov5n.pt')  # 加载YOLO模型


# ---------------------- 中文字体加载与绘制工具 ----------------------
def draw_chinese_with_pil(frame, text, org, color=(0, 255, 0)):
    try:
        h, w = frame.shape[:2]
        x, y = org
        text_width = len(text) * 16  # 估算中文字符宽度
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


# ---------------------- LLM介绍文字生成函数 ----------------------
def generate_detection_intro(detected_objs, llm):
    try:
        if not detected_objs:
            return "当前画面未检测到人、猫或狗，持续监控中..."

        # 整理检测结果
        obj_details = []
        for idx, obj in enumerate(detected_objs, 1):
            x1, y1, x2, y2, cls, conf = obj
            position = f"{'左侧' if x1 < 220 else '中间' if x1 < 420 else '右侧'}"
            area = (x2 - x1) * (y2 - y1)
            size = f"{'较小' if area < 10000 else '中等' if area < 30000 else '较大'}"
            obj_details.append(f"{idx}. {cls}（可信度{conf:.2f}）：位于画面{position}，体型{size}")

        # 构建LLM提示词
        messages = [
            SystemMessage(content="""你是智能监控助手，根据检测结果生成简洁介绍，要求：
1. 口语化，无专业术语；
2. 包含目标类型、数量、大致位置和可信度；
3. 不超过3句话，适合监控画面显示；
4. 多目标按可信度从高到低排序。"""),
            HumanMessage(content=f"检测结果：{'; '.join(obj_details)}，生成介绍")
        ]

        response = llm(messages)
        print(f"[{time.strftime('%H:%M:%S')}] " + "llm识别结果：" + response.content)
        return response.content

    except Exception as e:
        error_msg = f"介绍生成失败：{str(e)[:20]}..."
        print(f"LLM调用错误：{e}")
        return error_msg


# ---------------------- 新增：异常行为推理函数 ----------------------
def analyze_abnormal_behavior(behavior_history, llm):
    try:
        # 过滤无效历史
        valid_history = [item for item in behavior_history if len(item[1]) > 0]
        if len(valid_history) < 3:
            return False, "历史数据不足，暂不判断异常"

        # 格式化历史数据
        history_str = ""
        for idx, (ts, objs) in enumerate(reversed(valid_history[-5:])):
            time_str = time.strftime("%H:%M:%S", time.localtime(ts))
            obj_str = "; ".join([f"{obj['类别']}(位置{obj['中心点']})" for obj in objs])
            history_str += f"{idx + 1}. 时间{time_str}：{obj_str}\n"

        # 构建LangChain提示词
        messages = [
            SystemMessage(content="""你是监控系统的异常行为分析师，需基于历史和当前数据判断是否异常，规则如下：
1. 异常场景包括但不限于：
   - 同一目标在同一区域停留超过3秒（位置变化小于画面10%）；
   - 目标突然快速移动（连续2次检测中位置变化超过画面50%）；
   - 目标突然消失又在1秒内重新出现；
   - 多目标突然聚集（2人+1猫在同一区域重叠）。
2. 输出要求：
   - 先明确“正常”或“异常”；
   - 若异常，用1句话说明原因（口语化）；
   - 若正常，仅返回“正常”二字。"""),
            HumanMessage(content=f"历史检测数据（从新到旧）：\n{history_str}\n请判断当前行为是否异常，并按要求输出。")
        ]

        response = llm(messages)
        res_content = response.content.strip()

        # 解析LLM输出
        if res_content.startswith("异常"):
            # 记录异常日志
            log_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            with open("abnormal_log.txt", "a", encoding="utf-8") as f:
                f.write(f"[{log_time}] {res_content}\n")
            return True, res_content
        else:
            return False, "正常"

    except Exception as e:
        error_msg = f"异常分析失败：{str(e)[:20]}..."
        print(f"LLM异常分析错误：{e}")
        return False, error_msg


# ---------------------- 子线程：监听终端输入 ----------------------
def listen_terminal_input():
    print("\n=== 终端指令说明 ===")
    print("1. 输入 'restart' 并回车：重新生成介绍文字")
    print("2. 输入 'q' 并回车：退出程序")
    print("===================\n")

    while True:
        user_input = input().strip().lower()

        if user_input == "restart":
            print(f"[{time.strftime('%H:%M:%S')}] 收到restart指令，将重新生成介绍")
            need_regenerate.set()
        elif user_input == "q":
            print(f"[{time.strftime('%H:%M:%S')}] 收到退出指令，程序即将关闭")
            cv2.destroyAllWindows()
            os._exit(0)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] 未知指令，请输入'restart'或'q'")


# ---------------------- 主线程：摄像头定时检测与画面渲染 ----------------------
def camera_timed_detect_with_llm():
    global is_abnormal, abnormal_desc, last_abnormal_analysis_time

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头！请检查连接/权限/是否被占用")
        return

    # 基础配置
    target_classes = {0: '人', 15: '猫', 16: '狗'}
    confidence_threshold = 0.5
    detection_interval = 0.5
    last_detection_time = 0
    detected_objs = []
    current_intro = "初始化中，等待首次检测..."

    # 启动子线程
    input_thread = threading.Thread(target=listen_terminal_input, daemon=True)
    input_thread.start()

    # 主线程循环
    while True:
        ret, frame = cap.read()
        if not ret:
            print("警告：无法读取摄像头帧，重试中...")
            time.sleep(1)
            continue

        current_time = time.time()

        # 定时执行YOLO检测
        if current_time - last_detection_time >= detection_interval:
            results = model(
                frame,
                classes=[0, 15, 16],
                conf=confidence_threshold,
                verbose=False
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
            if current_intro == "初始化中，等待首次检测...":
                current_intro = generate_detection_intro(detected_objs, llm)

            # 记录行为历史
            formatted_objs = []
            for obj in detected_objs:
                x1, y1, x2, y2, cls, conf = obj
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                formatted_objs.append({
                    "类别": cls,
                    "可信度": conf,
                    "中心点": (center_x, center_y),
                    "时间": time.strftime("%H:%M:%S", time.localtime(current_time))
                })
            behavior_history.append((current_time, formatted_objs))
            if len(behavior_history) > HISTORY_MAX_LENGTH:
                behavior_history.pop(0)

        # 定时执行异常分析
        if current_time - last_abnormal_analysis_time >= abnormal_analysis_interval:
            is_abnormal, abnormal_desc = analyze_abnormal_behavior(behavior_history, llm)
            print(f"[{time.strftime('%H:%M:%S')}] " + "时序分析结果：" + str(is_abnormal) + abnormal_desc)
            last_abnormal_analysis_time = current_time

        # 检测restart指令
        if need_regenerate.is_set():
            current_intro = generate_detection_intro(detected_objs, llm)
            need_regenerate.clear()

        # 绘制目标检测框和标签
        for obj in detected_objs:
            x1, y1, x2, y2, cls, conf = obj
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text_y = y1 - 10 if y1 > 40 else y2 + 30
            frame = draw_chinese_with_pil(frame, f"{cls}：{conf:.2f}", (x1, text_y), (0, 255, 0))

        # 绘制LLM介绍文字
        frame = draw_chinese_with_pil(
            frame,
            current_intro,
            org=((frame.shape[1] - len(current_intro) * 16) // 2, 10),
            color=(179, 31, 69)
        )

        # 绘制异常分析结果
        if is_abnormal:
            # 异常时：红色背景+白色文字
            text_width = len(abnormal_desc) * 18
            text_x = (frame.shape[1] - text_width) // 2
            text_y = 40
            cv2.rectangle(frame, (text_x - 10, text_y - 5),
                          (text_x + text_width + 10, text_y + 25), (0, 0, 255), -1)
            frame = draw_chinese_with_pil(
                frame,
                abnormal_desc,
                org=(text_x, text_y),
                color=(255, 255, 255)
            )
        else:
            # 正常时：右上角灰色提示
            frame = draw_chinese_with_pil(
                frame,
                "行为正常",
                org=(frame.shape[1] - 120, 40),
                color=(128, 128, 128)
            )

        # 绘制检测间隔提示
        interval_text = f"检测间隔：{detection_interval}秒 | 终端输入'restart'重新生成"
        frame = draw_chinese_with_pil(
            frame,
            interval_text,
            org=(10, frame.shape[0] - 35),
            color=(255, 0, 0)
        )

        # 显示画面
        cv2.imshow("YOLOv5n + LLM + SA", frame)

        # 处理退出事件
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
