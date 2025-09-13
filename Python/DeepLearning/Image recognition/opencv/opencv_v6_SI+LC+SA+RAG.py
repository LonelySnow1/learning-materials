# 使用延迟识别技术，降低程序内存消费[定时检测] Scheduled inspection(SI)
# 使用Langchain集成llm，为识别信息输出一段简短的语句 (LC)
# 使用Langchain结合日志系统进行轻量时序分析 Time Series Analysis(SA)
# 使用Embedding模型在高维数据库Qdrant中嵌入时序分析日志，并做出分析 检索增强生成（Retrieval-Augmented Generation, RAG）

import cv2
import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from ultralytics import YOLO
from PIL import ImageDraw, ImageFont, Image
import init
import time
import os
import threading
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.vectorstores import Qdrant
from langchain.embeddings.base import Embeddings
from typing import List, Optional

# 线程通信信号量
need_regenerate = threading.Event()
need_regenerate.clear()

# ---------------------- 异常分析相关全局变量 ----------------------
behavior_history = []  # 行为历史记录：(时间戳, 目标列表)
HISTORY_MAX_LENGTH = 60  # 历史记录最大长度 0.5s检测一次，存半分钟
is_abnormal = False  # 异常状态标记
abnormal_desc = ""  # 异常结果描述
last_abnormal_analysis_time = 0  # 上次异常分析时间戳
abnormal_analysis_interval = 31  # 时序分析间隔（秒），适配API限频

# ---------------------- 初始化LLM与YOLO模型 ----------------------
try:
    llm = init.chat  # 假设init模块已正确初始化LLM
    if not llm:
        raise ValueError("LLM初始化失败，请检查init模块")
except Exception as e:
    print(f"LLM初始化错误: {e}")
    exit(1)

try:
    model = YOLO(r'../data/yolov5n.pt')  # 加载YOLO模型
except Exception as e:
    print(f"YOLO模型加载错误: {e}")
    exit(1)

# 1. 初始化Embedding模型（将文本转为向量）
try:
    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        cache_folder="../data/all-MiniLM-L6-v2"  # 本地缓存路径
    )
    # 测试嵌入模型是否正常工作
    test_embedding = embedding_model.encode(["test"], convert_to_tensor=False)
    if test_embedding is None or len(test_embedding) == 0:
        raise ValueError("嵌入模型无法生成有效向量")
except Exception as e:
    print(f"Embedding模型初始化错误: {e}")
    exit(1)


# ---------------------- 自定义Embeddings类（适配LangChain规范） ----------------------
class CustomEmbeddings(Embeddings):
    """符合LangChain规范的自定义Embeddings类，包装SentenceTransformer模型"""

    def __init__(self, model):
        self.model = model  # SentenceTransformer模型实例
        # 预计算嵌入维度（用于Qdrant集合初始化）
        self.dimension = len(self.embed_query("test"))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """生成文档向量（批量处理）"""
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            print(f"文档嵌入失败: {e}")
            return []

    def embed_query(self, text: str) -> List[float]:
        """生成查询向量（单条处理）"""
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            print(f"查询嵌入失败: {e}")
            return []


# 2. 初始化Qdrant客户端与向量存储（修复核心）
QDRANT_COLLECTION = "abnormal_logs"  # 向量集合名称
try:
    # 确保数据目录存在
    qdrant_data_path = "../data/qdrant_db"
    os.makedirs(qdrant_data_path, exist_ok=True)

    # 初始化Qdrant客户端
    qdrant_client = QdrantClient(path=qdrant_data_path)

    # 创建符合规范的Embeddings实例
    custom_embeddings = CustomEmbeddings(embedding_model)

    # 检查集合是否存在，不存在则创建（指定正确向量维度）
    if not qdrant_client.collection_exists(collection_name=QDRANT_COLLECTION):
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config={
                "size": custom_embeddings.dimension,  # 匹配模型输出维度
                "distance": "Cosine"  # 余弦相似度（适合文本向量）
            }
        )
        print(f"创建Qdrant集合: {QDRANT_COLLECTION}，向量维度: {custom_embeddings.dimension}")

    # 初始化Qdrant向量存储（使用规范的embeddings参数）
    qdrant_vectorstore = Qdrant(
        client=qdrant_client,
        collection_name=QDRANT_COLLECTION,
        embeddings=custom_embeddings  # 传入规范Embeddings实例，消除警告
    )
except Exception as e:
    print(f"Qdrant初始化错误: {e}")
    exit(1)


# ---------------------- 中文字体加载与绘制工具 ----------------------
def draw_chinese_with_pil(frame, text, org, color=(0, 255, 0)):
    """在OpenCV帧上绘制中文文本"""
    try:
        h, w = frame.shape[:2]
        x, y = org
        text_width = len(text) * 16  # 估算中文字符宽度
        # 确保文本在画面内
        x = max(0, min(x, w - text_width))
        y = max(30, min(y, h - 30))

        # BGR转RGB（PIL格式）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_img)

        # 尝试加载多种中文字体，提高兼容性
        font_paths = [
            'C:/Windows/Fonts/msyh.ttc',  # 微软雅黑
            'C:/Windows/Fonts/simhei.ttf',  # 黑体
            'C:/Windows/Fonts/simsun.ttc'  # 宋体
        ]
        font = None
        for path in font_paths:
            if os.path.exists(path):
                font = ImageFont.truetype(path, size=18)
                break

        if not font:
            # fallback到默认字体
            font = ImageFont.load_default()
            print("警告: 未找到中文字体，使用默认字体可能无法正常显示中文")

        draw.text((x, y), text, font=font, fill=color)
        # 转回BGR（OpenCV格式）
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"文本绘制错误: {e}")
        return frame


# ---------------------- LLM介绍文字生成函数 ----------------------
def generate_detection_intro(detected_objs, llm):
    """根据检测结果生成自然语言介绍"""
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
        result = response.content.strip()
        print(f"[{time.strftime('%H:%M:%S')}] LLM识别结果：{result}")
        return result

    except Exception as e:
        error_msg = f"介绍生成失败：{str(e)[:20]}..."
        print(f"LLM调用错误：{e}")
        return error_msg


# ---------------------- 异常行为推理函数 ----------------------
def analyze_abnormal_behavior(behavior_history, llm):
    """基于历史行为判断是否异常，并记录到向量数据库"""
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

        # 验证LLM输出格式
        if not res_content.startswith(("正常", "异常")):
            print(f"LLM输出格式异常: {res_content}")
            return False, "分析结果格式异常"

        # 解析LLM输出
        if res_content.startswith("异常"):
            # 记录异常日志
            log_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            full_abnormal_log = f"[{log_time}] {res_content}"  # 完整日志文本

            # 写入文本日志
            try:
                log_dir = "logs"
                os.makedirs(log_dir, exist_ok=True)
                with open(f"{log_dir}/abnormal_log.txt", "a", encoding="utf-8") as f:
                    f.write(f"{full_abnormal_log}\n")
            except Exception as e:
                print(f"写入异常日志文件失败: {e}")

            # 写入向量数据库（使用规范Embeddings，无需显式传向量）
            try:
                metadata = {
                    "timestamp": log_time,
                    "abnormal_type": res_content.split("：")[0] if "：" in res_content else "异常"
                }
                qdrant_vectorstore.add_texts(
                    texts=[full_abnormal_log],
                    metadatas=[metadata]
                )
                print(f"[{log_time}] 异常日志已写入Qdrant向量库")
            except Exception as e:
                print(f"写入Qdrant失败: {e}")

            return True, res_content
        else:
            return False, "正常"

    except Exception as e:
        error_msg = f"异常分析失败：{str(e)[:20]}..."
        print(f"LLM异常分析错误：{e}")
        return False, error_msg


# ——————————————————————— 问答反馈（修复检索异常） ——————————————————————
def query_abnormal_logs(user_question, llm, vectorstore, embeddings):
    """用户提问→检索相关异常日志→LLM结合上下文生成回答"""
    try:
        # 1. 显式生成查询向量（双重保障）
        query_embedding = embeddings.embed_query(user_question)
        if not query_embedding:
            return "查询向量生成失败，请重试"

        # 2. 向量检索：使用生成的向量查询相关日志
        relevant_logs = vectorstore.similarity_search_by_vector(
            embedding=query_embedding,
            k=3  # 返回Top3相关日志
        )

        # 3. 格式化检索到的日志（作为LLM的上下文）
        context = ""
        if relevant_logs:
            context = "相关异常日志：\n"
            for idx, doc in enumerate(relevant_logs, 1):
                context += f"{idx}. {doc.page_content}\n"
        else:
            context = "未检索到相关异常日志"

        # 4. 构建LLM提示词
        messages = [
            SystemMessage(content="""你是监控系统的日志问答助手，规则如下：
1. 仅基于提供的“相关异常日志”回答用户问题；
2. 若未检索到日志，直接告知“无相关记录”；
3. 回答需简洁、准确，包含日志中的时间和异常原因。"""),
            HumanMessage(content=f"用户问题：{user_question}\n{context}\n请回答用户问题")
        ]

        # 5. 调用LLM生成回答
        response = llm(messages)
        return response.content.strip()

    except Exception as e:
        return f"问答失败：{str(e)[:30]}..."


# ---------------------- 子线程：监听终端输入（修复问答调用） ----------------------
def listen_terminal_input():
    print("\n=== 终端指令说明 ===")
    print("1. 输入 'restart' 并回车：重新生成介绍文字")
    print("2. 输入 'q' 并回车：退出程序")
    print("3. 输入 'query 问题' 并回车：查询异常日志（如'query 今天14点有哪些异常'）")
    print("===================\n")

    while True:
        try:
            user_input = input().strip().lower()

            if user_input == "restart":
                print(f"[{time.strftime('%H:%M:%S')}] 收到restart指令，将重新生成介绍")
                need_regenerate.set()
            elif user_input == "q":
                print(f"[{time.strftime('%H:%M:%S')}] 收到退出指令，程序即将关闭")
                cv2.destroyAllWindows()
                os._exit(0)
            elif user_input.startswith("query "):
                user_question = user_input[len("query "):].strip()
                if not user_question:
                    print("请补充查询问题，格式：query 你的问题（如'query 今天有哪些异常'）")
                    continue
                print(f"[{time.strftime('%H:%M:%S')}] 正在查询...")
                # 传入embeddings实例，确保查询向量生成正常
                answer = query_abnormal_logs(user_question, llm, qdrant_vectorstore, custom_embeddings)
                print(f"[{time.strftime('%H:%M:%S')}] 问答结果：{answer}\n")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] 未知指令，请输入'restart'或'q'")
        except Exception as e:
            print(f"终端输入处理错误: {e}")


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
    detection_interval = 0.5  # 检测间隔（秒）
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
            try:
                results = model(
                    frame,
                    classes=[0, 15, 16],  # 只检测人、猫、狗
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
                # 保持历史记录长度
                while len(behavior_history) > HISTORY_MAX_LENGTH:
                    behavior_history.pop(0)
            except Exception as e:
                print(f"目标检测处理错误: {e}")

        # 定时执行异常分析
        if current_time - last_abnormal_analysis_time >= abnormal_analysis_interval:
            try:
                is_abnormal, abnormal_desc = analyze_abnormal_behavior(behavior_history, llm)
                print(f"[{time.strftime('%H:%M:%S')}] 时序分析结果：{is_abnormal} {abnormal_desc}")
                last_abnormal_analysis_time = current_time
            except Exception as e:
                print(f"异常分析执行错误: {e}")

        # 检测restart指令
        if need_regenerate.is_set():
            current_intro = generate_detection_intro(detected_objs, llm)
            need_regenerate.clear()

        # 绘制目标检测框和标签
        try:
            for obj in detected_objs:
                x1, y1, x2, y2, cls, conf = obj
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text_y = y1 - 10 if y1 > 40 else y2 + 30
                frame = draw_chinese_with_pil(frame, f"{cls}：{conf:.2f}", (x1, text_y), (0, 255, 0))
        except Exception as e:
            print(f"绘制检测框错误: {e}")

        # 绘制LLM介绍文字
        try:
            frame = draw_chinese_with_pil(
                frame,
                current_intro,
                org=((frame.shape[1] - len(current_intro) * 16) // 2, 10),
                color=(179, 31, 69)
            )
        except Exception as e:
            print(f"绘制介绍文字错误: {e}")

        # 绘制异常分析结果
        try:
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
        except Exception as e:
            print(f"绘制异常信息错误: {e}")

        # 绘制检测间隔提示
        try:
            interval_text = f"检测间隔：{detection_interval}秒 | 终端输入'restart'重新生成"
            frame = draw_chinese_with_pil(
                frame,
                interval_text,
                org=(10, frame.shape[0] - 35),
                color=(255, 0, 0)
            )
        except Exception as e:
            print(f"绘制状态信息错误: {e}")

        # 显示画面
        cv2.imshow("YOLOv5n + LLM 智能监控（带异常分析）", frame)

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
        # 创建必要的目录
        os.makedirs("../data", exist_ok=True)
        os.makedirs("../data/all-MiniLM-L6-v2", exist_ok=True)
        camera_timed_detect_with_llm()
    except Exception as e:
        print(f"程序启动失败：{str(e)}")
        exit(1)
