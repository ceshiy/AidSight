# demo/app_ui.py

import sys
import queue
import threading
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# ===================================================================
# 重要：从你现有的 test1.py 中导入必要的函数和类
# 假设 app_ui.py 与 test1.py 在同一目录下或 test1.py 在Python路径中
# ===================================================================
from test1 import (
    get_parser_infer,
    parse_args,
    set_default_infer,
    create_model,
    detect,
    segment,
    draw_result_on_frame,
    is_url,
    detection_to_prompt_input,
    call_openai_compatible_llm,
    is_scene_significant_change,
    logger
)
import mindspore as ms
import time

# 全局队列，用于在检测线程和LLM线程间通信
detection_queue = queue.Queue(maxsize=5)
# 全局变量，用于在LLM线程和UI线程间通信
llm_prompt_text = "正在初始化..."

# ===================================================================
# 后台工作线程 (封装了所有的AI和视频处理逻辑)
# ===================================================================
class WorkerThread(QThread):
    """
    在后台运行视频捕获、检测和LLM分析的线程
    """
    # 定义信号：一个用于更新视频帧，一个用于更新LLM文本
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_llm_text_signal = pyqtSignal(str)

    def __init__(self, args):
        super().__init__()
        self.args = args
        self._is_running = True

    def run(self):
        """线程主函数"""
        args = self.args
        set_default_infer(args)

        # 1. 初始化模型
        network = create_model(
            model_name=args.network.model_name, model_cfg=args.network,
            num_classes=args.data.nc, sync_bn=False, checkpoint_path=args.weight
        )
        network.set_train(False)
        ms.amp.auto_mixed_precision(network, amp_level=args.ms_amp_level)
        is_coco_dataset = "coco" in args.data.dataset_name

        # 2. 启动LLM分析子线程
        llm_thread = threading.Thread(
            target=self.llm_analysis_worker,
            args=(args.img_size, args.data.names, is_coco_dataset),
            daemon=True
        )
        llm_thread.start()

        # 3. 初始化视频源
        source = args.source
        cap = None
        if source.isdigit():
            cap = cv2.VideoCapture(int(source))
        elif is_url(source):
            cap = cv2.VideoCapture(source)
        elif source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            cap = cv2.VideoCapture(source)
        else:
            logger.error(f"UI模式仅支持视频源，不支持单张图片: {source}")
            return

        if not cap or not cap.isOpened():
            logger.error(f"无法打开视频源: {source}")
            return

        # 4. 主循环：读取帧 -> 推理 -> 发送信号
        last_detection_time = 0
        detection_interval = 0.2  # ~5 FPS for detection
        last_result_dict = {}

        while self._is_running:
            ret, frame = cap.read()
            if not ret:
                logger.warning("视频流结束或读取失败。")
                break

            current_time = time.time()
            run_detection = (current_time - last_detection_time) > detection_interval

            if run_detection:
                last_detection_time = current_time
                if args.task == "detect":
                    result_dict = detect(network, frame, args.conf_thres, args.iou_thres, args.conf_free, args.exec_nms,
                                         args.nms_time_limit, args.img_size, max(max(args.network.stride), 32),
                                         args.data.nc, is_coco_dataset)
                else: # segment
                    result_dict = segment(network, frame, args.conf_thres, args.iou_thres, args.conf_free,
                                          args.nms_time_limit, args.img_size, max(max(args.network.stride), 32),
                                          args.data.nc, is_coco_dataset)
                last_result_dict = result_dict

                try:
                    detection_queue.put_nowait(result_dict)
                except queue.Full:
                    pass

            # 无论是否检测，都用最新的结果绘制画面
            annotated_frame = frame
            if last_result_dict:
                annotated_frame = draw_result_on_frame(frame, last_result_dict, args.data.names, is_coco_dataset, is_seg=(args.task=="segment"))

            # 发射信号，将带框的图像传递给UI线程
            self.change_pixmap_signal.emit(annotated_frame)

            # 短暂休眠，避免CPU占用过高
            time.sleep(0.01)

        cap.release()

    def llm_analysis_worker(self, frame_width, class_names, is_coco_dataset):
        """LLM分析的子线程，与原版逻辑相同，但通过信号更新UI"""
        global llm_prompt_text
        last_result = None
        last_llm_call_time = 0
        llm_interval = 2.0 # 至少2秒调用一次LLM

        while self._is_running:
            try:
                result_dict = detection_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # 检查场景是否有显著变化
            if not is_scene_significant_change(result_dict, last_result, frame_width, class_names, is_coco_dataset):
                continue

            # 检查调用频率
            if time.time() - last_llm_call_time < llm_interval:
                continue

            prompt_input = detection_to_prompt_input(result_dict, frame_width, class_names, is_coco_dataset)
            llm_output = call_openai_compatible_llm(
                prompt=prompt_input,
                api_url=self.args.api_url,
                api_key=self.args.api_key,
                model=self.args.llm_model
            )

            if llm_output and llm_output != llm_prompt_text:
                llm_prompt_text = llm_output
                self.update_llm_text_signal.emit(llm_prompt_text) # 发射信号更新UI
                logger.info(f"[✅ LLM Output] {llm_prompt_text}")
                last_llm_call_time = time.time()

            last_result = result_dict.copy()

    def stop(self):
        """停止线程"""
        self._is_running = False
        self.wait()

# ===================================================================
# PyQt5 UI主窗口
# ===================================================================
class MainApp(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("MindYOLO 实时导盲演示")
        self.setGeometry(100, 100, 800, 700)

        # 1. 创建中心部件和布局
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 2. 创建用于显示视频的 QLabel
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_label, 1) # 占据更多空间

        # 3. 创建用于显示LLM提示的 QLabel
        self.llm_label = QLabel("等待LLM生成安全提示...", self)
        self.llm_label.setAlignment(Qt.AlignCenter)
        self.llm_label.setWordWrap(True)
        self.llm_label.setStyleSheet(
            "background-color: #2c3e50; color: #ecf0f1; font-size: 18px; padding: 15px; border-radius: 8px;"
        )
        self.llm_label.setMinimumHeight(80)
        layout.addWidget(self.llm_label, 0) # 占据较少空间

        # 4. 启动后台工作线程
        self.thread = WorkerThread(args=args)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_llm_text_signal.connect(self.update_llm_text)
        self.thread.start()

    def closeEvent(self, event):
        """关闭窗口时，停止后台线程"""
        self.thread.stop()
        event.accept()

    def update_image(self, cv_img):
        """更新视频帧"""
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)

    def update_llm_text(self, text):
        """更新LLM提示文本"""
        self.llm_label.setText(text)

    def convert_cv_qt(self, cv_img):
        """将OpenCV图像格式转换为PyQt图像格式"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # 缩放图像以适应标签大小，同时保持纵横比
        p = convert_to_Qt_format.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

if __name__ == "__main__":
    # 扩展参数解析器以包含LLM相关的配置
    parser = get_parser_infer()
    parser.add_argument("--api_url", type=str, default="https://api.aibh.site/v1", help="LLM API endpoint URL")
    parser.add_argument("--api_key", type=str, default="sk-Ns6AiUJLEeZWEENYzsks01blVVY2QO9l7DRg0Cn9vMP8ezOn", help="LLM API Key")
    parser.add_argument("--llm_model", type=str, default="gpt-4.1-nano-2025-04-14", help="LLM model name")
    args = parse_args(parser)

    app = QApplication(sys.argv)
    main_window = MainApp(args)
    main_window.show()
    sys.exit(app.exec_())



