"""
工具函数模块
提供配置加载、日志配置、性能监控等通用功能
"""

import os
import yaml
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> logging.Logger:
    """
    配置日志系统
    
    Args:
        log_level: 日志级别 (DEBUG/INFO/WARNING/ERROR/CRITICAL)
        log_file: 日志文件路径，如果为 None 则只输出到控制台
        
    Returns:
        配置好的 logger 对象
    """
    # 创建 logger
    logger = logging.getLogger('AidSight')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定了日志文件）
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def ensure_dir(directory: str) -> None:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        directory: 目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def get_project_root() -> Path:
    """
    获取项目根目录
    
    Returns:
        项目根目录路径
    """
    # 从当前文件位置向上查找项目根目录
    current_file = Path(__file__).resolve()
    return current_file.parent.parent


class PerformanceMonitor:
    """性能监控器，用于测量代码执行时间"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.logger = logging.getLogger('AidSight.Performance')
    
    def __enter__(self):
        """进入上下文管理器"""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器"""
        elapsed = time.time() - self.start_time
        self.logger.debug(f"{self.name} 执行时间: {elapsed:.4f} 秒")
        return False


class FPSCounter:
    """FPS 计数器，用于测量帧率"""
    
    def __init__(self, window_size: int = 30):
        """
        Args:
            window_size: 计算平均 FPS 的窗口大小
        """
        self.window_size = window_size
        self.timestamps = []
    
    def update(self) -> float:
        """
        更新计数器并返回当前 FPS
        
        Returns:
            当前 FPS
        """
        current_time = time.time()
        self.timestamps.append(current_time)
        
        # 保持窗口大小
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)
        
        # 计算 FPS
        if len(self.timestamps) < 2:
            return 0.0
        
        time_diff = self.timestamps[-1] - self.timestamps[0]
        if time_diff == 0:
            return 0.0
        
        fps = (len(self.timestamps) - 1) / time_diff
        return fps
    
    def reset(self):
        """重置计数器"""
        self.timestamps.clear()


def validate_model_path(model_path: str) -> bool:
    """
    验证模型文件是否存在
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        True 如果模型文件存在，否则 False
    """
    return os.path.exists(model_path) and os.path.isfile(model_path)


def format_detection_result(detections: list, class_names: dict) -> str:
    """
    格式化检测结果为可读字符串
    
    Args:
        detections: 检测结果列表
        class_names: 类别名称字典
        
    Returns:
        格式化后的字符串
    """
    if not detections:
        return "未检测到目标"
    
    result_lines = [f"检测到 {len(detections)} 个目标:"]
    for i, det in enumerate(detections, 1):
        class_id = det.get('class_id', -1)
        confidence = det.get('confidence', 0.0)
        class_name = class_names.get(class_id, f"未知类别{class_id}")
        result_lines.append(f"  {i}. {class_name} (置信度: {confidence:.2%})")
    
    return "\n".join(result_lines)
