"""
摄像头采集模块
负责从 USB 摄像头获取实时视频流
"""

import cv2
import logging
import threading
import queue
from typing import Optional, Tuple
import numpy as np


class CameraCapture:
    """摄像头捕获类，支持多线程读取"""
    
    def __init__(
        self, 
        camera_id: int = 0, 
        width: int = 1280, 
        height: int = 720, 
        fps: int = 30,
        buffer_size: int = 2
    ):
        """
        初始化摄像头
        
        Args:
            camera_id: 摄像头设备 ID
            width: 视频宽度
            height: 视频高度
            fps: 帧率
            buffer_size: 缓冲队列大小
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size
        
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.running = False
        self.capture_thread = None
        self.logger = logging.getLogger('AidSight.Camera')
    
    def open(self) -> bool:
        """
        打开摄像头
        
        Returns:
            是否成功打开
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                self.logger.error(f"无法打开摄像头: {self.camera_id}")
                return False
            
            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # 获取实际参数
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            self.logger.info(f"摄像头已打开: {actual_width}x{actual_height}@{actual_fps}fps")
            
            return True
            
        except Exception as e:
            self.logger.error(f"打开摄像头时出错: {e}")
            return False
    
    def start(self):
        """启动摄像头捕获线程"""
        if self.running:
            self.logger.warning("摄像头已经在运行")
            return
        
        if self.cap is None or not self.cap.isOpened():
            if not self.open():
                raise RuntimeError("无法启动摄像头")
        
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        self.logger.info("摄像头捕获线程已启动")
    
    def stop(self):
        """停止摄像头捕获"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # 清空队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        self.logger.info("摄像头已停止")
    
    def _capture_loop(self):
        """摄像头捕获循环（在独立线程中运行）"""
        self.logger.debug("摄像头捕获线程开始运行")
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    self.logger.warning("无法读取摄像头帧")
                    continue
                
                # 将帧放入队列（如果队列满了，丢弃旧帧）
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.frame_queue.put(frame)
                
            except Exception as e:
                self.logger.error(f"捕获帧时出错: {e}")
        
        self.logger.debug("摄像头捕获线程已退出")
    
    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        读取一帧
        
        Args:
            timeout: 超时时间（秒）
        
        Returns:
            图像帧，如果超时返回 None
        """
        try:
            frame = self.frame_queue.get(timeout=timeout)
            return frame
        except queue.Empty:
            return None
    
    def get_frame_size(self) -> Tuple[int, int]:
        """
        获取帧大小
        
        Returns:
            (width, height)
        """
        return (self.width, self.height)
    
    def is_opened(self) -> bool:
        """
        检查摄像头是否打开
        
        Returns:
            是否打开
        """
        return self.cap is not None and self.cap.isOpened()
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.stop()
        return False
