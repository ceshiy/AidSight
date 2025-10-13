"""
音频管理模块
实现音频播放系统，包括预录音频资源管理、蓝牙音频输出、播放队列管理
"""

import os
import queue
import logging
import threading
from pathlib import Path
from typing import Dict, Optional


class AudioManager:
    """音频播放管理器"""
    
    def __init__(self, audio_dir: str = 'audio_assets'):
        """
        初始化音频管理器
        
        Args:
            audio_dir: 音频资源目录路径
        """
        self.audio_dir = audio_dir
        self.play_queue = queue.PriorityQueue()  # 优先级队列 (priority, audio_key)
        self.is_playing = False
        self.bluetooth_device = None
        self.audio_files = {}
        self.logger = logging.getLogger('AidSight.Audio')
        
        # 播放线程
        self.play_thread = None
        self.running = False
        
        # 音频映射（消息到音频文件的映射）
        self.message_to_audio = {
            # 障碍物提示
            'obstacle_front': 'obstacle_front.wav',
            'obstacle_left': 'obstacle_left.wav',
            'obstacle_right': 'obstacle_right.wav',
            'danger_close': 'danger_close.wav',
            
            # 红绿灯提示
            'red_light': 'red_light.wav',
            'yellow_light': 'yellow_light.wav',
            'green_light': 'green_light.wav',
            
            # 其他提示
            'system_start': 'system_start.wav',
            'system_stop': 'system_stop.wav',
        }
        
        # 加载音频文件
        self.load_audio_files()
    
    def load_audio_files(self):
        """
        加载所有预录音频
        扫描音频目录并记录可用的音频文件
        """
        if not os.path.exists(self.audio_dir):
            self.logger.warning(f"音频目录不存在: {self.audio_dir}")
            os.makedirs(self.audio_dir, exist_ok=True)
            return
        
        # 扫描音频文件
        for filename in os.listdir(self.audio_dir):
            if filename.endswith('.wav'):
                audio_path = os.path.join(self.audio_dir, filename)
                audio_key = filename[:-4]  # 去掉 .wav 后缀
                self.audio_files[audio_key] = audio_path
                self.logger.debug(f"加载音频文件: {audio_key} -> {audio_path}")
        
        self.logger.info(f"已加载 {len(self.audio_files)} 个音频文件")
    
    def start(self):
        """启动音频播放线程"""
        if self.running:
            self.logger.warning("音频管理器已经在运行")
            return
        
        self.running = True
        self.play_thread = threading.Thread(target=self._play_loop, daemon=True)
        self.play_thread.start()
        self.logger.info("音频播放线程已启动")
    
    def stop(self):
        """停止音频播放线程"""
        self.running = False
        if self.play_thread:
            # 放入一个空消息以唤醒线程
            try:
                self.play_queue.put((0, None), timeout=1)
            except queue.Full:
                pass
            self.play_thread.join(timeout=2)
        self.logger.info("音频播放线程已停止")
    
    def play(self, message: str, priority: int = 1):
        """
        播放音频（支持优先级）
        
        Args:
            message: 消息内容（可以是文本或音频键）
            priority: 优先级 (0=紧急, 1=警告, 2=提示)
                     数字越小优先级越高
        """
        if not self.running:
            self.logger.warning("音频管理器未启动，无法播放音频")
            return
        
        # 将消息添加到播放队列
        try:
            self.play_queue.put((priority, message), timeout=1)
            self.logger.debug(f"音频已加入队列: {message} (优先级: {priority})")
        except queue.Full:
            self.logger.warning(f"播放队列已满，无法添加音频: {message}")
    
    def play_audio_key(self, audio_key: str, priority: int = 1):
        """
        直接播放音频文件键
        
        Args:
            audio_key: 音频文件键（不含扩展名）
            priority: 优先级
        """
        self.play(audio_key, priority)
    
    def _play_loop(self):
        """播放线程循环"""
        self.logger.debug("播放线程开始运行")
        
        while self.running:
            try:
                # 从队列获取音频（阻塞等待，超时 1 秒）
                priority, message = self.play_queue.get(timeout=1)
                
                if message is None:
                    continue
                
                # 播放音频
                self._play_audio(message)
                
                # 标记任务完成
                self.play_queue.task_done()
                
            except queue.Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                self.logger.error(f"播放音频时出错: {e}")
        
        self.logger.debug("播放线程已退出")
    
    def _play_audio(self, message: str):
        """
        实际播放音频的内部方法
        
        Args:
            message: 消息内容或音频键
        """
        # 尝试查找对应的音频文件
        audio_key = self._message_to_audio_key(message)
        
        if audio_key not in self.audio_files:
            self.logger.warning(f"未找到音频文件: {audio_key}")
            # 如果没有预录音频，可以在这里使用 TTS（文本转语音）
            self._synthesize_speech(message)
            return
        
        audio_path = self.audio_files[audio_key]
        
        try:
            self.is_playing = True
            self.logger.info(f"播放音频: {audio_path}")
            
            # 这里使用 pygame 或其他音频库播放
            # 由于可能没有安装音频库，这里只做模拟
            self._play_with_system(audio_path)
            
            self.is_playing = False
        except Exception as e:
            self.logger.error(f"播放音频失败: {e}")
            self.is_playing = False
    
    def _message_to_audio_key(self, message: str) -> str:
        """
        将消息转换为音频键
        
        Args:
            message: 消息内容
        
        Returns:
            音频键
        """
        # 如果消息本身就是音频键，直接返回
        if message in self.audio_files:
            return message
        
        # 尝试从映射表查找
        for key, audio_file in self.message_to_audio.items():
            if key in message.lower() or message in audio_file:
                return audio_file[:-4]  # 去掉 .wav 后缀
        
        # 根据消息内容智能匹配
        if '危险' in message or '停止' in message:
            return 'danger_close'
        elif '前方' in message:
            return 'obstacle_front'
        elif '左侧' in message:
            return 'obstacle_left'
        elif '右侧' in message:
            return 'obstacle_right'
        elif '红灯' in message:
            return 'red_light'
        elif '黄灯' in message:
            return 'yellow_light'
        elif '绿灯' in message:
            return 'green_light'
        
        # 默认返回原消息作为键
        return message
    
    def _play_with_system(self, audio_path: str):
        """
        使用系统命令播放音频
        
        Args:
            audio_path: 音频文件路径
        """
        # 这里可以根据不同系统使用不同的播放命令
        # Linux: aplay, paplay, ffplay
        # macOS: afplay
        # Windows: 使用 winsound 或 pygame
        
        import platform
        import subprocess
        
        system = platform.system()
        
        try:
            if system == 'Linux':
                # 尝试使用 aplay
                subprocess.run(['aplay', '-q', audio_path], check=True, timeout=10)
            elif system == 'Darwin':  # macOS
                subprocess.run(['afplay', audio_path], check=True, timeout=10)
            else:
                self.logger.warning(f"不支持的系统: {system}，无法播放音频")
        except FileNotFoundError:
            self.logger.warning("未找到音频播放程序，尝试使用其他方法")
            # 可以尝试 pygame 或其他库
        except subprocess.TimeoutExpired:
            self.logger.warning("音频播放超时")
        except Exception as e:
            self.logger.error(f"播放音频时出错: {e}")
    
    def _synthesize_speech(self, text: str):
        """
        使用 TTS（文本转语音）合成语音
        
        Args:
            text: 要合成的文本
        """
        # 这里可以集成 pyttsx3、gTTS 或其他 TTS 引擎
        # 由于是可选功能，这里只做日志记录
        self.logger.info(f"TTS 合成: {text}")
        # TODO: 实现 TTS 功能
    
    def connect_bluetooth(self, device_address: str) -> bool:
        """
        连接蓝牙骨传导耳机
        
        Args:
            device_address: 蓝牙设备地址 (MAC 地址格式: 00:11:22:33:44:55)
        
        Returns:
            连接是否成功
        """
        try:
            self.logger.info(f"尝试连接蓝牙设备: {device_address}")
            
            # 这里需要使用 bluez (Linux) 或其他蓝牙库
            # 由于涉及系统级别的蓝牙连接，这里只做模拟
            
            # TODO: 实现真实的蓝牙连接逻辑
            # 可以使用 pybluez 或 subprocess 调用 bluetoothctl
            
            self.bluetooth_device = device_address
            self.logger.info(f"蓝牙设备已连接: {device_address}")
            return True
            
        except Exception as e:
            self.logger.error(f"连接蓝牙设备失败: {e}")
            return False
    
    def disconnect_bluetooth(self):
        """断开蓝牙连接"""
        if self.bluetooth_device:
            self.logger.info(f"断开蓝牙设备: {self.bluetooth_device}")
            self.bluetooth_device = None
    
    def set_volume(self, volume: int):
        """
        设置音量
        
        Args:
            volume: 音量值 (0-100)
        """
        if not 0 <= volume <= 100:
            self.logger.warning(f"无效的音量值: {volume}，应该在 0-100 之间")
            return
        
        self.logger.info(f"设置音量: {volume}")
        # TODO: 实现音量控制
    
    def clear_queue(self):
        """清空播放队列"""
        with self.play_queue.mutex:
            self.play_queue.queue.clear()
        self.logger.info("播放队列已清空")
