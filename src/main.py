"""
AidSight 主程序
实现完整的系统集成，包括多线程架构、资源管理、命令行参数解析
"""

import sys
import time
import signal
import argparse
import logging
from pathlib import Path

from camera import CameraCapture
from detector import ObjectDetector
from decision import DecisionMaker
from audio_manager import AudioManager
from utils import load_config, setup_logging, FPSCounter


class AidSightSystem:
    """AidSight 主系统"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        初始化系统
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = load_config(config_path)
        
        # 设置日志
        log_config = self.config.get('logging', {})
        self.logger = setup_logging(
            log_level=log_config.get('level', 'INFO'),
            log_file=log_config.get('file', 'logs/aidsight.log')
        )
        
        self.logger.info("=" * 60)
        self.logger.info("AidSight 系统启动")
        self.logger.info("=" * 60)
        
        # 运行标志
        self.running = False
        
        # 初始化各模块
        self._init_modules()
        
        # FPS 计数器
        self.fps_counter = FPSCounter()
        
        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _init_modules(self):
        """初始化各功能模块"""
        try:
            # 摄像头
            camera_config = self.config.get('camera', {})
            self.camera = CameraCapture(
                camera_id=camera_config.get('device_id', 0),
                width=camera_config.get('width', 1280),
                height=camera_config.get('height', 720),
                fps=camera_config.get('fps', 30)
            )
            self.logger.info("摄像头模块初始化完成")
            
            # 检测器
            models_config = self.config.get('models', {})
            detection_config = self.config.get('detection', {})
            self.detector = ObjectDetector(
                obstacle_model=models_config.get('obstacle', ''),
                traffic_model=models_config.get('traffic', ''),
                conf_threshold=detection_config.get('conf_threshold', 0.5),
                nms_threshold=detection_config.get('nms_threshold', 0.45)
            )
            self.logger.info("检测器模块初始化完成")
            
            # 决策引擎
            self.decision_maker = DecisionMaker()
            self.logger.info("决策引擎初始化完成")
            
            # 音频管理器
            audio_config = self.config.get('audio', {})
            self.audio_manager = AudioManager(audio_dir='audio_assets')
            
            # 连接蓝牙设备（如果配置了）
            bt_device = audio_config.get('bluetooth_device')
            if bt_device:
                self.audio_manager.connect_bluetooth(bt_device)
            
            # 设置音量
            volume = audio_config.get('volume', 80)
            self.audio_manager.set_volume(volume)
            
            self.logger.info("音频管理器初始化完成")
            
        except Exception as e:
            self.logger.error(f"模块初始化失败: {e}")
            raise
    
    def start(self):
        """启动系统"""
        try:
            self.logger.info("正在启动系统...")
            
            # 加载检测模型
            self.logger.info("加载检测模型...")
            if not self.detector.load_models():
                raise RuntimeError("模型加载失败")
            
            # 启动摄像头
            self.logger.info("启动摄像头...")
            self.camera.start()
            
            # 启动音频管理器
            self.logger.info("启动音频管理器...")
            self.audio_manager.start()
            
            # 播放启动提示音
            self.audio_manager.play_audio_key('system_start', priority=2)
            
            # 设置运行标志
            self.running = True
            
            self.logger.info("系统启动成功，进入检测循环")
            
            # 进入主循环
            self.detection_loop()
            
        except KeyboardInterrupt:
            self.logger.info("收到中断信号")
        except Exception as e:
            self.logger.error(f"系统启动失败: {e}")
            raise
        finally:
            self.stop()
    
    def detection_loop(self):
        """检测主循环"""
        detection_interval = 1.0 / self.config.get('detection', {}).get('fps', 10)
        last_detection_time = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                # 控制检测帧率
                if current_time - last_detection_time < detection_interval:
                    time.sleep(0.01)
                    continue
                
                last_detection_time = current_time
                
                # 1. 获取帧
                frame = self.camera.read(timeout=1.0)
                if frame is None:
                    self.logger.warning("未能获取摄像头帧")
                    continue
                
                # 更新 FPS
                fps = self.fps_counter.update()
                if int(current_time) % 10 == 0:  # 每 10 秒打印一次
                    self.logger.debug(f"当前 FPS: {fps:.2f}")
                
                # 2. 检测障碍物和红绿灯
                obstacles = self.detector.detect_obstacles(frame)
                traffic = self.detector.detect_traffic_light(frame)
                
                # 3. 决策
                danger_info = self.decision_maker.evaluate_danger(obstacles, frame.shape)
                traffic_info = self.decision_maker.analyze_traffic_light(traffic)
                
                # 4. 语音提示
                # 优先播放危险提示
                if danger_info['level'] == 'danger':
                    self.audio_manager.play(danger_info['message'], priority=0)
                elif danger_info['level'] == 'warning':
                    self.audio_manager.play(danger_info['message'], priority=1)
                
                # 红绿灯提示
                if traffic_info['action'] != 'go' and traffic_info['message']:
                    self.audio_manager.play(traffic_info['message'], priority=1)
                
            except Exception as e:
                self.logger.error(f"检测循环出错: {e}")
                time.sleep(0.1)
    
    def stop(self):
        """停止系统"""
        self.logger.info("正在停止系统...")
        
        # 设置停止标志
        self.running = False
        
        # 播放停止提示音
        if hasattr(self, 'audio_manager'):
            self.audio_manager.play_audio_key('system_stop', priority=0)
            time.sleep(1)  # 等待播放完成
        
        # 停止各模块
        if hasattr(self, 'camera'):
            self.camera.stop()
            self.logger.info("摄像头已停止")
        
        if hasattr(self, 'audio_manager'):
            self.audio_manager.stop()
            self.logger.info("音频管理器已停止")
        
        if hasattr(self, 'detector'):
            self.detector.unload_models()
            self.logger.info("检测模型已卸载")
        
        self.logger.info("=" * 60)
        self.logger.info("AidSight 系统已停止")
        self.logger.info("=" * 60)
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        self.logger.info(f"收到信号 {signum}")
        self.running = False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='AidSight - 视障辅助导航系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                          # 使用默认配置启动
  %(prog)s --config my_config.yaml  # 使用自定义配置
  %(prog)s --log-level DEBUG        # 设置日志级别
  %(prog)s --daemon                 # 后台运行（需要实现）
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='配置文件路径 (默认: config.yaml)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='日志级别（会覆盖配置文件中的设置）'
    )
    
    parser.add_argument(
        '--daemon',
        action='store_true',
        help='后台运行模式'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='AidSight 1.0.0'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 检查配置文件
    if not Path(args.config).exists():
        print(f"错误: 配置文件不存在: {args.config}")
        sys.exit(1)
    
    # 创建并启动系统
    try:
        system = AidSightSystem(config_path=args.config)
        
        # 如果指定了日志级别，更新配置
        if args.log_level:
            system.logger.setLevel(getattr(logging, args.log_level))
        
        # 启动系统
        system.start()
        
    except Exception as e:
        print(f"系统启动失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
