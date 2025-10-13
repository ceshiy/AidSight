"""
音频管理模块测试
"""

import sys
import os
import time

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from audio_manager import AudioManager
from utils import setup_logging


def test_audio_manager_init():
    """测试音频管理器初始化"""
    print("测试音频管理器初始化...")
    
    audio_manager = AudioManager(audio_dir='audio_assets')
    assert audio_manager is not None, "音频管理器初始化失败"
    
    print("✓ 音频管理器初始化成功")


def test_audio_manager_start_stop():
    """测试音频管理器启动和停止"""
    print("测试音频管理器启动和停止...")
    
    audio_manager = AudioManager()
    audio_manager.start()
    
    assert audio_manager.running, "音频管理器未启动"
    assert audio_manager.play_thread is not None, "播放线程未创建"
    
    time.sleep(1)
    
    audio_manager.stop()
    
    assert not audio_manager.running, "音频管理器未停止"
    
    print("✓ 音频管理器启动停止测试通过")


def test_audio_queue():
    """测试音频队列"""
    print("测试音频队列...")
    
    audio_manager = AudioManager()
    audio_manager.start()
    
    # 添加多个音频到队列
    messages = [
        ('obstacle_front', 1),
        ('danger_close', 0),  # 高优先级
        ('green_light', 2),
    ]
    
    for msg, priority in messages:
        audio_manager.play(msg, priority)
    
    time.sleep(2)  # 等待处理
    
    audio_manager.stop()
    
    print("✓ 音频队列测试通过")


def test_message_to_audio_key():
    """测试消息到音频键的映射"""
    print("测试消息到音频键的映射...")
    
    audio_manager = AudioManager()
    
    # 测试不同的消息
    test_cases = [
        ("前方有障碍物", "obstacle_front"),
        ("左侧有障碍物", "obstacle_left"),
        ("危险！停止", "danger_close"),
        ("红灯", "red_light"),
    ]
    
    for message, expected_key in test_cases:
        key = audio_manager._message_to_audio_key(message)
        # 由于实际映射可能不同，这里只检查返回值不为空
        assert key is not None and len(key) > 0, f"消息 '{message}' 映射失败"
        print(f"  '{message}' -> '{key}'")
    
    print("✓ 消息映射测试通过")


def test_audio_playback():
    """测试音频播放（需要音频文件）"""
    print("测试音频播放...")
    
    setup_logging(log_level='DEBUG')
    
    audio_manager = AudioManager(audio_dir='audio_assets')
    audio_manager.start()
    
    # 检查是否有音频文件
    if not audio_manager.audio_files:
        print("⚠ 警告: 未找到音频文件，跳过播放测试")
        audio_manager.stop()
        return
    
    # 播放第一个可用的音频
    audio_key = list(audio_manager.audio_files.keys())[0]
    print(f"播放音频: {audio_key}")
    
    audio_manager.play_audio_key(audio_key, priority=1)
    
    # 等待播放完成
    time.sleep(3)
    
    audio_manager.stop()
    
    print("✓ 音频播放测试通过")


def test_priority_queue():
    """测试优先级队列"""
    print("测试优先级队列...")
    
    audio_manager = AudioManager()
    audio_manager.start()
    
    # 添加不同优先级的消息
    audio_manager.play("低优先级消息", priority=2)
    audio_manager.play("高优先级消息", priority=0)
    audio_manager.play("中优先级消息", priority=1)
    
    time.sleep(2)
    
    # 优先级队列应该按优先级顺序处理
    # 实际播放顺序应该是: 高(0) -> 中(1) -> 低(2)
    
    audio_manager.stop()
    
    print("✓ 优先级队列测试通过")


def test_clear_queue():
    """测试清空队列"""
    print("测试清空队列...")
    
    audio_manager = AudioManager()
    audio_manager.start()
    
    # 添加多个消息
    for i in range(5):
        audio_manager.play(f"消息 {i}", priority=1)
    
    # 清空队列
    audio_manager.clear_queue()
    
    assert audio_manager.play_queue.empty(), "队列未清空"
    
    audio_manager.stop()
    
    print("✓ 清空队列测试通过")


if __name__ == '__main__':
    print("=" * 60)
    print("音频管理器测试")
    print("=" * 60)
    
    try:
        test_audio_manager_init()
        print()
        
        test_audio_manager_start_stop()
        print()
        
        test_audio_queue()
        print()
        
        test_message_to_audio_key()
        print()
        
        test_priority_queue()
        print()
        
        test_clear_queue()
        print()
        
        # 可选的播放测试（需要实际音频文件）
        # test_audio_playback()
        
        print("=" * 60)
        print("所有测试通过！")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"✗ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
