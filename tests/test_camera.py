"""
摄像头模块测试
"""

import sys
import os
import time
import cv2

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from camera import CameraCapture


def test_camera_open():
    """测试摄像头打开"""
    print("测试摄像头打开...")
    camera = CameraCapture(camera_id=0)
    
    assert camera.open(), "摄像头打开失败"
    assert camera.is_opened(), "摄像头未正确打开"
    
    camera.stop()
    print("✓ 摄像头打开测试通过")


def test_camera_capture():
    """测试摄像头捕获"""
    print("测试摄像头捕获...")
    camera = CameraCapture(camera_id=0)
    
    camera.start()
    time.sleep(1)  # 等待摄像头预热
    
    # 读取几帧
    frames_captured = 0
    for i in range(10):
        frame = camera.read(timeout=2.0)
        if frame is not None:
            frames_captured += 1
            assert len(frame.shape) == 3, "帧格式不正确"
            assert frame.shape[2] == 3, "帧应该是 RGB 格式"
    
    camera.stop()
    
    assert frames_captured > 0, "未能捕获任何帧"
    print(f"✓ 成功捕获 {frames_captured} 帧")


def test_camera_context_manager():
    """测试上下文管理器"""
    print("测试上下文管理器...")
    
    with CameraCapture(camera_id=0) as camera:
        time.sleep(1)
        frame = camera.read(timeout=2.0)
        assert frame is not None, "未能读取帧"
    
    print("✓ 上下文管理器测试通过")


def test_camera_display():
    """测试摄像头显示（手动测试）"""
    print("测试摄像头显示（按 'q' 退出）...")
    camera = CameraCapture(camera_id=0)
    
    camera.start()
    time.sleep(1)
    
    print("显示摄像头画面 5 秒...")
    start_time = time.time()
    
    while time.time() - start_time < 5:
        frame = camera.read(timeout=1.0)
        if frame is not None:
            # 在图像上添加文本
            cv2.putText(
                frame, 
                "Camera Test - Press 'q' to quit", 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # 显示图像（如果有显示器）
            try:
                cv2.imshow('Camera Test', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except:
                # 如果没有显示器，只打印消息
                print(f"帧大小: {frame.shape}")
                time.sleep(0.1)
    
    cv2.destroyAllWindows()
    camera.stop()
    print("✓ 摄像头显示测试完成")


if __name__ == '__main__':
    print("=" * 60)
    print("摄像头测试")
    print("=" * 60)
    
    try:
        test_camera_open()
        print()
        
        test_camera_capture()
        print()
        
        test_camera_context_manager()
        print()
        
        # 可选的显示测试
        # test_camera_display()
        
        print("=" * 60)
        print("所有测试通过！")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"✗ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ 测试出错: {e}")
        sys.exit(1)
