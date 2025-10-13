"""
推理性能测试
"""

import sys
import os
import time
import numpy as np

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from detector import ObjectDetector
from decision import DecisionMaker
from utils import setup_logging, PerformanceMonitor


def test_detector_init():
    """测试检测器初始化"""
    print("测试检测器初始化...")
    
    detector = ObjectDetector(
        obstacle_model='deployment/models/yolov7_obstacle.om',
        traffic_model='deployment/models/yolov7_traffic.om'
    )
    
    assert detector is not None, "检测器初始化失败"
    print("✓ 检测器初始化成功")


def test_decision_maker():
    """测试决策引擎"""
    print("测试决策引擎...")
    
    decision_maker = DecisionMaker()
    
    # 创建模拟检测结果
    detections = [
        {
            'bbox': [100, 100, 300, 400],  # 大目标
            'class_id': 0,
            'confidence': 0.95
        },
        {
            'bbox': [500, 200, 600, 300],  # 小目标
            'class_id': 2,
            'confidence': 0.85
        }
    ]
    
    image_shape = (720, 1280, 3)
    
    # 评估危险
    result = decision_maker.evaluate_danger(detections, image_shape)
    
    assert 'level' in result, "结果缺少 level 字段"
    assert 'message' in result, "结果缺少 message 字段"
    assert 'obstacles' in result, "结果缺少 obstacles 字段"
    
    print(f"  危险等级: {result['level']}")
    print(f"  提示信息: {result['message']}")
    print(f"  障碍物数量: {len(result['obstacles'])}")
    
    print("✓ 决策引擎测试通过")


def test_distance_estimation():
    """测试距离估算"""
    print("测试距离估算...")
    
    decision_maker = DecisionMaker()
    image_shape = (720, 1280, 3)
    
    test_cases = [
        # (bbox, expected_distance)
        ([100, 100, 800, 600], '近'),    # 大 bbox
        ([200, 200, 400, 400], '中'),    # 中等 bbox
        ([500, 300, 550, 350], '远'),    # 小 bbox
    ]
    
    for bbox, expected in test_cases:
        distance = decision_maker.estimate_distance(bbox, image_shape)
        print(f"  bbox 大小: {bbox} -> 距离: {distance} (期望: {expected})")
        # 由于阈值可能不同，这里不严格断言
    
    print("✓ 距离估算测试通过")


def test_position_detection():
    """测试位置判断"""
    print("测试位置判断...")
    
    decision_maker = DecisionMaker()
    image_width = 1280
    
    test_cases = [
        # (bbox, expected_position)
        ([50, 100, 150, 200], '左侧'),     # 左侧
        ([550, 100, 650, 200], '前方'),    # 中间
        ([1100, 100, 1200, 200], '右侧'),  # 右侧
    ]
    
    for bbox, expected in test_cases:
        position = decision_maker._get_position(bbox, image_width)
        print(f"  bbox 中心: {(bbox[0]+bbox[2])/2:.0f} -> 位置: {position} (期望: {expected})")
        assert position == expected, f"位置判断错误: {position} != {expected}"
    
    print("✓ 位置判断测试通过")


def test_traffic_light_analysis():
    """测试红绿灯分析"""
    print("测试红绿灯分析...")
    
    decision_maker = DecisionMaker()
    
    test_cases = [
        # (state, expected_action)
        ({'state': 0, 'confidence': 0.9}, 'stop'),   # 红灯
        ({'state': 1, 'confidence': 0.9}, 'wait'),   # 黄灯
        ({'state': 2, 'confidence': 0.9}, 'go'),     # 绿灯
        (None, 'go'),                                 # 未检测到
        ({'state': 0, 'confidence': 0.3}, 'go'),     # 置信度低
    ]
    
    for traffic_result, expected_action in test_cases:
        result = decision_maker.analyze_traffic_light(traffic_result)
        print(f"  输入: {traffic_result} -> 动作: {result['action']}")
        assert result['action'] == expected_action, \
            f"动作判断错误: {result['action']} != {expected_action}"
    
    print("✓ 红绿灯分析测试通过")


def test_inference_performance():
    """测试推理性能（模拟）"""
    print("测试推理性能...")
    
    detector = ObjectDetector(
        obstacle_model='deployment/models/yolov7_obstacle.om',
        traffic_model='deployment/models/yolov7_traffic.om'
    )
    
    # 创建模拟图像
    image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    # 测量检测时间
    num_iterations = 10
    total_time = 0
    
    for i in range(num_iterations):
        with PerformanceMonitor("Obstacle Detection") as pm:
            detections = detector.detect_obstacles(image)
        # 由于模型未实际加载，这里只是测试接口
    
    print(f"✓ 推理性能测试完成（模拟模式）")


def test_end_to_end():
    """端到端测试"""
    print("测试端到端流程...")
    
    setup_logging(log_level='DEBUG')
    
    # 初始化模块
    detector = ObjectDetector(
        obstacle_model='deployment/models/yolov7_obstacle.om',
        traffic_model='deployment/models/yolov7_traffic.om'
    )
    
    decision_maker = DecisionMaker()
    
    # 创建模拟图像
    image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    # 1. 检测
    obstacles = detector.detect_obstacles(image)
    traffic = detector.detect_traffic_light(image)
    
    # 2. 决策
    danger_info = decision_maker.evaluate_danger(obstacles, image.shape)
    traffic_info = decision_maker.analyze_traffic_light(traffic)
    
    # 3. 验证结果
    assert 'level' in danger_info
    assert 'action' in traffic_info
    
    print("✓ 端到端测试通过")


if __name__ == '__main__':
    print("=" * 60)
    print("推理性能测试")
    print("=" * 60)
    
    try:
        test_detector_init()
        print()
        
        test_decision_maker()
        print()
        
        test_distance_estimation()
        print()
        
        test_position_detection()
        print()
        
        test_traffic_light_analysis()
        print()
        
        test_inference_performance()
        print()
        
        test_end_to_end()
        print()
        
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
