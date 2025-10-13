"""
决策逻辑模块
负责分析检测结果，评估危险等级，生成语音提示
"""

import logging
from typing import Dict, List, Tuple, Any, Optional


class DecisionMaker:
    """决策引擎，负责分析检测结果并生成语音提示"""
    
    def __init__(self, danger_threshold: float = 0.3, warning_threshold: float = 0.15):
        """
        初始化决策引擎
        
        Args:
            danger_threshold: 危险距离阈值（基于 bbox 面积占屏幕比例）
            warning_threshold: 警告距离阈值（基于 bbox 面积占屏幕比例）
        """
        self.danger_threshold = danger_threshold  # 30% 屏幕面积
        self.warning_threshold = warning_threshold  # 15% 屏幕面积
        self.logger = logging.getLogger('AidSight.Decision')
        
        # 目标类别定义
        self.obstacle_classes = {
            0: 'person',      # 行人
            1: 'bicycle',     # 自行车
            2: 'car',         # 汽车
            3: 'motorcycle',  # 摩托车
            5: 'bus',         # 公交车
            7: 'truck',       # 卡车
            # 可根据实际模型添加更多类别
        }
        
        # 红绿灯状态
        self.traffic_light_states = {
            0: 'red',     # 红灯
            1: 'yellow',  # 黄灯
            2: 'green',   # 绿灯
        }
    
    def evaluate_danger(self, detections: List[Dict], image_shape: Tuple[int, int, int]) -> Dict[str, Any]:
        """
        评估危险等级并生成提示
        
        Args:
            detections: 检测结果列表，每个检测包含 bbox, class_id, confidence
                       bbox 格式: [x1, y1, x2, y2]
            image_shape: 图像形状 (height, width, channels)
        
        Returns:
            字典包含:
                - level: 危险等级 ('danger'/'warning'/'safe')
                - message: 语音提示信息
                - obstacles: 障碍物详细信息列表
        """
        if not detections:
            return {
                'level': 'safe',
                'message': None,
                'obstacles': []
            }
        
        height, width = image_shape[:2]
        image_area = height * width
        
        # 分析每个检测到的目标
        obstacles = []
        max_danger_level = 'safe'
        most_dangerous_obstacle = None
        
        for det in detections:
            bbox = det['bbox']
            class_id = det['class_id']
            confidence = det['confidence']
            
            # 计算障碍物信息
            obstacle_info = self._analyze_obstacle(bbox, class_id, confidence, image_shape)
            obstacles.append(obstacle_info)
            
            # 更新最危险的障碍物
            if obstacle_info['danger_level'] == 'danger':
                max_danger_level = 'danger'
                if most_dangerous_obstacle is None or \
                   obstacle_info['area_ratio'] > most_dangerous_obstacle['area_ratio']:
                    most_dangerous_obstacle = obstacle_info
            elif obstacle_info['danger_level'] == 'warning' and max_danger_level != 'danger':
                max_danger_level = 'warning'
                if most_dangerous_obstacle is None or \
                   obstacle_info['area_ratio'] > most_dangerous_obstacle['area_ratio']:
                    most_dangerous_obstacle = obstacle_info
        
        # 生成语音提示
        message = self._generate_message(most_dangerous_obstacle) if most_dangerous_obstacle else None
        
        result = {
            'level': max_danger_level,
            'message': message,
            'obstacles': obstacles
        }
        
        self.logger.debug(f"危险评估结果: {max_danger_level}, 检测到 {len(obstacles)} 个障碍物")
        return result
    
    def _analyze_obstacle(
        self, 
        bbox: List[float], 
        class_id: int, 
        confidence: float,
        image_shape: Tuple[int, int, int]
    ) -> Dict[str, Any]:
        """
        分析单个障碍物
        
        Args:
            bbox: 边界框 [x1, y1, x2, y2]
            class_id: 类别 ID
            confidence: 置信度
            image_shape: 图像形状
        
        Returns:
            障碍物详细信息字典
        """
        height, width = image_shape[:2]
        x1, y1, x2, y2 = bbox
        
        # 计算 bbox 面积和占比
        bbox_area = (x2 - x1) * (y2 - y1)
        image_area = height * width
        area_ratio = bbox_area / image_area
        
        # 估算距离
        distance = self.estimate_distance(bbox, image_shape)
        
        # 判断位置
        position = self._get_position(bbox, width)
        
        # 评估危险等级
        if area_ratio >= self.danger_threshold:
            danger_level = 'danger'
        elif area_ratio >= self.warning_threshold:
            danger_level = 'warning'
        else:
            danger_level = 'safe'
        
        # 获取类别名称
        class_name = self.obstacle_classes.get(class_id, f'unknown_{class_id}')
        
        return {
            'class_id': class_id,
            'class_name': class_name,
            'confidence': confidence,
            'bbox': bbox,
            'area_ratio': area_ratio,
            'distance': distance,
            'position': position,
            'danger_level': danger_level
        }
    
    def estimate_distance(self, bbox: List[float], image_shape: Tuple[int, int, int]) -> str:
        """
        根据边界框估算距离（近/中/远）
        
        基于边界框面积占屏幕比例进行估算:
        - 大于 30%: 近距离
        - 15%-30%: 中距离
        - 小于 15%: 远距离
        
        Args:
            bbox: 边界框 [x1, y1, x2, y2]
            image_shape: 图像形状 (height, width, channels)
        
        Returns:
            距离描述 ('近'/'中'/'远')
        """
        height, width = image_shape[:2]
        x1, y1, x2, y2 = bbox
        
        bbox_area = (x2 - x1) * (y2 - y1)
        image_area = height * width
        area_ratio = bbox_area / image_area
        
        if area_ratio >= 0.3:
            return '近'
        elif area_ratio >= 0.15:
            return '中'
        else:
            return '远'
    
    def _get_position(self, bbox: List[float], image_width: int) -> str:
        """
        判断障碍物位置（左侧/中间/右侧）
        
        Args:
            bbox: 边界框 [x1, y1, x2, y2]
            image_width: 图像宽度
        
        Returns:
            位置描述 ('左侧'/'中间'/'右侧')
        """
        x1, _, x2, _ = bbox
        center_x = (x1 + x2) / 2
        
        # 将图像分为三个区域
        left_boundary = image_width * 0.33
        right_boundary = image_width * 0.67
        
        if center_x < left_boundary:
            return '左侧'
        elif center_x > right_boundary:
            return '右侧'
        else:
            return '前方'
    
    def _generate_message(self, obstacle: Dict[str, Any]) -> str:
        """
        生成语音提示信息
        
        Args:
            obstacle: 障碍物信息字典
        
        Returns:
            语音提示字符串
        """
        class_name_cn = self._translate_class_name(obstacle['class_name'])
        position = obstacle['position']
        distance = obstacle['distance']
        
        if obstacle['danger_level'] == 'danger':
            return f"危险！{position}有{class_name_cn}，距离很近，请停止"
        elif obstacle['danger_level'] == 'warning':
            return f"注意，{position}有{class_name_cn}，请小心"
        else:
            return None
    
    def _translate_class_name(self, class_name: str) -> str:
        """
        将英文类别名翻译为中文
        
        Args:
            class_name: 英文类别名
        
        Returns:
            中文类别名
        """
        translations = {
            'person': '行人',
            'bicycle': '自行车',
            'car': '汽车',
            'motorcycle': '摩托车',
            'bus': '公交车',
            'truck': '卡车',
        }
        return translations.get(class_name, class_name)
    
    def analyze_traffic_light(self, traffic_result: Optional[Dict]) -> Dict[str, Any]:
        """
        分析红绿灯状态并生成提示
        
        Args:
            traffic_result: 红绿灯检测结果，包含 state 字段
                          state: 0=红灯, 1=黄灯, 2=绿灯
        
        Returns:
            字典包含:
                - action: 动作建议 ('stop'/'wait'/'go')
                - message: 语音提示信息
                - state: 红绿灯状态
        """
        if not traffic_result:
            return {
                'action': 'go',
                'message': None,
                'state': None
            }
        
        state = traffic_result.get('state')
        confidence = traffic_result.get('confidence', 0.0)
        
        # 置信度过低则忽略
        if confidence < 0.5:
            return {
                'action': 'go',
                'message': None,
                'state': None
            }
        
        state_name = self.traffic_light_states.get(state, 'unknown')
        
        # 根据红绿灯状态生成提示
        if state == 0:  # 红灯
            action = 'stop'
            message = '红灯，请停止等待'
        elif state == 1:  # 黄灯
            action = 'wait'
            message = '黄灯，请减速等待'
        elif state == 2:  # 绿灯
            action = 'go'
            message = '绿灯，可以通行'
        else:
            action = 'go'
            message = None
        
        self.logger.debug(f"红绿灯状态: {state_name}, 动作: {action}")
        
        return {
            'action': action,
            'message': message,
            'state': state_name
        }
