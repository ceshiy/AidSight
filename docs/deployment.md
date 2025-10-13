# 部署指南

本文档介绍如何将训练好的模型部署到香橙派 AI Pro（昇腾 NPU）上。

## 📋 部署流程概览

```
MindSpore (.ckpt) → ONNX (.onnx) → ATC 工具 → OM (.om) → MindIE 推理
```

## 🔄 模型转换

### 1. MindSpore → ONNX

#### 转换脚本

创建 `deployment/convert_to_onnx.py`:

```python
"""
将 MindSpore 模型转换为 ONNX 格式
"""
import numpy as np
import mindspore as ms
from mindspore import Tensor, export
from mindyolo.models import create_model

def convert_to_onnx(
    checkpoint_path: str,
    output_path: str,
    input_shape: tuple = (1, 3, 640, 640)
):
    """
    转换 MindSpore 模型为 ONNX
    
    Args:
        checkpoint_path: MindSpore 检查点路径
        output_path: 输出 ONNX 文件路径
        input_shape: 输入形状 (batch, channels, height, width)
    """
    # 创建模型
    model = create_model('yolov7', num_classes=6)
    
    # 加载权重
    param_dict = ms.load_checkpoint(checkpoint_path)
    ms.load_param_into_net(model, param_dict)
    
    # 设置为推理模式
    model.set_train(False)
    
    # 创建输入张量
    input_tensor = Tensor(np.random.randn(*input_shape), ms.float32)
    
    # 导出 ONNX
    export(
        model,
        input_tensor,
        file_name=output_path,
        file_format='ONNX'
    )
    
    print(f"模型已转换为 ONNX: {output_path}.onnx")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='MindSpore checkpoint path')
    parser.add_argument('--output', required=True, help='Output ONNX path')
    parser.add_argument('--input-size', type=int, default=640, help='Input size')
    
    args = parser.parse_args()
    
    convert_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        input_shape=(1, 3, args.input_size, args.input_size)
    )
```

运行转换:
```bash
python deployment/convert_to_onnx.py \
    --checkpoint models/checkpoints/yolov7_best.ckpt \
    --output deployment/yolov7_obstacle \
    --input-size 640
```

### 2. ONNX → OM (昇腾)

#### 使用 ATC 工具转换

ATC (Ascend Tensor Compiler) 是华为昇腾的模型转换工具。

```bash
# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 转换命令
atc \
    --model=deployment/yolov7_obstacle.onnx \
    --framework=5 \
    --output=deployment/models/yolov7_obstacle \
    --input_format=NCHW \
    --input_shape="images:1,3,640,640" \
    --soc_version=Ascend310P3 \
    --log=error
```

**参数说明**:
- `--model`: 输入 ONNX 模型路径
- `--framework`: 5 表示 ONNX 格式
- `--output`: 输出 OM 模型路径（不含扩展名）
- `--input_format`: 输入数据格式（NCHW 或 NHWC）
- `--input_shape`: 输入张量形状
- `--soc_version`: 昇腾芯片型号（310P3 对应 AI Pro）
- `--log`: 日志级别

#### 转换脚本

创建 `deployment/convert_to_om.py`:

```python
"""
将 ONNX 模型转换为昇腾 OM 格式
"""
import os
import subprocess

def convert_to_om(
    onnx_path: str,
    output_path: str,
    input_shape: str = "images:1,3,640,640",
    soc_version: str = "Ascend310P3"
):
    """
    使用 ATC 工具转换 ONNX 到 OM
    
    Args:
        onnx_path: ONNX 模型路径
        output_path: 输出 OM 路径（不含扩展名）
        input_shape: 输入形状
        soc_version: 昇腾芯片版本
    """
    # 检查 ONNX 文件是否存在
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX 文件不存在: {onnx_path}")
    
    # 构建 ATC 命令
    cmd = [
        'atc',
        f'--model={onnx_path}',
        '--framework=5',
        f'--output={output_path}',
        '--input_format=NCHW',
        f'--input_shape={input_shape}',
        f'--soc_version={soc_version}',
        '--log=error'
    ]
    
    # 执行转换
    print(f"执行转换命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"转换成功: {output_path}.om")
    else:
        print(f"转换失败: {result.stderr}")
        raise RuntimeError("ATC 转换失败")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', required=True, help='ONNX model path')
    parser.add_argument('--output', required=True, help='Output OM path')
    parser.add_argument('--input-shape', default='images:1,3,640,640', help='Input shape')
    parser.add_argument('--soc-version', default='Ascend310P3', help='SOC version')
    
    args = parser.parse_args()
    
    convert_to_om(
        onnx_path=args.onnx,
        output_path=args.output,
        input_shape=args.input_shape,
        soc_version=args.soc_version
    )
```

运行转换:
```bash
python deployment/convert_to_om.py \
    --onnx deployment/yolov7_obstacle.onnx \
    --output deployment/models/yolov7_obstacle
```

## 🚀 MindIE 推理引擎

### 推理代码

创建 `deployment/inference_mindie.py`:

```python
"""
使用 MindIE 进行昇腾 NPU 推理
"""
import numpy as np
import cv2
from typing import List, Dict

# 注意：实际使用时需要导入 MindIE SDK
# from mindie import Model, Tensor

class AscendInference:
    """昇腾 NPU 推理器"""
    
    def __init__(self, model_path: str, device_id: int = 0):
        """
        初始化推理器
        
        Args:
            model_path: OM 模型路径
            device_id: NPU 设备 ID
        """
        self.model_path = model_path
        self.device_id = device_id
        self.model = None
        self.input_shape = (640, 640)
        
    def load_model(self):
        """加载 OM 模型"""
        # 实际实现需要使用 MindIE SDK
        # self.model = Model(self.model_path, self.device_id)
        print(f"模型已加载: {self.model_path}")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像
        
        Args:
            image: 输入图像 (H, W, C)
        
        Returns:
            预处理后的图像 (1, C, H, W)
        """
        # Resize
        resized = cv2.resize(image, self.input_shape)
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Transpose to NCHW format
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)
        
        return batched
    
    def inference(self, input_data: np.ndarray) -> List[np.ndarray]:
        """
        执行推理
        
        Args:
            input_data: 预处理后的输入数据
        
        Returns:
            模型输出列表
        """
        # 实际实现需要使用 MindIE SDK
        # outputs = self.model.infer([input_data])
        # return outputs
        
        # 这里返回模拟输出
        return []
    
    def postprocess(
        self,
        outputs: List[np.ndarray],
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.45
    ) -> List[Dict]:
        """
        后处理推理结果
        
        Args:
            outputs: 模型原始输出
            conf_threshold: 置信度阈值
            nms_threshold: NMS 阈值
        
        Returns:
            检测结果列表
        """
        # 解析 YOLO 输出
        # 实际实现需要根据具体的 YOLO 版本调整
        detections = []
        
        # TODO: 实现后处理逻辑
        # 1. 解码边界框
        # 2. 过滤低置信度
        # 3. NMS
        # 4. 坐标转换
        
        return detections
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        端到端检测
        
        Args:
            image: 输入图像
        
        Returns:
            检测结果
        """
        # 预处理
        input_data = self.preprocess(image)
        
        # 推理
        outputs = self.inference(input_data)
        
        # 后处理
        detections = self.postprocess(outputs)
        
        return detections
    
    def unload_model(self):
        """卸载模型"""
        self.model = None
        print("模型已卸载")

# 使用示例
if __name__ == '__main__':
    # 创建推理器
    inferencer = AscendInference('deployment/models/yolov7_obstacle.om')
    inferencer.load_model()
    
    # 读取测试图像
    image = cv2.imread('test_image.jpg')
    
    # 执行检测
    detections = inferencer.detect(image)
    
    # 打印结果
    for det in detections:
        print(f"类别: {det['class']}, 置信度: {det['confidence']:.2f}")
```

## 📊 性能测试

### 基准测试脚本

创建 `deployment/benchmark.py`:

```python
"""
推理性能测试
"""
import time
import numpy as np
from inference_mindie import AscendInference

def benchmark(
    model_path: str,
    input_shape: tuple = (640, 640),
    num_iterations: int = 100,
    warmup_iterations: int = 10
):
    """
    性能测试
    
    Args:
        model_path: 模型路径
        input_shape: 输入尺寸
        num_iterations: 测试迭代次数
        warmup_iterations: 预热迭代次数
    """
    # 创建推理器
    inferencer = AscendInference(model_path)
    inferencer.load_model()
    
    # 创建随机输入
    dummy_input = np.random.randint(0, 255, (input_shape[0], input_shape[1], 3), dtype=np.uint8)
    
    # 预热
    print(f"预热 {warmup_iterations} 次...")
    for _ in range(warmup_iterations):
        _ = inferencer.detect(dummy_input)
    
    # 测试
    print(f"测试 {num_iterations} 次...")
    start_time = time.time()
    
    for _ in range(num_iterations):
        _ = inferencer.detect(dummy_input)
    
    elapsed = time.time() - start_time
    
    # 计算指标
    avg_time = elapsed / num_iterations
    fps = num_iterations / elapsed
    
    print("\n" + "=" * 50)
    print("性能测试结果")
    print("=" * 50)
    print(f"总时间: {elapsed:.2f} 秒")
    print(f"平均推理时间: {avg_time*1000:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print("=" * 50)
    
    # 卸载模型
    inferencer.unload_model()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='OM model path')
    parser.add_argument('--input-size', type=int, default=640, help='Input size')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations')
    
    args = parser.parse_args()
    
    benchmark(
        model_path=args.model,
        input_shape=(args.input_size, args.input_size),
        num_iterations=args.iterations
    )
```

运行测试:
```bash
python deployment/benchmark.py \
    --model deployment/models/yolov7_obstacle.om \
    --input-size 640 \
    --iterations 100
```

## ⚡ 性能优化

### 1. 模型量化

```bash
# 使用 ATC 工具进行 INT8 量化
atc \
    --model=deployment/yolov7_obstacle.onnx \
    --framework=5 \
    --output=deployment/models/yolov7_obstacle_int8 \
    --input_format=NCHW \
    --input_shape="images:1,3,640,640" \
    --precision_mode=allow_fp32_to_int8 \
    --soc_version=Ascend310P3
```

### 2. 输入尺寸优化

尝试不同的输入尺寸:
- 320x320: 更快，精度稍低
- 416x416: 平衡
- 640x640: 更准确，稍慢
- 1280x1280: 最准确，最慢

### 3. Batch 推理

如果有多个图像需要处理，使用 batch 推理:

```bash
atc \
    --model=deployment/yolov7_obstacle.onnx \
    --framework=5 \
    --output=deployment/models/yolov7_obstacle_batch4 \
    --input_shape="images:4,3,640,640" \
    --soc_version=Ascend310P3
```

## 📦 部署集成

### 更新检测器

修改 `src/detector.py` 以使用部署的模型:

```python
from deployment.inference_mindie import AscendInference

class ObjectDetector:
    def __init__(self, obstacle_model: str, traffic_model: str):
        self.obstacle_inferencer = AscendInference(obstacle_model)
        self.traffic_inferencer = AscendInference(traffic_model)
    
    def load_models(self):
        self.obstacle_inferencer.load_model()
        self.traffic_inferencer.load_model()
    
    def detect_obstacles(self, frame):
        return self.obstacle_inferencer.detect(frame)
```

## 🧪 部署验证

### 验证清单

- [ ] 模型转换成功（.ckpt → .onnx → .om）
- [ ] 推理引擎正常工作
- [ ] 推理结果正确
- [ ] 性能满足要求（FPS >= 10）
- [ ] 内存占用合理（< 2GB）
- [ ] NPU 利用率正常

### 测试命令

```bash
# 测试推理
python deployment/inference_mindie.py \
    --model deployment/models/yolov7_obstacle.om \
    --image test_images/test.jpg

# 性能测试
python deployment/benchmark.py \
    --model deployment/models/yolov7_obstacle.om

# 集成测试
python tests/test_inference.py
```

## 🔗 相关资源

- [昇腾 CANN 文档](https://www.hiascend.com/document)
- [ATC 工具指南](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/inferapplicationdev/atctool/atctool_000001.html)
- [MindIE 推理引擎](https://www.hiascend.com/software/mindie)
- [模型优化技巧](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/63RC1alpha001/optimizationdevg/modeloptdevg/)

---

**下一步**: 参考[使用说明](usage.md)运行完整的 AidSight 系统。
