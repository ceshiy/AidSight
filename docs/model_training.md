# 模型训练教程

本文档介绍如何训练 AidSight 所需的 YOLO 目标检测模型。

## 📋 准备工作

### 环境要求
- MindSpore 2.2+
- MindYOLO
- Python 3.9+
- GPU（推荐）或 NPU

### 安装 MindYOLO
```bash
# 克隆 MindYOLO 仓库
cd ~
git clone https://github.com/mindspore-lab/mindyolo.git
cd mindyolo

# 安装依赖
pip install -r requirements.txt

# 安装 MindYOLO
pip install -e .
```

## 📊 数据集准备

### 1. 障碍物检测数据集

#### 推荐数据集
- **COCO 数据集**: 包含行人、车辆等常见障碍物
- **Pascal VOC**: 经典目标检测数据集
- **自定义数据集**: 针对特定场景标注

#### COCO 数据集下载
```bash
# 创建数据目录
mkdir -p data/obstacle
cd data/obstacle

# 下载 COCO 2017
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# 解压
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip

# 目录结构
# data/obstacle/
# ├── train2017/
# ├── val2017/
# └── annotations/
#     ├── instances_train2017.json
#     └── instances_val2017.json
```

#### 转换为 YOLO 格式
```bash
# 使用转换脚本
python data/scripts/coco_to_yolo.py \
    --json data/obstacle/annotations/instances_train2017.json \
    --output data/obstacle/labels
```

### 2. 红绿灯检测数据集

#### 推荐数据集
- **LISA Traffic Light Dataset**
- **Bosch Small Traffic Lights Dataset**
- **自定义采集**: 在本地环境采集红绿灯图像

#### 自定义数据采集
```python
# 使用摄像头采集红绿灯图像
import cv2
import os

cap = cv2.VideoCapture(0)
save_dir = 'data/traffic_light/images'
os.makedirs(save_dir, exist_ok=True)

count = 0
print("按空格键保存图像，按 'q' 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow('Capture', frame)
    key = cv2.waitKey(1)
    
    if key == ord(' '):  # 空格键保存
        filename = os.path.join(save_dir, f'traffic_{count:04d}.jpg')
        cv2.imwrite(filename, frame)
        print(f"已保存: {filename}")
        count += 1
    elif key == ord('q'):  # q 键退出
        break

cap.release()
cv2.destroyAllWindows()
```

### 3. 数据标注

#### 使用 LabelImg
```bash
# 安装 LabelImg
pip install labelImg

# 启动标注工具
labelimg data/obstacle/images data/obstacle/labels
```

**标注流程**:
1. 打开图像
2. 按 'w' 创建边界框
3. 选择类别
4. 保存标注（YOLO 格式）
5. 下一张图像（按 'd'）

#### 类别定义

**障碍物类别** (`obstacle_classes.txt`):
```
person
bicycle
car
motorcycle
bus
truck
```

**红绿灯类别** (`traffic_classes.txt`):
```
red
yellow
green
```

### 4. 数据集划分

```python
# data/scripts/split_dataset.py
import os
import random
from shutil import copyfile

def split_dataset(images_dir, labels_dir, train_ratio=0.8):
    """将数据集划分为训练集和验证集"""
    
    # 获取所有图像文件
    images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    random.shuffle(images)
    
    # 划分
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    # 创建目录
    for subset in ['train', 'val']:
        os.makedirs(f'data/obstacle/{subset}/images', exist_ok=True)
        os.makedirs(f'data/obstacle/{subset}/labels', exist_ok=True)
    
    # 复制文件
    for img in train_images:
        # 复制图像和标注
        pass
    
    print(f"训练集: {len(train_images)}, 验证集: {len(val_images)}")

if __name__ == '__main__':
    split_dataset('data/obstacle/images', 'data/obstacle/labels')
```

## 🏋️ 模型训练

### 1. 配置文件

创建 `models/configs/yolov7_obstacle.yaml`:

```yaml
# YOLOv7 障碍物检测配置

model:
  type: YOLOv7
  backbone:
    type: ELANNet
  neck:
    type: ELANFPN
  head:
    type: YOLOv7Head
    num_classes: 6  # 障碍物类别数

data:
  train:
    dataset:
      type: COCODataset
      data_root: data/obstacle
      ann_file: train2017
      img_prefix: train2017
    transforms:
      - type: Resize
        size: [640, 640]
      - type: RandomFlip
        prob: 0.5
      - type: Normalize
        mean: [0, 0, 0]
        std: [255, 255, 255]
    
  val:
    dataset:
      type: COCODataset
      data_root: data/obstacle
      ann_file: val2017
      img_prefix: val2017

train:
  batch_size: 16
  epochs: 300
  optimizer:
    type: SGD
    lr: 0.01
    momentum: 0.9
    weight_decay: 0.0005
  lr_scheduler:
    type: CosineAnnealing
    T_max: 300
```

### 2. 训练脚本

创建 `models/train.py`:

```python
"""
模型训练脚本
"""
import mindspore as ms
from mindspore import nn, context
from mindyolo.models import create_model
from mindyolo.data import create_dataloader
from mindyolo.utils import load_config

def train():
    # 设置上下文
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    
    # 加载配置
    cfg = load_config('models/configs/yolov7_obstacle.yaml')
    
    # 创建模型
    model = create_model(
        model_name='yolov7',
        num_classes=cfg['model']['head']['num_classes']
    )
    
    # 创建数据加载器
    train_loader = create_dataloader(cfg['data']['train'])
    val_loader = create_dataloader(cfg['data']['val'])
    
    # 定义损失函数和优化器
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    optimizer = nn.SGD(
        model.trainable_params(),
        learning_rate=cfg['train']['optimizer']['lr']
    )
    
    # 训练循环
    epochs = cfg['train']['epochs']
    for epoch in range(epochs):
        # 训练一个 epoch
        model.set_train()
        for batch in train_loader:
            # 前向传播和反向传播
            pass
        
        # 验证
        if epoch % 10 == 0:
            model.set_train(False)
            # 验证逻辑
            pass
        
        # 保存检查点
        if epoch % 50 == 0:
            ms.save_checkpoint(model, f'models/checkpoints/yolov7_epoch_{epoch}.ckpt')

if __name__ == '__main__':
    train()
```

### 3. 启动训练

```bash
# 单卡训练
python models/train.py --config models/configs/yolov7_obstacle.yaml

# 多卡训练
mpirun -n 4 python models/train.py --config models/configs/yolov7_obstacle.yaml --distributed

# 使用昇腾 NPU 训练
python models/train.py --device_target Ascend
```

### 4. 监控训练过程

```python
# 使用 TensorBoard 监控
from mindspore.train.callback import TensorboardCallback

# 在训练脚本中添加回调
tensorboard_cb = TensorboardCallback(log_dir='./logs')
```

查看训练曲线:
```bash
tensorboard --logdir=./logs --port=6006
```

## 📈 模型评估

### 评估脚本

创建 `models/eval.py`:

```python
"""
模型评估脚本
"""
import mindspore as ms
from mindyolo.models import create_model
from mindyolo.data import create_dataloader
from mindyolo.utils import load_config, calculate_map

def evaluate():
    # 加载配置和模型
    cfg = load_config('models/configs/yolov7_obstacle.yaml')
    model = create_model('yolov7', num_classes=6)
    
    # 加载权重
    ms.load_checkpoint('models/checkpoints/best.ckpt', model)
    
    # 创建验证数据加载器
    val_loader = create_dataloader(cfg['data']['val'])
    
    # 评估
    model.set_train(False)
    results = []
    
    for batch in val_loader:
        outputs = model(batch['images'])
        results.append(outputs)
    
    # 计算 mAP
    map_score = calculate_map(results, val_loader.dataset)
    print(f"mAP@0.5: {map_score:.4f}")

if __name__ == '__main__':
    evaluate()
```

运行评估:
```bash
python models/eval.py --checkpoint models/checkpoints/best.ckpt
```

## 🔄 模型导出

### 导出为 ONNX

创建 `models/export.py`:

```python
"""
模型导出脚本
"""
import mindspore as ms
from mindyolo.models import create_model

def export_onnx():
    # 创建模型
    model = create_model('yolov7', num_classes=6)
    
    # 加载权重
    ms.load_checkpoint('models/checkpoints/best.ckpt', model)
    
    # 设置为推理模式
    model.set_train(False)
    
    # 导出 ONNX
    input_shape = [1, 3, 640, 640]
    input_tensor = ms.Tensor(np.random.randn(*input_shape), ms.float32)
    
    ms.export(
        model,
        input_tensor,
        file_name='models/yolov7_obstacle',
        file_format='ONNX'
    )
    
    print("模型已导出为 ONNX 格式")

if __name__ == '__main__':
    export_onnx()
```

运行导出:
```bash
python models/export.py
```

## 🎯 超参数调优

### 重要超参数

| 参数 | 默认值 | 调优建议 |
|------|--------|----------|
| 学习率 | 0.01 | 尝试 0.001-0.1 |
| batch_size | 16 | 根据 GPU 内存调整 |
| 输入尺寸 | 640x640 | 更大尺寸提高精度但降低速度 |
| 数据增强 | 中等 | 增加随机裁剪、旋转等 |
| Anchor 大小 | 预设 | 根据数据集聚类调整 |

### 学习率调度

```python
# 使用余弦退火
from mindspore.nn.learning_rate_schedule import CosineDecayLR

lr_schedule = CosineDecayLR(
    min_lr=0.0001,
    max_lr=0.01,
    decay_steps=300 * steps_per_epoch
)
```

## 📊 结果分析

### 训练指标

监控以下指标:
- **Loss**: 应持续下降
- **mAP@0.5**: 验证集上的平均精度
- **Precision**: 精确率
- **Recall**: 召回率
- **FPS**: 推理速度

### 常见问题

**过拟合**:
- 增加数据增强
- 使用 Dropout
- 减小模型容量
- 早停

**欠拟合**:
- 增加模型容量
- 训练更多 epoch
- 提高学习率
- 检查数据质量

**推理速度慢**:
- 减小输入尺寸
- 使用轻量化模型
- 模型剪枝
- 量化

## 🔗 相关资源

- [MindYOLO 官方文档](https://github.com/mindspore-lab/mindyolo)
- [YOLO 系列论文](https://arxiv.org/abs/2207.02696)
- [数据增强技术](https://github.com/albumentations-team/albumentations)
- [模型压缩技术](https://github.com/microsoft/nni)

---

**下一步**: 参考[部署指南](deployment.md)将训练好的模型部署到昇腾设备。
