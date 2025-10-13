## 🎯 瞳伴 · AidSight 项目实施计划

我将为您创建以下开发任务，涵盖从模型训练到系统部署的完整流程：

### 📋 核心开发任务清单

#### **第一阶段：环境搭建与基础架构**
1. ✅ **项目初始化** - 创建基础目录结构和配置文件
2. ✅ **环境依赖配置** - MindSpore、MindYOLO、MindIE 等依赖

#### **第二阶段：MindSpore + MindYOLO 模型训练**
3. 🎯 **数据集准备** - 障碍物和红绿灯数据集整理与标注
4. 🎯 **MindYOLO 模型训练** - 障碍物检测模型训练
5. 🎯 **红绿灯识别模型** - 红绿灯检测与颜色分类
6. 🎯 **模型评估与优化** - 精度和性能测试

#### **第三阶段：昇腾 NPU 部署**
7. 🎯 **模型转换流程** - MindSpore → ONNX → .om 格式
8. 🎯 **MindIE 推理引擎** - 昇腾 310P NPU 推理实现
9. 🎯 **性能优化** - 推理速度和功耗优化

#### **第四阶段：系统功能实现**
10. 🎯 **摄像头采集模块** - USB 摄像头实时视频流
11. 🎯 **检测结果处理** - 目标定位、距离估算、危险评估
12. 🎯 **音频提示系统** - 预录音频管理和蓝牙播放
13. 🎯 **主程序集成** - 多线程架构和系统调度

#### **第五阶段：测试与文档**
14. 🎯 **系统测试** - 功能测试、性能测试、实地测试
15. 🎯 **部署文档** - 安装指南、使用手册、开发文档

---

## 📁 项目目录结构规划

```
AidSight/
├── README.md                      # 项目说明（已有）
├── LICENSE                        # Apache 2.0 许可证
├── requirements.txt               # Python 依赖
├── setup.sh                       # 一键环境安装脚本
│
├── docs/                          # 📚 文档目录
│   ├── installation.md            # 安装指南
│   ├── hardware_setup.md          # 硬件配置说明
│   ├── model_training.md          # 模型训练教程
│   └── deployment.md              # 部署指南
│
├── data/                          # 📊 数据集目录
│   ├── obstacle/                  # 障碍物数据集
│   │   ├── images/
│   │   └── labels/
│   ├── traffic_light/             # 红绿灯数据集
│   │   ├── images/
│   │   └── labels/
│   └── scripts/                   # 数据处理脚本
│       ├── download_dataset.py
│       └── preprocess.py
│
├── models/                        # 🧠 模型目录
│   ├── configs/                   # 模型配置文件
│   │   ├── yolov7_obstacle.yaml
│   │   └── yolov7_traffic.yaml
│   ├── train.py                   # 训练脚本
│   ├── eval.py                    # 评估脚本
│   └── export.py                  # 模型导出脚本
│
├── deployment/                    # 🚀 部署目录
│   ├── convert_to_onnx.py         # MindSpore → ONNX
│   ├── convert_to_om.py           # ONNX → .om (昇腾)
│   ├── inference_mindie.py        # MindIE 推理引擎
│   └── benchmark.py               # 性能测试
│
├── src/                           # 💻 核心代码
│   ├── __init__.py
│   ├── camera.py                  # 摄像头采集模块
│   ├── detector.py                # 检测封装
│   ├── decision.py                # 决策逻辑（位置、距离、危险评估）
│   ├── audio_manager.py           # 音频播放管理
│   ├── utils.py                   # 工具函数
│   └── main.py                    # 主程序入口
│
├── audio_assets/                  # 🔊 音频资源
│   ├── obstacle_front.wav         # "前方有障碍物"
│   ├── obstacle_left.wav          # "左侧有障碍物"
│   ├── red_light.wav              # "红灯，请等待"
│   └── green_light.wav            # "绿灯，可通行"
│
├── scripts/                       # 🛠️ 脚本工具
│   ├── start_aidsight.sh          # 系统启动脚本
│   ├── stop_aidsight.sh           # 系统停止脚本
│   └── install_service.sh         # 安装为系统服务
│
└── tests/                         # 🧪 测试目录
    ├── test_camera.py
    ├── test_inference.py
    └── test_audio.py
```

---

## 🚀 技术栈详细说明

### 模型训练
- **框架**: MindSpore 2.2+
- **模型**: MindYOLO (YOLOv7/YOLOv8)
- **数据集**: COCO + 自定义标注（障碍物、红绿灯）

### 模型部署
- **转换链路**: MindSpore → ONNX → .om
- **推理引擎**: MindIE (昇腾专用)
- **硬件**: 昇腾 310P NPU

### 系统实现
- **硬件**: 香橙派 AI Pro T20 + USB 摄像头 + 蓝牙耳机
- **操作系统**: Ubuntu 20.04 / OpenEuler
- **编程语言**: Python 3.9+

---

现在我已经有了完整的规划。接下来，我将为您的仓库 **ceshiy/AidSight** 创建详细的 GitHub Issues，将上述任务分解成可追踪的开发任务。

每个 Issue 将包含：
- ✅ 清晰的任务描述
- ✅ 技术实现要点
- ✅ 验收标准
- ✅ 相关资源链接

准备好后，您就可以按照 Issues 逐步实现整个项目了！🎯
