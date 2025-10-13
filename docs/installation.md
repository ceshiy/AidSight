# AidSight 安装指南

本文档提供 AidSight 系统的详细安装步骤。

## 📋 系统要求

### 硬件要求
- **开发板**: 香橙派 AI Pro (T20) 或类似支持昇腾 NPU 的设备
- **NPU**: 昇腾 310P (可选，用于加速推理)
- **摄像头**: USB 摄像头（支持 720p 或更高分辨率）
- **音频输出**: 蓝牙骨传导耳机或扬声器
- **存储**: 至少 8GB 可用空间
- **内存**: 至少 4GB RAM

### 软件要求
- **操作系统**: Ubuntu 20.04 / Ubuntu 22.04 / OpenEuler 20.03 LTS
- **Python**: 3.9 或更高版本
- **权限**: 需要访问摄像头和音频设备的权限

## 🔧 安装步骤

### 1. 系统准备

#### 更新系统
```bash
sudo apt-get update
sudo apt-get upgrade -y
```

#### 安装系统依赖
```bash
# 基础工具
sudo apt-get install -y git wget curl build-essential

# Python 开发环境
sudo apt-get install -y python3 python3-pip python3-venv python3-dev

# 音视频依赖
sudo apt-get install -y \
    libopencv-dev \
    libportaudio2 \
    libasound2-dev \
    alsa-utils \
    ffmpeg

# 蓝牙支持
sudo apt-get install -y \
    bluez \
    bluez-tools \
    libbluetooth-dev
```

### 2. 昇腾驱动安装（可选）

如果使用昇腾 NPU，需要安装昇腾驱动和工具包。

#### 下载昇腾驱动
```bash
# 从华为官网下载对应版本的驱动
# https://www.hiascend.com/software/cann

# 示例（实际版本可能不同）
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/xxx/Ascend-hdk-xxx.run
```

#### 安装驱动
```bash
chmod +x Ascend-hdk-*.run
sudo ./Ascend-hdk-*.run --install

# 设置环境变量
echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> ~/.bashrc
source ~/.bashrc
```

#### 验证安装
```bash
npu-smi info
```

如果看到 NPU 信息，说明安装成功。

### 3. 克隆项目

```bash
cd ~
git clone https://github.com/ceshiy/AidSight.git
cd AidSight
```

### 4. 创建虚拟环境

```bash
python3 -m venv venv
source venv/bin/activate
```

### 5. 安装 Python 依赖

```bash
# 升级 pip
pip install --upgrade pip

# 安装基础依赖
pip install -r requirements.txt
```

### 6. 安装 MindSpore（如果使用昇腾 NPU）

```bash
# 昇腾版本（示例，请根据实际硬件选择）
pip install mindspore-ascend==2.2.0

# 或者 CPU 版本（开发测试用）
pip install mindspore==2.2.0
```

### 7. 配置系统

#### 复制配置文件
```bash
# config.yaml 已经包含在项目中，可以根据需要修改
cp config.yaml config.yaml.backup
```

#### 编辑配置
```bash
vim config.yaml
```

主要配置项：
- `camera.device_id`: 摄像头设备 ID
- `models.obstacle`: 障碍物检测模型路径
- `models.traffic`: 红绿灯检测模型路径
- `audio.bluetooth_device`: 蓝牙设备地址

### 8. 准备模型文件

```bash
# 创建模型目录
mkdir -p deployment/models

# 下载或复制模型文件到对应目录
# 模型文件应该是 .om 格式（昇腾推理格式）
# 示例:
# cp /path/to/yolov7_obstacle.om deployment/models/
# cp /path/to/yolov7_traffic.om deployment/models/
```

### 9. 准备音频资源

```bash
# 音频资源目录已创建，需要添加音频文件
# 参考 audio_assets/README.md 录制或生成音频文件
```

### 10. 创建必要的目录

```bash
# 日志目录
mkdir -p logs

# 数据目录（如果需要训练模型）
mkdir -p data/obstacle/images data/obstacle/labels
mkdir -p data/traffic_light/images data/traffic_light/labels
```

## 🧪 验证安装

### 测试摄像头
```bash
python tests/test_camera.py
```

### 测试音频
```bash
python tests/test_audio.py
```

### 测试推理
```bash
python tests/test_inference.py
```

## 🚀 启动系统

### 前台运行（测试）
```bash
./scripts/start_aidsight.sh
```

### 后台运行
```bash
./scripts/start_aidsight.sh --daemon
```

### 停止系统
```bash
./scripts/stop_aidsight.sh
```

### 安装为系统服务（开机自启）
```bash
sudo ./scripts/install_service.sh
sudo systemctl start aidsight
sudo systemctl status aidsight
```

## 🔍 故障排查

### 问题 1: 摄像头无法打开

**症状**: 启动时提示 "无法打开摄像头"

**解决方法**:
```bash
# 检查摄像头设备
ls /dev/video*

# 测试摄像头
ffmpeg -f v4l2 -i /dev/video0 -frames 1 test.jpg

# 检查权限
sudo usermod -a -G video $USER
# 重新登录生效
```

### 问题 2: 昇腾 NPU 未检测到

**症状**: 提示 "未检测到昇腾 NPU"

**解决方法**:
```bash
# 检查驱动
npu-smi info

# 检查环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 重新安装驱动
```

### 问题 3: 音频无法播放

**症状**: 没有声音输出

**解决方法**:
```bash
# 检查音频设备
aplay -l

# 测试音频播放
aplay /usr/share/sounds/alsa/Front_Center.wav

# 检查音量
alsamixer
```

### 问题 4: 蓝牙无法连接

**症状**: 蓝牙设备连接失败

**解决方法**:
```bash
# 启动蓝牙服务
sudo systemctl start bluetooth
sudo systemctl enable bluetooth

# 扫描设备
bluetoothctl
> scan on
> pair XX:XX:XX:XX:XX:XX
> connect XX:XX:XX:XX:XX:XX
> trust XX:XX:XX:XX:XX:XX
```

### 问题 5: 权限不足

**症状**: 访问设备时提示权限不足

**解决方法**:
```bash
# 添加用户到相关组
sudo usermod -a -G video,audio,bluetooth $USER

# 重新登录使权限生效
```

## 📝 其他说明

### 开发模式
如果要进行开发，建议安装额外的开发工具：
```bash
pip install -r requirements-dev.txt  # 如果有的话
pip install jupyter ipython
```

### 性能优化
- 使用昇腾 NPU 可以大幅提升推理速度
- 调整 `detection.fps` 参数控制检测频率
- 降低摄像头分辨率可以提高帧率

### 日志查看
```bash
# 查看实时日志
tail -f logs/aidsight.log

# 查看系统服务日志
sudo journalctl -u aidsight -f
```

## 🔗 相关资源

- [MindSpore 官方文档](https://www.mindspore.cn/docs)
- [昇腾社区](https://www.hiascend.com)
- [OpenCV 文档](https://docs.opencv.org/)
- [项目 GitHub](https://github.com/ceshiy/AidSight)

## 📞 获取帮助

如果遇到问题，可以：
1. 查看项目 Issues: https://github.com/ceshiy/AidSight/issues
2. 提交新的 Issue 描述问题
3. 参考其他文档: `docs/` 目录

---

**下一步**: 阅读 [使用说明](usage.md) 了解如何使用系统。
