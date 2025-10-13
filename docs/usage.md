# AidSight 使用说明

本文档介绍如何使用 AidSight 系统。

## 🚀 快速开始

### 启动系统

#### 前台运行（推荐用于测试）
```bash
cd ~/AidSight
./scripts/start_aidsight.sh
```

系统会显示实时日志，按 `Ctrl+C` 停止。

#### 后台运行
```bash
./scripts/start_aidsight.sh --daemon
```

#### 指定日志级别
```bash
# DEBUG - 详细调试信息
./scripts/start_aidsight.sh --log-level DEBUG

# INFO - 一般信息（默认）
./scripts/start_aidsight.sh --log-level INFO

# WARNING - 仅警告和错误
./scripts/start_aidsight.sh --log-level WARNING
```

### 停止系统

```bash
./scripts/stop_aidsight.sh
```

### 查看状态

```bash
# 如果作为系统服务运行
sudo systemctl status aidsight

# 查看日志
tail -f logs/aidsight.log
```

## ⚙️ 配置文件说明

配置文件位于项目根目录的 `config.yaml`。

### 摄像头配置

```yaml
camera:
  device_id: 0          # 摄像头设备 ID
  width: 1280           # 视频宽度
  height: 720           # 视频高度
  fps: 30               # 摄像头帧率
```

**device_id 说明**:
- `0` 表示第一个摄像头（通常是默认摄像头）
- 如果有多个摄像头，可以设置为 `1`, `2` 等
- 使用 `ls /dev/video*` 查看可用设备

### 模型配置

```yaml
models:
  obstacle: "deployment/models/yolov7_obstacle.om"
  traffic: "deployment/models/yolov7_traffic.om"
```

确保模型文件路径正确，且文件存在。

### 检测配置

```yaml
detection:
  conf_threshold: 0.5   # 置信度阈值（0-1）
  nms_threshold: 0.45   # NMS 阈值（0-1）
  fps: 10               # 检测帧率
```

**参数说明**:
- `conf_threshold`: 越高越严格，减少误检但可能漏检
- `nms_threshold`: 控制重叠目标的去除
- `fps`: 检测频率，降低可节省计算资源

### 决策配置

```yaml
decision:
  danger_threshold: 0.3    # 危险阈值（0-1）
  warning_threshold: 0.15  # 警告阈值（0-1）
```

**阈值说明**:
- 基于目标边界框占屏幕的面积比例
- `danger_threshold`: 超过此值触发紧急警告
- `warning_threshold`: 超过此值触发普通警告

### 音频配置

```yaml
audio:
  bluetooth_device: ""  # 蓝牙设备 MAC 地址
  volume: 80            # 音量（0-100）
```

**蓝牙配置**:
1. 留空使用默认音频输出
2. 填写 MAC 地址连接蓝牙设备
3. 使用 `bluetoothctl` 查看已配对设备的 MAC 地址

### 日志配置

```yaml
logging:
  level: "INFO"              # 日志级别
  file: "logs/aidsight.log"  # 日志文件
```

## 📊 理解系统输出

### 日志级别

- **DEBUG**: 详细的调试信息，包括每一帧的处理细节
- **INFO**: 系统状态信息，如启动、停止、检测结果
- **WARNING**: 警告信息，如未检测到设备、配置问题
- **ERROR**: 错误信息，如模块加载失败、硬件问题
- **CRITICAL**: 严重错误，导致系统无法运行

### 典型日志示例

```
2024-01-01 10:00:00 - AidSight - INFO - ============================================
2024-01-01 10:00:00 - AidSight - INFO - AidSight 系统启动
2024-01-01 10:00:00 - AidSight - INFO - ============================================
2024-01-01 10:00:01 - AidSight.Camera - INFO - 摄像头已打开: 1280x720@30fps
2024-01-01 10:00:02 - AidSight.Detector - INFO - 检测模型加载成功
2024-01-01 10:00:03 - AidSight.Audio - INFO - 已加载 9 个音频文件
2024-01-01 10:00:04 - AidSight - INFO - 系统启动成功，进入检测循环
2024-01-01 10:00:10 - AidSight.Decision - DEBUG - 危险评估结果: warning, 检测到 1 个障碍物
2024-01-01 10:00:10 - AidSight.Audio - INFO - 播放音频: obstacle_front.wav
```

## 🎯 使用场景

### 场景 1: 室内导航

**适用**: 在室内环境中避开障碍物

**配置建议**:
- 设置较低的危险阈值（0.25）
- 提高检测帧率（15 fps）
- 启用所有障碍物类别

### 场景 2: 室外过马路

**适用**: 检测红绿灯和来往车辆

**配置建议**:
- 启用红绿灯检测模型
- 设置适中的危险阈值（0.3）
- 关注车辆、自行车等类别

### 场景 3: 人群密集区域

**适用**: 在人流密集的地方行走

**配置建议**:
- 提高置信度阈值（0.6）以减少误检
- 降低检测帧率（8 fps）以减少延迟
- 主要关注行人类别

## 🔧 高级功能

### 自定义音频提示

1. 录制新的音频文件（参考 `audio_assets/README.md`）
2. 将文件放入 `audio_assets/` 目录
3. 修改 `src/audio_manager.py` 中的映射关系

### 调整检测类别

编辑 `src/decision.py`，修改 `obstacle_classes` 字典：

```python
self.obstacle_classes = {
    0: 'person',      # 保留
    2: 'car',         # 保留
    # 5: 'bus',       # 注释掉不需要的类别
}
```

### 性能监控

启用性能监控以查看系统性能指标：

```yaml
performance:
  enable_monitoring: true
  show_fps: true
```

在日志中会显示 FPS 和各模块的执行时间。

## 📱 系统服务管理

### 安装为系统服务

```bash
sudo ./scripts/install_service.sh
```

### 服务管理命令

```bash
# 启动服务
sudo systemctl start aidsight

# 停止服务
sudo systemctl stop aidsight

# 重启服务
sudo systemctl restart aidsight

# 查看状态
sudo systemctl status aidsight

# 启用开机自启
sudo systemctl enable aidsight

# 禁用开机自启
sudo systemctl disable aidsight

# 查看日志
sudo journalctl -u aidsight -f
```

## 🐛 常见问题

### Q: 系统启动后没有声音

**A**: 检查以下几点：
1. 音频文件是否存在（`audio_assets/` 目录）
2. 音频设备是否正常（`aplay -l`）
3. 音量是否设置正确
4. 蓝牙设备是否连接（如果配置了）

### Q: 检测延迟很高

**A**: 优化建议：
1. 降低检测帧率（`detection.fps`）
2. 降低摄像头分辨率（`camera.width`, `camera.height`）
3. 使用昇腾 NPU 加速
4. 调整模型输入尺寸

### Q: 误报率很高

**A**: 调整参数：
1. 提高置信度阈值（`detection.conf_threshold`）
2. 调整危险阈值（`decision.danger_threshold`）
3. 过滤不需要的类别

### Q: 摄像头画面黑屏

**A**: 检查：
1. 摄像头是否正确连接
2. 设备 ID 是否正确（`camera.device_id`）
3. 是否有权限访问摄像头
4. 驱动是否安装

## 📈 性能优化建议

### 硬件层面
- 使用昇腾 NPU 可提升 5-10 倍推理速度
- 使用高性能 USB 摄像头（USB 3.0）
- 使用低延迟蓝牙设备

### 软件层面
- 合理设置检测帧率
- 使用模型量化和优化
- 关闭不需要的日志输出
- 使用异步音频播放

### 配置层面
```yaml
# 高性能配置
detection:
  fps: 15
camera:
  width: 1280
  height: 720

# 省电配置
detection:
  fps: 8
camera:
  width: 640
  height: 480
```

## 📝 日常维护

### 查看日志
```bash
# 查看最新日志
tail -n 100 logs/aidsight.log

# 实时查看日志
tail -f logs/aidsight.log

# 搜索错误
grep ERROR logs/aidsight.log
```

### 清理日志
```bash
# 手动清理
rm logs/*.log

# 或者让系统自动轮转（配置文件中设置）
```

### 更新系统
```bash
cd ~/AidSight
git pull
source venv/bin/activate
pip install -r requirements.txt --upgrade
```

## 🔗 更多资源

- [安装指南](installation.md)
- [硬件配置](hardware_setup.md)
- [模型训练](model_training.md)
- [部署指南](deployment.md)

---

**提示**: 首次使用建议在安全环境中测试系统，熟悉各项功能后再实际使用。
