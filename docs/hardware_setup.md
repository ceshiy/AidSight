# 硬件配置指南

本文档介绍 AidSight 系统的硬件选择、连接和配置。

## 🖥️ 硬件清单

### 必需硬件

| 组件 | 规格要求 | 推荐型号 | 价格参考 |
|------|----------|----------|----------|
| 开发板 | 支持 Ubuntu，至少 4GB RAM | 香橙派 AI Pro (T20) | ¥899 |
| USB 摄像头 | 720p 或更高，30fps | 罗技 C920 / 小米摄像头 | ¥200-500 |
| 音频输出 | 蓝牙 5.0 或 3.5mm | 韶音骨传导耳机 / 普通耳机 | ¥300-800 |
| 电源 | 5V/3A 或更高 | 官方电源适配器 | ¥50 |
| Micro SD 卡 | 至少 32GB，Class 10 | SanDisk / Samsung | ¥50-100 |

### 可选硬件

| 组件 | 用途 | 推荐型号 | 价格参考 |
|------|------|----------|----------|
| 移动电源 | 便携使用 | 小米移动电源 20000mAh | ¥100-200 |
| 散热风扇 | 长时间运行散热 | 开发板专用散热风扇 | ¥20-50 |
| 外壳 | 保护和便携 | 定制外壳 / 3D 打印 | ¥50-100 |
| GPS 模块 | 位置记录 | U-blox NEO-6M | ¥50-100 |

## 🔌 硬件连接

### 香橙派 AI Pro 接口说明

```
┌─────────────────────────────────┐
│  [USB-C]  [HDMI]  [3.5mm音频]   │
│                                 │
│     香橙派 AI Pro (T20)          │
│                                 │
│  [USB3.0] [USB2.0] [GPIO]       │
│                                 │
│  [网口]   [Micro SD]  [电源]    │
└─────────────────────────────────┘
```

### 连接步骤

#### 1. USB 摄像头连接

**推荐连接方式**:
- 使用 USB 3.0 接口（蓝色接口）以获得更好性能
- 如果摄像头自带线缆较短，使用 USB 延长线

**连接步骤**:
```bash
# 1. 插入 USB 摄像头
# 2. 检查是否识别
ls /dev/video*

# 应该看到 /dev/video0 或类似设备

# 3. 测试摄像头
v4l2-ctl --device=/dev/video0 --all

# 4. 拍摄测试照片
ffmpeg -f v4l2 -i /dev/video0 -frames 1 test.jpg
```

#### 2. 蓝牙音频设备配对

**骨传导耳机推荐**:
- 韶音 OpenMove
- 韶音 OpenRun
- 飞利浦 TAA7607

**配对步骤**:
```bash
# 1. 启动蓝牙服务
sudo systemctl start bluetooth
sudo systemctl enable bluetooth

# 2. 进入蓝牙控制界面
bluetoothctl

# 3. 在 bluetoothctl 中执行
power on
agent on
default-agent
scan on

# 4. 找到设备后（记下 MAC 地址）
# 例如: Device AA:BB:CC:DD:EE:FF OpenMove

# 5. 配对和连接
pair AA:BB:CC:DD:EE:FF
trust AA:BB:CC:DD:EE:FF
connect AA:BB:CC:DD:EE:FF

# 6. 退出
quit

# 7. 测试音频
aplay -D bluealsa /usr/share/sounds/alsa/Front_Center.wav
```

**配置文件更新**:
```yaml
# config.yaml
audio:
  bluetooth_device: "AA:BB:CC:DD:EE:FF"  # 替换为实际 MAC 地址
```

#### 3. 电源连接

**选项 1: 固定使用（室内）**
- 使用官方 5V/3A 电源适配器
- 连接到稳定的电源插座

**选项 2: 便携使用（户外）**
- 使用移动电源（推荐 20000mAh 以上）
- 确保移动电源支持 5V/3A 输出
- 使用优质 USB-C 数据线

**功耗估算**:
- 空闲: ~3W
- 运行 AidSight: ~8-12W
- 使用 20000mAh 移动电源可运行约 6-8 小时

## 🎛️ 硬件配置

### 摄像头设置

#### 调整摄像头参数
```bash
# 查看支持的分辨率
v4l2-ctl --device=/dev/video0 --list-formats-ext

# 设置亮度
v4l2-ctl --device=/dev/video0 --set-ctrl=brightness=128

# 设置对比度
v4l2-ctl --device=/dev/video0 --set-ctrl=contrast=32

# 设置自动曝光
v4l2-ctl --device=/dev/video0 --set-ctrl=exposure_auto=3

# 禁用自动对焦（提高稳定性）
v4l2-ctl --device=/dev/video0 --set-ctrl=focus_auto=0
```

#### 固定摄像头位置

**佩戴方案**:

1. **胸前佩戴**（推荐）
   - 视野范围: 前方 60-90°
   - 高度: 胸口位置（约 1.2-1.4m）
   - 角度: 略向下倾斜 10-15°
   - 优点: 稳定、覆盖地面障碍物

2. **帽子/头部佩戴**
   - 视野范围: 前方 90-120°
   - 高度: 约 1.6-1.8m
   - 角度: 向下倾斜 20-30°
   - 优点: 视野更接近人眼，便于检测红绿灯

3. **手持或导盲杖安装**
   - 高度: 可调节
   - 角度: 可调节
   - 优点: 灵活性高
   - 缺点: 不够稳定

**固定方法**:
- 使用魔术贴或夹子固定
- 3D 打印专用固定架
- 使用 GoPro 配件系统

### 音频输出优化

#### 骨传导耳机优势
- 不遮挡耳朵，保持环境音感知
- 适合长时间佩戴
- 防汗防水

#### 音频设置
```bash
# 设置默认音频输出为蓝牙
# 编辑 ~/.asoundrc
pcm.!default {
    type plug
    slave.pcm "bluealsa"
}

# 调整音量
amixer set Master 80%

# 测试左右声道
speaker-test -c 2 -t wav
```

### 散热管理

**散热方案**:

1. **被动散热**
   - 安装散热片
   - 确保通风良好
   - 避免密闭外壳

2. **主动散热**
   - 安装 5V 小风扇
   - 连接到 GPIO 5V 引脚
   - 使用 PWM 控制风扇转速

**温度监控**:
```bash
# 查看 CPU 温度
cat /sys/class/thermal/thermal_zone0/temp
# 输出单位是毫度（例如 45000 表示 45°C）

# 持续监控
watch -n 1 'cat /sys/class/thermal/thermal_zone0/temp'

# 如果温度过高（>75°C），考虑降低性能或加强散热
```

## 🔧 系统级配置

### GPIO 配置（可选）

如果需要连接额外的传感器或指示灯：

```python
# 安装 GPIO 库
pip install OrangePi.GPIO

# 示例代码
import OPi.GPIO as GPIO
import time

# 设置 GPIO 模式
GPIO.setmode(GPIO.BOARD)

# 设置引脚为输出（例如 LED 指示灯）
LED_PIN = 11
GPIO.setup(LED_PIN, GPIO.OUT)

# 闪烁 LED
for i in range(5):
    GPIO.output(LED_PIN, GPIO.HIGH)
    time.sleep(0.5)
    GPIO.output(LED_PIN, GPIO.LOW)
    time.sleep(0.5)

# 清理
GPIO.cleanup()
```

### 权限配置

```bash
# 添加用户到必要的组
sudo usermod -a -G video,audio,bluetooth,gpio $USER

# 重新登录使权限生效
```

### 自动挂载和启动

```bash
# 编辑 /etc/fstab 自动挂载 SD 卡（如果需要）

# 设置开机自启
sudo systemctl enable aidsight
```

## 📐 外壳设计

### 3D 打印外壳

推荐尺寸（单位: mm）:
- 长度: 120
- 宽度: 80
- 高度: 40
- 散热孔: 多个 5mm 直径

**设计要点**:
1. 预留 USB 接口位置
2. 摄像头固定位
3. 通风孔设计
4. 便携挂绳孔
5. 电源按钮位置

**材料选择**:
- PLA（易打印，环保）
- ABS（更耐用，耐高温）
- PETG（折中选择）

## 🔋 电源管理

### 低功耗配置

```bash
# 降低 CPU 频率
echo "powersave" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 降低屏幕亮度（如果有屏幕）
echo 50 | sudo tee /sys/class/backlight/*/brightness

# 禁用不必要的服务
sudo systemctl disable bluetooth  # 如果不用蓝牙
```

### 电量监控

```bash
# 如果使用带智能管理的移动电源，可以读取电量

# 或者使用 UPS HAT（不间断电源扩展板）
# 安装对应的监控脚本
```

## 📊 性能测试

### 摄像头性能测试

```bash
# 测试不同分辨率的帧率
python - << EOF
import cv2
import time

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

start = time.time()
frame_count = 0

while time.time() - start < 10:
    ret, frame = cap.read()
    if ret:
        frame_count += 1

fps = frame_count / (time.time() - start)
print(f"FPS: {fps:.2f}")

cap.release()
EOF
```

### 整体系统性能测试

```bash
# 运行系统并监控性能
./scripts/start_aidsight.sh --log-level DEBUG &

# 监控 CPU 和内存使用
top -p $(pgrep -f "python.*main.py")

# 或使用 htop
htop -p $(pgrep -f "python.*main.py")
```

## 🛠️ 故障排查

### 摄像头问题
- 检查 USB 连接是否松动
- 尝试不同的 USB 接口
- 检查摄像头驱动

### 蓝牙连接问题
- 重启蓝牙服务: `sudo systemctl restart bluetooth`
- 重新配对设备
- 检查设备电量

### 散热问题
- 清理灰尘
- 改善通风
- 添加散热装置
- 降低性能要求

## 🔗 相关资源

- [香橙派官方文档](http://www.orangepi.org/)
- [V4L2 工具指南](https://www.kernel.org/doc/html/v4.9/media/uapi/v4l/v4l2.html)
- [BlueZ 蓝牙指南](http://www.bluez.org/)

---

**下一步**: 参考[安装指南](installation.md)完成软件安装。
