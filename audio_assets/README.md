# 音频资源目录

本目录用于存放 AidSight 系统使用的预录音频文件。

## 📋 需要录制的音频文件

### 障碍物提示

| 文件名 | 内容 | 用途 |
|--------|------|------|
| `obstacle_front.wav` | "前方有障碍物" | 检测到前方有障碍物 |
| `obstacle_left.wav` | "左侧有障碍物" | 检测到左侧有障碍物 |
| `obstacle_right.wav` | "右侧有障碍物" | 检测到右侧有障碍物 |
| `danger_close.wav` | "危险，请停止" | 检测到非常近的障碍物 |

### 红绿灯提示

| 文件名 | 内容 | 用途 |
|--------|------|------|
| `red_light.wav` | "红灯，请等待" | 检测到红灯 |
| `yellow_light.wav` | "黄灯，请减速" | 检测到黄灯 |
| `green_light.wav` | "绿灯，可通行" | 检测到绿灯 |

### 系统提示

| 文件名 | 内容 | 用途 |
|--------|------|------|
| `system_start.wav` | "系统已启动" | 系统启动完成 |
| `system_stop.wav` | "系统已停止" | 系统关闭 |

## 🎙️ 录制指南

### 录制要求

1. **格式**: WAV 格式（推荐）
2. **采样率**: 16000 Hz 或 22050 Hz
3. **位深度**: 16-bit
4. **声道**: 单声道（Mono）
5. **音量**: 适中，避免爆音
6. **语速**: 适中偏慢，吐字清晰
7. **语气**: 平和、友好

### 录制建议

- 使用专业麦克风或高质量录音设备
- 在安静的环境中录制，减少背景噪音
- 可以录制多个版本，选择最清晰的
- 录制完成后进行降噪处理
- 统一各音频文件的音量大小

### 推荐工具

#### Linux / macOS
- **Audacity** (开源、跨平台)
  ```bash
  sudo apt-get install audacity  # Ubuntu
  brew install audacity           # macOS
  ```

- **命令行录音**
  ```bash
  # 使用 arecord (Linux)
  arecord -f cd -d 5 obstacle_front.wav
  
  # 使用 sox
  rec -r 16000 -c 1 obstacle_front.wav
  ```

#### Windows
- **Audacity** (推荐)
- **Adobe Audition** (专业)
- **Windows 录音机**

## 🔧 音频转换

如果音频格式不符合要求，可以使用以下工具转换：

### 使用 FFmpeg
```bash
# 安装 ffmpeg
sudo apt-get install ffmpeg  # Ubuntu
brew install ffmpeg           # macOS

# 转换音频格式
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav

# 批量转换
for f in *.mp3; do
    ffmpeg -i "$f" -ar 16000 -ac 1 "${f%.mp3}.wav"
done
```

### 使用 SoX (Sound eXchange)
```bash
# 安装 sox
sudo apt-get install sox  # Ubuntu
brew install sox           # macOS

# 转换音频
sox input.mp3 -r 16000 -c 1 output.wav
```

## 🧪 测试音频

录制完成后，可以使用测试脚本验证音频文件：

```bash
# 播放单个音频文件
aplay obstacle_front.wav  # Linux
afplay obstacle_front.wav # macOS

# 运行音频测试
python tests/test_audio.py
```

## 📝 使用 TTS（可选）

如果不想录制音频，也可以使用 TTS（文本转语音）自动生成：

### 使用 gTTS (Google Text-to-Speech)
```python
from gtts import gTTS

texts = {
    'obstacle_front': '前方有障碍物',
    'obstacle_left': '左侧有障碍物',
    'obstacle_right': '右侧有障碍物',
    'danger_close': '危险，请停止',
    'red_light': '红灯，请等待',
    'yellow_light': '黄灯，请减速',
    'green_light': '绿灯，可通行',
    'system_start': '系统已启动',
    'system_stop': '系统已停止',
}

for key, text in texts.items():
    tts = gTTS(text=text, lang='zh-cn')
    tts.save(f'{key}.wav')
```

### 使用 pyttsx3 (离线 TTS)
```python
import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 150)  # 语速
engine.setProperty('volume', 0.9)  # 音量

texts = {
    'obstacle_front': '前方有障碍物',
    # ... 其他文本
}

for key, text in texts.items():
    engine.save_to_file(text, f'{key}.wav')
    
engine.runAndWait()
```

## 🔗 相关资源

- [Audacity 官网](https://www.audacityteam.org/)
- [FFmpeg 官网](https://ffmpeg.org/)
- [gTTS 文档](https://gtts.readthedocs.io/)
- [pyttsx3 文档](https://pyttsx3.readthedocs.io/)

## 📄 许可证

音频文件请使用开源许可证或确保有使用权限。
