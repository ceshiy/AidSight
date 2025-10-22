## 🎯 瞳伴 · AidSight  
## 基于昇腾AI的多模态实时智能导盲系统

> **让AI成为视障人士的“第二双眼睛”**  
> 融合 MindYOLO 目标检测与大语言模型（LLM），实时生成简洁、可操作的安全出行提示。

---

## 🌟 项目简介

**瞳伴 · AidSight** 是一个面向视障人群的智能导盲辅助系统原型。  
系统通过摄像头实时感知环境，利用 **MindYOLO** 检测/分割障碍物，并调用 **OpenAI 兼容大模型 API** 生成如“前方有台阶，请小心”这样的语义化语音提示（文本形式），帮助用户安全、自信出行。

- ✅ 支持 **目标检测（detect）** 与 **实例分割（segment）** 双模式  
- ✅ 集成 **PyQt5 图形界面**，直观展示视频与提示  
- ✅ 智能节流：仅当场景显著变化时调用 LLM，避免冗余  
- ✅ 深度适配 **昇腾AI全栈生态**（MindSpore + Ascend）

---

## 🧩 技术栈

- **AI框架**：[MindSpore](https://www.mindspore.cn/) ≥ 2.3  
- **模型库**：[MindYOLO](https://github.com/mindspore-lab/mindyolo)  
- **硬件支持**：昇腾 Ascend / GPU / CPU  
- **前端界面**：PyQt5  
- **LLM接口**：OpenAI 兼容 API（如 `api.aibh.site/v1`）

---

## 🚀 快速开始

### 1. 环境依赖

```bash
pip install opencv-python pyyaml requests pyqt5 numpy
# MindSpore 安装请参考官方文档（根据硬件选择 Ascend/GPU/CPU 版本）
# https://www.mindspore.cn/install
