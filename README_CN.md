[English](README.md)

# VoiceAPI - 流式语音识别与合成服务

基于 [ruzhila/voiceapi](https://github.com/ruzhila/voiceapi) 修改开发。
在原项目基础上针对**电话呼叫中心场景**进行了专项优化，增加了多进程并发、音量控制、VAD 可关闭、8kHz 电话音频支持等功能。

## 相较原项目的改动

### app.py

| 改动 | 说明 |
|---|---|
| `--workers` 参数 | 支持多进程启动，充分利用多核 CPU |
| `--no-vad` 参数 | 禁用 VAD，收到 `done` 信号后整段识别，适合电话场景 |
| `--speed` 参数 | 全局设置 TTS 语速 |
| `--volume` 参数 | 全局设置 TTS 音量 |
| 子进程参数传递修复 | 使用环境变量 `MASTER_ARGS` 解决多 worker 模式下参数丢失问题 |
| `parse_known_args` | 防止子进程因遇到内部命令行参数而崩溃 |
| `ThreadPoolExecutor` | 启动时设置线程池为 16，提升并发推理吞吐 |
| ASR WebSocket 兼容性 | 支持混合接收二进制音频 + 文本 `done` 信号（兼容 Java 客户端） |
| ASR 结果过滤 | 空文本结果不发送，减少无效网络传输 |
| 日志轮转 | `RotatingFileHandler`，单文件最大 10MB，保留 5 份 |
| `/asr_file` 重采样异步化 | 使用 `asyncio.to_thread` 避免阻塞事件循环 |
| POST `/tts` 采样率修复 | 自动读取模型真实采样率，避免 8kHz 模型输出错误采样率 |

### tts.py

| 改动 | 说明 |
|---|---|
| 新增 `vits-icefall-zh-aishell3` 模型 | 原生 8kHz 输出，专为电话场景设计 |
| `volume` 参数 | `TTSStream` 支持音量调节，在 `on_process` 回调中对音频数据乘系数 |
| CUDA 延迟加载 | 顶部设置 `CUDA_MODULE_LOADING=LAZY`，加快非 GPU 场景启动速度 |
| `TTSStream` 默认采样率 | 从硬编码 16000 改为读取模型配置的真实采样率 |
| `generate()` 重采样逻辑 | 重写为显式目标采样率控制，确保输出文件采样率与模型配置一致 |

---

## 支持的模型

### ASR 语音识别

| 模型名 | 语言 | 类型 |
|---|---|---|
| `sensevoice` | 中/英/日/韩/粤 | 离线 |
| `sensevoice-int8` | 中/英/日/韩/粤 | 离线（量化，更快） |
| `paraformer-zh` | 中文 | 离线 |
| `paraformer-zh-int8` | 中文 | 离线（量化） |
| `paraformer-trilingual` | 中/粤/英 | 离线 |
| `paraformer-en` | 英文 | 离线 |
| `zipformer-bilingual` | 中/英 | 在线（流式） |
| `fireredasr` | 中/英 | 离线 |

### TTS 语音合成

| 模型名 | 语言 | 采样率 | 说话人数 |
|---|---|---|---|
| `vits-zh-hf-theresa` | 中文 | 22050Hz | 804 |
| `vits-melo-tts-zh_en` | 中/英 | 44100Hz | 1 |
| `kokoro-multi-lang-v1_0` | 中/英 | 24000Hz | 53 |
| `vits-icefall-zh-aishell3` ⭐ | 中文 | **8000Hz** | 多 |

> ⭐ `vits-icefall-zh-aishell3` 为新增模型，原生 8kHz 输出，适合电话呼叫中心场景

---

## 环境要求

- Python 3.10+
- 依赖见 `requirements.txt`（CPU）或 `requirements.cuda.txt`（GPU）

---

## 快速开始

```bash
# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 安装依赖
pip install -r requirements.txt

# 启动服务（默认）
python app.py

# 指定模型
python app.py --asr-model sensevoice --tts-model vits-zh-hf-theresa

# 电话场景（8kHz + 无VAD + 多进程）
python app.py --asr-model paraformer-zh-int8 --tts-model vits-icefall-zh-aishell3 --no-vad --workers 4 --threads 3
```

访问 `http://localhost:8000` 查看演示页面。

---

## 启动参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--asr-model` | `sensevoice` | ASR 模型名 |
| `--tts-model` | `vits-zh-hf-theresa` | TTS 模型名 |
| `--workers` | `1` | uvicorn 进程数（多核并发） |
| `--threads` | `2` | ONNX 推理线程数 |
| `--no-vad` | `False` | 禁用 VAD，收到 done 信号后整段识别 |
| `--speed` | `1.0` | TTS 全局语速 |
| `--volume` | `1.0` | TTS 全局音量 |
| `--port` | `8000` | 监听端口 |
| `--asr-provider` | `cpu` | ASR 推理后端：cpu 或 cuda |
| `--tts-provider` | `cpu` | TTS 推理后端：cpu 或 cuda |

---

## 模型文件

模型文件需单独下载，放到 `models/` 目录（不含在仓库中）。
下载方式参考 [原项目文档](https://github.com/ruzhila/voiceapi#download-models)。

---

## 并发配置参考

| 并发路数 | 推荐配置 |
|---|---|
| 5 路以下 | `--workers 2 --threads 3`，8 核 CPU |
| 5~15 路 | `--workers 4 --threads 3`，16 核 CPU |
| 15 路以上 | `--workers 8 --threads 3`，32 核+ 服务器 CPU |

---

## 致谢

本项目基于 [ruzhila/voiceapi](https://github.com/ruzhila/voiceapi) 修改开发，感谢原作者的开源贡献。
底层推理引擎使用 [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)。
