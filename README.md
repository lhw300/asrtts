[中文文档](README_CN.md)

# VoiceAPI - Streaming ASR & TTS Service

Built on top of [ruzhila/voiceapi](https://github.com/ruzhila/voiceapi), with targeted optimizations for **telephone call center scenarios**, including multi-process concurrency, volume control, optional VAD, and native 8kHz audio support.

## Changes from Original Project

### app.py

| Change | Description |
|---|---|
| `--workers` parameter | Multi-process startup to fully utilize multi-core CPUs |
| `--no-vad` parameter | Disables VAD; recognizes the full segment after receiving a `done` signal — ideal for telephony |
| `--speed` parameter | Global TTS speech rate control |
| `--volume` parameter | Global TTS volume control |
| Sub-process argument fix | Uses `MASTER_ARGS` environment variable to pass config to worker processes correctly |
| `parse_known_args` | Prevents worker process crashes caused by internal CLI arguments |
| `ThreadPoolExecutor` | Sets thread pool to 16 at startup to improve concurrent inference throughput |
| ASR WebSocket compatibility | Supports mixed reception of binary audio + text `done` signal (compatible with Java clients) |
| ASR result filtering | Empty text results are not sent, reducing unnecessary network traffic |
| Log rotation | `RotatingFileHandler` with 10MB max file size, retaining 5 backups |
| `/asr_file` async resampling | Uses `asyncio.to_thread` to avoid blocking the event loop |
| POST `/tts` sample rate fix | Automatically reads the model's true sample rate to prevent incorrect output for 8kHz models |

### tts.py

| Change | Description |
|---|---|
| Added `vits-icefall-zh-aishell3` model | Native 8kHz output, designed for telephone scenarios |
| `volume` parameter | `TTSStream` supports volume adjustment via coefficient applied in `on_process` callback |
| CUDA lazy loading | Sets `CUDA_MODULE_LOADING=LAZY` at startup to speed up launch in non-GPU environments |
| `TTSStream` default sample rate | Changed from hardcoded 16000 to the model's configured sample rate |
| `generate()` resampling logic | Rewritten with explicit target sample rate control to ensure output matches model config |

---

## Supported Models

### ASR (Speech Recognition)

| Model | Language | Type |
|---|---|---|
| `sensevoice` | ZH / EN / JA / KO / YUE | Offline |
| `sensevoice-int8` | ZH / EN / JA / KO / YUE | Offline (quantized, faster) |
| `paraformer-zh` | Chinese | Offline |
| `paraformer-zh-int8` | Chinese | Offline (quantized) |
| `paraformer-trilingual` | ZH / Cantonese / EN | Offline |
| `paraformer-en` | English | Offline |
| `zipformer-bilingual` | ZH / EN | Online (streaming) |
| `fireredasr` | ZH / EN | Offline |

### TTS (Speech Synthesis)

| Model | Language | Sample Rate | Speakers |
|---|---|---|---|
| `vits-zh-hf-theresa` | Chinese | 22050Hz | 804 |
| `vits-melo-tts-zh_en` | ZH / EN | 44100Hz | 1 |
| `kokoro-multi-lang-v1_0` | ZH / EN | 24000Hz | 53 |
| `vits-icefall-zh-aishell3` ⭐ | Chinese | **8000Hz** | Multiple |

> ⭐ `vits-icefall-zh-aishell3` is a newly added model with native 8kHz output, optimized for telephone call center use cases.

---

## Requirements

- Python 3.10+
- See `requirements.txt` (CPU) or `requirements.cuda.txt` (GPU)

---

## Quick Start

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Start with default settings
python app.py

# Specify models
python app.py --asr-model sensevoice --tts-model vits-zh-hf-theresa

# Telephony scenario (8kHz + no VAD + multi-process)
python app.py --asr-model paraformer-zh-int8 --tts-model vits-icefall-zh-aishell3 --no-vad --workers 4 --threads 3
```

Visit `http://localhost:8000` to see the demo page.

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `--asr-model` | `sensevoice` | ASR model name |
| `--tts-model` | `vits-zh-hf-theresa` | TTS model name |
| `--workers` | `1` | Number of uvicorn worker processes |
| `--threads` | `2` | Number of ONNX inference threads |
| `--no-vad` | `False` | Disable VAD; recognize full segment on `done` signal |
| `--speed` | `1.0` | Global TTS speech rate |
| `--volume` | `1.0` | Global TTS volume |
| `--port` | `8000` | Listening port |
| `--asr-provider` | `cpu` | ASR inference backend: `cpu` or `cuda` |
| `--tts-provider` | `cpu` | TTS inference backend: `cpu` or `cuda` |

---

## Model Files

Model files must be downloaded separately and placed in the `models/` directory (not included in this repository).
See the [original project documentation](https://github.com/ruzhila/voiceapi#download-models) for download instructions.

---

## Concurrency Reference

| Concurrent Sessions | Recommended Config |
|---|---|
| Up to 5 | `--workers 2 --threads 3`, 8-core CPU |
| 5 to 15 | `--workers 4 --threads 3`, 16-core CPU |
| 15+ | `--workers 8 --threads 3`, 32-core+ server CPU |

---

## Acknowledgements

This project is built upon [ruzhila/voiceapi](https://github.com/ruzhila/voiceapi). Thanks to the original author for the open source contribution.
Inference engine powered by [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).
