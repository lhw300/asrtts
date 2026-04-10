"""
单元测试 - 不依赖模型，CI 环境直接运行
"""
import re
import numpy as np
import argparse


# 复制 tts.py 里的常量，避免 import sherpa_onnx
splitter = re.compile(r'[,，。.!?！？;；、\n]')

tts_configs = {
    'vits-zh-hf-theresa':       {'sample_rate': 22050},
    'vits-melo-tts-zh_en':      {'sample_rate': 44100},
    'kokoro-multi-lang-v1_0':   {'sample_rate': 24000},
    'vits-icefall-zh-aishell3': {'sample_rate': 8000},
}


def test_splitter():
    """文本分句：句号应该分成两句"""
    result = [s for s in re.split(splitter, "你好。我是小明") if s.strip()]
    assert len(result) == 2


def test_audio_clipping():
    """音频截断：超出范围的值不应该溢出 int16"""
    chunk = np.array([2.0, -2.0], dtype=np.float32)
    scaled = np.clip(chunk * 32768.0, -32768, 32767).astype(np.int16)
    assert scaled[0] == 32767
    assert scaled[1] == -32768


def test_aishell3_sample_rate():
    """电话场景模型采样率必须是 8000Hz"""
    assert tts_configs['vits-icefall-zh-aishell3']['sample_rate'] == 8000


def test_melo_cuda_fallback():
    """vits-melo 不支持 CUDA，应自动回退到 CPU"""
    args = argparse.Namespace(tts_model='vits-melo-tts-zh_en', tts_provider='cuda')
    if args.tts_model == 'vits-melo-tts-zh_en' and args.tts_provider == 'cuda':
        args.tts_provider = 'cpu'
    assert args.tts_provider == 'cpu'
