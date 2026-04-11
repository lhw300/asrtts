import os
# 1. 第一步：必须在最前面设置环境变量！
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

# 2. 第二步：再导入其他库
from typing import *
import time
import logging
import numpy as np
import asyncio
import soundfile
from scipy.signal import resample
import io
import re

# 3. 第三步：最后导入底层推理库
import sherpa_onnx



logger = logging.getLogger(__file__)

splitter = re.compile(r'[,，。.!?！？;；、\n]')
_tts_engines = {}

tts_configs = {
    'vits-zh-hf-theresa': {
        'model': 'theresa.onnx',
        'lexicon': 'lexicon.txt',
        'dict_dir': 'dict',
        'tokens': 'tokens.txt',
        'sample_rate': 22050,
        'rule_fsts': ['phone.fst', 'date.fst', 'number.fst', 'new_heteronym.fst'],
    },
    'vits-melo-tts-zh_en': {
        'model': 'model.onnx',
        'lexicon': 'lexicon.txt',
        'dict_dir': 'dict',
        'tokens': 'tokens.txt',
        'sample_rate': 44100,
        'rule_fsts': ['phone.fst', 'date.fst', 'number.fst', 'new_heteronym.fst'],
    },
    'kokoro-multi-lang-v1_0': {
        'model': 'model.onnx',
        #'lexicon': ['lexicon-zh.txt','lexicon-us-en.txt','lexicon-gb-en.txt'],
        'lexicon': 'lexicon-zh.txt',
        'dict_dir': 'dict',
        'tokens': 'tokens.txt',
        'sample_rate': 24000,
        'rule_fsts': ['date-zh.fst', 'number-zh.fst'],
    },
    'vits-icefall-zh-aishell3': {
        'model': 'model.onnx',
        'lexicon': 'lexicon.txt',
        'dict_dir': 'dict',
        'tokens': 'tokens.txt',
        'sample_rate': 8000,
        'rule_fsts': ['phone.fst', 'date.fst', 'number.fst', 'new_heteronym.fst'],
    },
}


def load_tts_model(name: str, model_root: str, provider: str, num_threads: int = 1, max_num_sentences: int = 20) -> sherpa_onnx.OfflineTtsConfig:
    cfg = tts_configs[name]
    fsts = []
    model_dir = os.path.join(model_root, name)
    for f in cfg.get('rule_fsts', ''):
        fsts.append(os.path.join(model_dir, f))
    tts_rule_fsts = ','.join(fsts) if fsts else ''

    if 'kokoro' in name:
        kokoro_model_config = sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=os.path.join(model_dir, cfg['model']),
            voices=os.path.join(model_dir, 'voices.bin'),
            lexicon=os.path.join(model_dir, cfg['lexicon']),
            data_dir=os.path.join(model_dir, 'espeak-ng-data'),
            dict_dir=os.path.join(model_dir, cfg['dict_dir']),
            tokens=os.path.join(model_dir, cfg['tokens']),
        )
        model_config = sherpa_onnx.OfflineTtsModelConfig(
            kokoro=kokoro_model_config,
            provider=provider,
            debug=0,
            num_threads=num_threads,
        )
    elif 'vits' in name:
        vits_model_config = sherpa_onnx.OfflineTtsVitsModelConfig(
            model=os.path.join(model_dir, cfg['model']),
            lexicon=os.path.join(model_dir, cfg['lexicon']),
            dict_dir=os.path.join(model_dir, cfg['dict_dir']),
            tokens=os.path.join(model_dir, cfg['tokens']),
        )
        model_config = sherpa_onnx.OfflineTtsModelConfig(
            vits=vits_model_config,
            provider=provider,
            debug=0,
            num_threads=num_threads,
        )
    
    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=model_config,
        rule_fsts=tts_rule_fsts,
        max_num_sentences=max_num_sentences)

    if not tts_config.validate():
        raise ValueError("tts: invalid config")

    return tts_config


def get_tts_engine(args) -> Tuple[sherpa_onnx.OfflineTts, int]:
    sample_rate = tts_configs[args.tts_model]['sample_rate']
    cache_engine = _tts_engines.get(args.tts_model)
    if cache_engine:
        return cache_engine, sample_rate
    st = time.time()
    #tts_config = load_tts_model(
        #args.tts_model, args.models_root, args.tts_provider)
    # 修改后 (补上 num_threads=args.threads)：
    tts_config = load_tts_model(
        args.tts_model, args.models_root, args.tts_provider, num_threads=args.threads)

    cache_engine = sherpa_onnx.OfflineTts(tts_config)
    elapsed = time.time() - st
    logger.info(f"tts: loaded {args.tts_model} in {elapsed:.2f}s")
    _tts_engines[args.tts_model] = cache_engine

    return cache_engine, sample_rate


class TTSResult:
    def __init__(self, pcm_bytes: bytes, finished: bool):
        self.pcm_bytes = pcm_bytes
        self.finished = finished
        self.progress: float = 0.0
        self.elapsed: float = 0.0
        self.audio_duration: float = 0.0
        self.audio_size: int = 0

    def to_dict(self):
        return {
            "progress": self.progress,
            "elapsed": f'{int(self.elapsed * 1000)}ms',
            "duration": f'{self.audio_duration:.2f}s',
            "size": self.audio_size
        }


class TTSStream:
    #def __init__(self, engine, sid: int, speed: float = 1.0, sample_rate: int = 16000, original_sample_rate: int = 16000):
    # 修改前
    # def __init__(self, engine, sid: int, speed: float = 1.0, sample_rate: int = 16000, original_sample_rate: int = 16000):

    # 修改后 (建议去掉硬编码的 16000)
    def __init__(self, engine, sid: int, speed: float = 1.0, volume: float = 1.0, sample_rate: int = 8000, original_sample_rate: int = 8000):
        self.engine = engine
        self.sid = sid
        self.speed = speed
        self.outbuf: asyncio.Queue[TTSResult | None] = asyncio.Queue()
        self.is_closed = False
        self.target_sample_rate = sample_rate
        self.original_sample_rate = original_sample_rate
        self.volume=volume
    def on_process(self, chunk: np.ndarray, progress: float):
        if self.is_closed:
            return 0

        # resample to target sample rate
        if self.target_sample_rate != self.original_sample_rate:
            num_samples = int(
                len(chunk) * self.target_sample_rate / self.original_sample_rate)
            resampled_chunk = resample(chunk, num_samples)
            chunk = resampled_chunk.astype(np.float32)

       #logger.info("tts: chunk="+chunk);
        chunk = chunk *self.volume # add by lhw
       # logger.info("tts: vol chunk="+chunk);
        scaled_chunk = chunk * 32768.0
        clipped_chunk = np.clip(scaled_chunk, -32768, 32767)
        int16_chunk = clipped_chunk.astype(np.int16)   #float---> int
        samples = int16_chunk.tobytes()
        self.outbuf.put_nowait(TTSResult(samples, False))
        return self.is_closed and 0 or 1

    async def write(self, text: str, split: bool, pause: float = 0.2):
        start = time.time()
        if split:
            texts = re.split(splitter, text)
        else:
            texts = [text]

        audio_duration = 0.0
        audio_size = 0

        for idx, text in enumerate(texts):
            text = text.strip()
            if not text:
                continue
            sub_start = time.time()

            audio = await asyncio.to_thread(self.engine.generate,
                                            text, self.sid, self.speed,
                                            self.on_process)

            if not audio or not audio.sample_rate or not audio.samples:
                logger.error(f"tts: failed to generate audio for "
                             f"'{text}' (audio={audio})")
                continue

            if split and idx < len(texts) - 1:  # add a pause between sentences
                noise = np.zeros(int(audio.sample_rate * pause))
                self.on_process(noise, 1.0)
                audio.samples = np.concatenate([audio.samples, noise])

            audio_duration += len(audio.samples) / audio.sample_rate
            audio_size += len(audio.samples)
            elapsed_seconds = time.time() - sub_start
            logger.info(f"tts: generated audio for '{text}', "
                        f"audio duration: {audio_duration:.2f}s, "
                        f"elapsed: {elapsed_seconds:.2f}s")

        elapsed_seconds = time.time() - start
        logger.info(f"tts: generated audio in {elapsed_seconds:.2f}s, "
                    f"audio duration: {audio_duration:.2f}s")

        r = TTSResult(None, True)
        r.elapsed = elapsed_seconds
        r.audio_duration = audio_duration
        r.progress = 1.0
        r.finished = True
        await self.outbuf.put(r)

    async def close(self):
        self.is_closed = True
        self.outbuf.put_nowait(None)
        logger.info("tts: stream closed")

    async def read(self) -> TTSResult:
        return await self.outbuf.get()
    '''
    async def generate(self,  text: str) -> io.BytesIO:
        start = time.time()
        audio = await asyncio.to_thread(self.engine.generate,
                                        text, self.sid, self.speed)
        elapsed_seconds = time.time() - start
        audio_duration = len(audio.samples) / audio.sample_rate

        logger.info(f"tts: generated audio in {elapsed_seconds:.2f}s, "
                    f"audio duration: {audio_duration:.2f}s, "
                    f"sample rate: {audio.sample_rate}")

        if self.target_sample_rate != audio.sample_rate:
            audio.samples = resample(audio.samples,
                                     int(len(audio.samples) * self.target_sample_rate / audio.sample_rate))
            audio.sample_rate = self.target_sample_rate

        output = io.BytesIO()
        soundfile.write(output,
                        audio.samples,
                        samplerate=audio.sample_rate,
                        subtype="ALAW",
                        format="WAV")
        output.seek(0)
        return output
    '''
    async def generate(self, text: str) -> io.BytesIO:
        start = time.time()
        #audio = await asyncio.to_thread(self.engine.generate,
        #                                text, self.sid, self.speed)
                                        
                              
                                        
        # 修改后
        audio = await asyncio.to_thread(self.engine.generate,
                                text, 
                                self.sid, 
                                self.speed)                                
                                        
        print(f"DEBUG: 目标采样率配置为: {self.target_sample_rate}")
        
        # --- 以下是新逻辑，确保输出物理文件为 8000Hz ---
        if not audio or audio.sample_rate <= 0:
            logger.error("TTS 合成失败：采样率无效")
            return io.BytesIO()

        elapsed_seconds = time.time() - start
        audio_duration = len(audio.samples) / audio.sample_rate
        
        logger.info(f"tts: generated audio in {elapsed_seconds:.2f}s, "
                    f"duration: {audio_duration:.2f}s, "
                    f"original rate: {audio.sample_rate}")

        #final_samples = audio.samples
        #final_sample_rate = audio.sample_rate
        final_samples = np.array(audio.samples, dtype=np.float32) * self.volume
        final_samples = np.clip(final_samples, -1.0, 1.0)
        final_sample_rate = audio.sample_rate

        print(f"DEBUG: 1 正在写入文件，采样率设定为: {final_sample_rate}")
        # 执行重采样
        if self.target_sample_rate and self.target_sample_rate != audio.sample_rate:
            num_samples = int(len(audio.samples) * self.target_sample_rate / audio.sample_rate)
            final_samples = resample(audio.samples, num_samples)
            final_sample_rate = self.target_sample_rate
            
        print(f"DEBUG: 2 正在写入文件，采样率设定为: {final_sample_rate}")
        output = io.BytesIO()
        soundfile.write(output,
                        final_samples,
                        samplerate=final_sample_rate, 
                        subtype="ALAW",
                        format="WAV")
        output.seek(0)
        return output
'''
async def start_tts_stream(sid: int, sample_rate: int, speed: float, args) -> TTSStream:
    engine, original_sample_rate = get_tts_engine(args)
    return TTSStream(engine, sid, speed, sample_rate, original_sample_rate) 
'''
# 修改文件最后几行
async def start_tts_stream(sid: int, sample_rate: int, speed: float, volume: float,args) -> TTSStream:
    # get_tts_engine 会返回模型在 tts_configs 里配置的采样率 (即 8000)
    engine, model_config_sample_rate = get_tts_engine(args) 
    
    # 显式将 model_config_sample_rate 传给 TTSStream
    return TTSStream(engine, sid, speed, volume,model_config_sample_rate, model_config_sample_rate)
