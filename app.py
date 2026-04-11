from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, Query, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import logging
from pydantic import BaseModel, Field
import uvicorn
from voiceapi.tts import TTSResult, start_tts_stream, TTSStream
from voiceapi.asr import start_asr_stream, ASRStream, ASRResult, process_asr_file
from voiceapi.tts import TTSResult, start_tts_stream, TTSStream, get_tts_engine
import logging
import json
import argparse
import os
import soundfile as sf
import io
import numpy as np
from scipy.signal import resample
from typing import List
import time
import copy

# 在 app.py 的最上方 imports 之后加入：
import asyncio
from concurrent.futures import ThreadPoolExecutor
# 1. 顶部导入并设置单个 Worker 的内部线程池为 16
from contextlib import asynccontextmanager

models_root = './models'

for d in ['.', '..', '../..']:
    if os.path.isdir(f'{d}/models'):
        models_root = f'{d}/models'
        break

parser = argparse.ArgumentParser()
parser.add_argument("--workers", type=int, default=1, help="number of uvicorn workers")

# 在 app.py 中添加这一行
parser.add_argument("--no-vad", action="store_true", default=False,
                    help="禁用VAD，收到done信号后整段识别")
parser.add_argument("--speed", type=float, default=1.0, help="设置全局语速")
parser.add_argument("--volume", type=float, default=1.0, help="设置全局volume")
parser.add_argument("--port", type=int, default=8000, help="port number")
parser.add_argument("--addr", type=str,
                    default="0.0.0.0", help="serve address")

parser.add_argument("--asr-provider", type=str,
                    default="cpu", help="asr provider, cpu or cuda")
parser.add_argument("--tts-provider", type=str,
                    default="cpu", help="tts provider, cpu or cuda")

parser.add_argument("--threads", type=int, default=2,
                    help="number of threads")

parser.add_argument("--models-root", type=str, default=models_root,
                    help="model root directory")

parser.add_argument("--asr-model", type=str, default='sensevoice',
                    help="ASR model name: zipformer-bilingual, sensevoice, sensevoice-int8, paraformer-trilingual, paraformer-en, fireredasr")

parser.add_argument("--asr-lang", type=str, default='zh',
                    help="ASR language, zh, en, ja, ko, yue")

parser.add_argument("--tts-model", type=str, default='vits-zh-hf-theresa',
                    help="TTS model name: vits-zh-hf-theresa, vits-melo-tts-zh_en, kokoro-multi-lang-v1_0")

# ★★★ 修复 1：使用 parse_known_args 防止子进程因遇到内部命令而崩溃 ★★★
args, _ = parser.parse_known_args()

# ★★★ 修复 2：使用环境变量“偷渡”用户配置给子进程 ★★★
if "MASTER_ARGS" in os.environ:
    # 如果当前是子进程：直接从环境变量中恢复主进程解析好的真实参数
    args.__dict__.update(json.loads(os.environ["MASTER_ARGS"]))
else:
    # 如果当前是主进程：把带有用户自定义配置的 args 存入环境变量，传给即将孵化的子进程
    os.environ["MASTER_ARGS"] = json.dumps(args.__dict__)

if args.tts_model == 'vits-melo-tts-zh_en' and args.tts_provider == 'cuda':
    logger.warning(
        "vits-melo-tts-zh_en does not support CUDA fallback to CPU")
    args.tts_provider = 'cpu'

@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=16)) # 平衡点
    yield


app = FastAPI(lifespan=lifespan)
logger = logging.getLogger(__file__)


@app.websocket("/asr")
async def websocket_asr(websocket: WebSocket,
                        samplerate: int = Query(16000, title="Sample Rate",
                                                description="The sample rate of the audio."),):
    await websocket.accept()

    asr_stream: ASRStream = await start_asr_stream(samplerate, args)
    if not asr_stream:
        logger.error("failed to start ASR stream")
        await websocket.close()
        return

    async def task_recv_pcm2():
        while True:
            pcm_bytes = await websocket.receive_bytes()
            if not pcm_bytes:
                return
            await asr_stream.write(pcm_bytes)
    async def task_recv_pcm():
        try:
            while True:
                # 使用 receive() 替代 receive_bytes() 以兼容字符串信号
                message = await websocket.receive()
                
                # 处理音频数据
                ## 收到二进制数据时
                #message = {"type": "websocket.receive", "bytes": b'\x00\x01...', "text": None}
                # 收到文本数据时
                #message = {"type": "websocket.receive", "bytes": None, "text": "done"}
                if "bytes" in message:
                    await asr_stream.write(message["bytes"])
                
                # 处理 Java 发来的结束文本 "done"
                elif "text" in message:
                    if message["text"] == "done":
                        logger.info("收到结束信号，等待识别完成...")
                        # 往输入队列写一个空信号，让 run_offline 知道输入结束
                        asr_stream.inbuf.put_nowait(None)
                        break
        except Exception as e:
            logger.error(f"接收任务异常: {e}")
            
    async def task_send_result2():
        while True:
            result: ASRResult = await asr_stream.read()
            if not result:
                return
            await websocket.send_json(result.to_dict())
            
    async def task_send_result():
        while True:
        # 即使暂时没有结果，也应该继续等待，除非 stream 彻底销毁
            result: ASRResult = await asr_stream.read()
        
            if result is None:
            # 只有当 asr_stream.close() 被调用且队列清空后才退出
                await websocket.close(1000, "done")  # ★ 主动正常关闭
                logger.info("ASR 结果读取完毕，退出发送任务")
                break
            
        # 只有真正有文字时才发送
            if result.text.strip():
                await websocket.send_json(result.to_dict())        
            
            
            
    try:
        await asyncio.gather(task_recv_pcm(), task_send_result())
    except WebSocketDisconnect:
        logger.info("asr: disconnected")
    finally:
        await asr_stream.close()


@app.websocket("/tts")
async def websocket_tts(websocket: WebSocket,
                        samplerate: int = Query(16000,
                                                title="Sample Rate",
                                                description="The sample rate of the generated audio."),
                        interrupt: bool = Query(True,
                                                title="Interrupt",
                                                description="Interrupt the current TTS stream when a new text is received."),
                        sid: int = Query(0,
                                         title="Speaker ID",
                                         description="The ID of the speaker to use for TTS."),

                        chunk_size: int = Query(1024,
                                                title="Chunk Size",
                                                description="The size of the chunk to send to the client."),
                        speed: float = Query(1.0,
                                             title="Speed",
                                             description="The speed of the generated audio."),
                        volume: float = Query(1.0,
                                            title="volume ",
                                            description="The volume of the generated audio, 0.5-2.0."),
                        split: bool = Query(True,
                                            title="Split",
                                            description="Split the text into sentences.")):

    await websocket.accept()
    tts_stream: TTSStream = None

    async def task_recv_text():
        nonlocal tts_stream
        while True:
            text = await websocket.receive_text()
            if not text:
                return

            if interrupt or not tts_stream:
                if tts_stream:
                    await tts_stream.close()
                    logger.info("tts: stream interrupt")

                tts_stream = await start_tts_stream(sid, samplerate, speed,volume, args)
                if not tts_stream:
                    logger.error("tts: failed to allocate tts stream")
                    await websocket.close()
                    return
            logger.info(f"tts: received: {text} (split={split})")
            await tts_stream.write(text, split)

    async def task_send_pcm():
        nonlocal tts_stream
        while not tts_stream:
            # wait for tts stream to be created
            await asyncio.sleep(0.1)

        while True:
            result: TTSResult = await tts_stream.read()
            if not result:
                return

            if result.finished:
                await websocket.send_json(result.to_dict())
            else:
                for i in range(0, len(result.pcm_bytes), chunk_size):
                    await websocket.send_bytes(result.pcm_bytes[i:i+chunk_size])

    try:
        await asyncio.gather(task_recv_text(), task_send_pcm())
    except WebSocketDisconnect:
        logger.info("tts: disconnected")
    finally:
        if tts_stream:
            await tts_stream.close()


class TTSRequest(BaseModel):
    text: str = Field(..., title="Text",
                      description="The text to be converted to speech.",
                      examples=["Hello, world!"])
    sid: int = Field(0, title="Speaker ID",
                     description="The ID of the speaker to use for TTS.")
    samplerate: int = Field(16000, title="Sample Rate",
                            description="The sample rate of the generated audio.")
    speed: float = Field(1.0, title="Speed",
                         description="The speed of the generated audio.")
    volume: float = Field(1.0, title="Volume",
                         description="The volume of the generated audio.")

@ app.post("/tts",
           description="Generate speech audio from text.",
           response_class=StreamingResponse, responses={200: {"content": {"audio/wav": {}}}})
 
 
async def tts_generate(req: TTSRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="text is required")

    # 1. 获取模型配置的真实采样率 (Aishell3 会返回 8000)
    _, model_rate = get_tts_engine(args)

    # 2. 传入 model_rate 确保生成 8k 文件，传入 args.speed 应用命令行语速
    tts_stream = await start_tts_stream(
        sid=req.sid, 
        sample_rate=model_rate, 
        speed=args.speed,
        volume=args.volume,
        args=args
    )

    if not tts_stream:
        raise HTTPException(
            status_code=500, detail="failed to start TTS stream")

    r = await tts_stream.generate(req.text)
    return StreamingResponse(r, media_type="audio/wav")




@app.post("/asr_file",
          description="Transcribe an uploaded audio file and return timestamped segments.",
          responses={200: {"description": "Transcription results with timestamps"}})
async def asr_file_endpoint(file: UploadFile = File(...),
                            samplerate: int = Query(16000, description="Target sample rate for processing"),
                            use_int8: bool = Query(False, description="Use the SenseVoice int8 model for offline ASR")):
    if not file.filename.lower().endswith(('.mp3', '.wav', '.ogg')):
        raise HTTPException(status_code=400, detail="Unsupported file format. Currently supports wav and ogg.")
    start_time = time.time()
    file_data = await file.read()
    
    try:
        data, sr = sf.read(io.BytesIO(file_data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read audio file: {str(e)}")

    if sr != samplerate:
        target_samples = int(data.shape[0] * samplerate / sr)
        #data = resample(data, target_samples, axis=0)
        data = await asyncio.to_thread(resample, data, target_samples, axis=0)

    if data.ndim == 1:
        channels = [(0, data)]
    else:
        channels = [(idx, data[:, idx]) for idx in range(data.shape[1])]

    selected_args = args
    if use_int8:
        selected_args = copy.copy(args)
        selected_args.asr_model = 'sensevoice-int8'

    all_results: List[ASRResult] = []
    for channel_idx, channel_samples in channels:
        channel_results = await process_asr_file(np.asarray(channel_samples, dtype=np.float32), samplerate, selected_args, channel=channel_idx)
        all_results.extend(channel_results)

    all_results.sort(key=lambda r: (r.start, r.channel or 0))
    for idx, result in enumerate(all_results):
        result.idx = idx

    elapsed = time.time() - start_time
    audio_duration = float(data.shape[0]) / samplerate if samplerate > 0 else None
    rtf = (elapsed / audio_duration) if audio_duration and audio_duration > 0 else None

    return {
        "elapsed": elapsed,
        "data_length": len(file_data),
        "audio_duration": audio_duration,
        "rtf": rtf,
        "asr_model": selected_args.asr_model,
        "segments": [r.to_dict() for r in all_results]
    }



    #lhw uvicorn.run(app, host=args.addr, port=args.port)

    # 修改前
    # uvicorn.run(app, host=args.addr, port=args.port)

app.mount("/", app=StaticFiles(directory="./assets", html=True), name="assets")
if __name__ == "__main__":
    from logging.handlers import RotatingFileHandler

    logging.basicConfig(
        format='%(levelname)s: %(asctime)s %(name)s:%(lineno)s %(message)s',
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            RotatingFileHandler(
                'asrtts.log',
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
        ]
    )

    uvicorn.run("app:app", host=args.addr, port=args.port, workers=args.workers)
