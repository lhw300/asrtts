document.addEventListener('alpine:init', () => {
    Alpine.data('demoapp', () => ({
        // --- 1. 变量定义 (确保 index.html 中的 x-model 都能找到对应变量) ---
        text: '通知：光猫拨号异常，请各单位派员核查。',
        targetSid: 6,         // 默认说话人ID
        targetSpeed: 0.95,    // 默认语速
        targetRate: 8000,     // 默认采样率 (适配 aishell3)
        
        recording: false,
        asrWS: null,
        currentText: null,
        disabled: false,
        elapsedTime: null,
        logs: [],
        
        file: null,
        fileResults: [],
        fileElapsed: null,
        fileSize: null,
        fileAudioDuration: null,
        fileRtf: null,
        fileModel: null,
        useInt8: false,

        // --- 2. 工具函数 (用于动态构建 WebSocket 完整地址) ---
        getWsUrl(path) {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            return `${protocol}//${window.location.host}${path}`;
        },

        formatBytes(bytes) {
            if (typeof bytes !== 'number' || !isFinite(bytes)) return '-';
            if (bytes === 0) return '0 B';
            const units = ['B', 'KB', 'MB', 'GB', 'TB'];
            const exponent = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
            const value = bytes / Math.pow(1024, exponent);
            const digits = value >= 10 ? 0 : 1;
            return `${value.toFixed(digits)} ${units[exponent]}`;
        },

        formatSeconds(value) {
            if (typeof value !== 'number' || !isFinite(value)) return '-';
            if (value === 0) return '0s';
            const digits = value >= 10 ? 1 : 2;
            return `${value.toFixed(digits)}s`;
        },

        // --- 3. TTS (文本转语音) 逻辑 ---
        async dotts() {
            const rate = parseInt(this.targetRate || 8000);
            const audioContext = new AudioContext({ sampleRate: rate });
            
            if (audioContext.state === 'suspended') {
                await audioContext.resume();
            }
            
            try {
                await audioContext.audioWorklet.addModule('./audio_process.js');
                
                // 修复点：使用 getWsUrl 构建完整地址，解决 URL invalid 报错
                const wsUrl = this.getWsUrl(`/tts?samplerate=${rate}&sid=${this.targetSid}&speed=${this.targetSpeed}&split=true`);
                console.log("TTS 连接中:", wsUrl);
                const ws = new WebSocket(wsUrl);

                ws.onopen = () => {
                    ws.send(this.text);
                };

                const playNode = new AudioWorkletNode(audioContext, 'play-audio-processor');
                playNode.connect(audioContext.destination);

                this.disabled = true;
                ws.onmessage = async (e) => {
                    if (e.data instanceof Blob) {
                        const arrayBuffer = await e.data.arrayBuffer();
                        const int16Array = new Int16Array(arrayBuffer);
                        const float32Array = new Float32Array(int16Array.length);
                        for (let i = 0; i < int16Array.length; i++) {
                            float32Array[i] = int16Array[i] / 32768.;
                        }
                        playNode.port.postMessage({ message: 'audioData', audioData: float32Array });
                    } else {
                        const result = JSON.parse(e.data);
                        this.elapsedTime = result?.elapsed;
                        this.disabled = false;
                    }
                };

                ws.onerror = (err) => {
                    console.error("TTS WebSocket 错误:", err);
                    this.disabled = false;
                };
            } catch (err) {
                console.error("AudioWorklet 加载失败:", err);
            }
        },

        // --- 4. 在线 ASR (语音识别) 逻辑 ---
        async doasr() {
            try {
                const mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                
                // 修复点：录音识别同样需要完整 ws:// 地址
                const wsUrl = this.getWsUrl('/asr');
                const ws = new WebSocket(wsUrl);
                
                ws.onopen = () => {
                    this.logs = [];
                };

                ws.onmessage = (e) => {
                    const data = JSON.parse(e.data);
                    const { text, finished, idx } = data;
                    this.currentText = text;

                    if (finished) {
                        this.logs.push({ text: text, idx: idx });
                        this.currentText = null;
                    }
                };

                const audioContext = new AudioContext({ sampleRate: 16000 });
                await audioContext.audioWorklet.addModule('./audio_process.js');

                const recordNode = new AudioWorkletNode(audioContext, 'record-audio-processor');
                recordNode.connect(audioContext.destination);
                
                recordNode.port.onmessage = (event) => {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(event.data.data.buffer);
                    }
                };

                const source = audioContext.createMediaStreamSource(mediaStream);
                source.connect(recordNode);
                
                this.asrWS = ws;
                this.recording = true;
            } catch (err) {
                console.error("录音启动失败:", err);
                alert("请检查麦克风权限");
            }
        },

        async stopasr() {
            if (this.asrWS) {
                this.asrWS.close();
                this.asrWS = null;
            }
            this.recording = false;
            this.currentText = null;
        },

        // --- 5. 离线 ASR (文件上传识别) 逻辑 ---
        async uploadFile() {
            if (!this.file) {
                alert('请先选择文件');
                return;
            }

            const formData = new FormData();
            formData.append('file', this.file);
            
            // 重置状态
            this.fileResults = [];
            this.fileElapsed = null;

            try {
                const endpoint = this.useInt8 ? '/asr_file?use_int8=true' : '/asr_file';
                const response = await fetch(endpoint, { method: 'POST', body: formData });

                if (!response.ok) throw new Error(`HTTP 错误: ${response.status}`);

                const result = await response.json();
                this.fileResults = (result.segments || []).sort((a, b) => a.start - b.start);
                this.fileElapsed = result.elapsed;
                this.fileSize = result.data_length;
                this.fileAudioDuration = result.audio_duration;
                this.fileRtf = result.rtf;
                this.fileModel = result.asr_model;
            } catch (error) {
                console.error('上传失败:', error);
                alert('识别失败: ' + error.message);
            }
        }
    }));
});