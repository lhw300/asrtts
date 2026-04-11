document.addEventListener('alpine:init', () => {
    Alpine.data('demoapp', () => ({
        // --- 1. 变量定义 (确保 index.html 中的 x-model 都能找到对应变量) ---
        text: '您好，我是智能语音助手，请问有什么需要帮助的吗？',
        targetSid: 6,         // 默认说话人ID
        targetSpeed: 0.95,    // 默认语速
        targetRate: 8000,     // 默认采样率
        targetVolume: 1.0,    // 默认音量
        
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
            if (!this.text || !this.text.trim()) return;
            this.disabled = true;
            this.elapsedTime = null;

            try {
                const start = Date.now();
                const url = `${window.location.origin}/tts`;
                const resp = await fetch(url, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        text: this.text,
                        sid: parseInt(this.targetSid) || 0,
                        samplerate: parseInt(this.targetRate) || 8000,
                        speed: parseFloat(this.targetSpeed) || 1.0,
                        volume: parseFloat(this.targetVolume) || 1.0,
                    })
                });

                if (!resp.ok) {
                    console.error('TTS 请求失败:', resp.status);
                    this.disabled = false;
                    return;
                }

                const blob = await resp.blob();
                const audioUrl = URL.createObjectURL(blob);
                const audio = new Audio(audioUrl);

                const elapsed = ((Date.now() - start) / 1000).toFixed(2);
                this.elapsedTime = `${elapsed}s`;

                audio.play();
                audio.onended = () => {
                    URL.revokeObjectURL(audioUrl);
                    this.disabled = false;
                };
                audio.onerror = () => {
                    this.disabled = false;
                };

            } catch (err) {
                console.error('TTS 错误:', err);
                this.disabled = false;
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
				if (this.asrWS.readyState === WebSocket.OPEN) {
					this.asrWS.send('done');
				}
               // this.asrWS.close();
                this.asrWS = null;
            }
            this.recording = false;
           // this.currentText = null;
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