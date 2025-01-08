from queue import Queue, Empty
import threading
import numpy as np
import webrtcvad
import pyaudio
import asyncio
import torch
import time
import queue
import io
from pydub import AudioSegment
import wave
import sounddevice as sd
from io import BytesIO

class VADIterator:
    def __init__(self,
                 model,
                 threshold: float = 0.5,
                 sampling_rate: int = 16000,
                 min_silence_duration_ms: int = 500,  # makes sense on one recording that I checked
                 speech_pad_ms: int = 100             # same 
                 ):

        """
        Class for stream imitation

        Parameters
        ----------
        model: preloaded .jit silero VAD model

        threshold: float (default - 0.5)
            Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
            It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.

        sampling_rate: int (default - 16000)
            Currently silero VAD models support 8000 and 16000 sample rates

        min_silence_duration_ms: int (default - 100 milliseconds)
            In the end of each speech chunk wait for min_silence_duration_ms before separating it

        speech_pad_ms: int (default - 30 milliseconds)
            Final speech chunks are padded by speech_pad_ms each side
        """

        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate

        if sampling_rate not in [8000, 16000]:
            raise ValueError('VADIterator does not support sampling rates other than [8000, 16000]')

        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        self.reset_states()

    def reset_states(self):

        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    def __call__(self, x, return_seconds=False):
        """
        x: torch.Tensor
            audio chunk (see examples in repo)

        return_seconds: bool (default - False)
            whether return timestamps in seconds (default - samples)
        """

        if not torch.is_tensor(x):
            try:
                x = torch.Tensor(x)
            except:
                raise TypeError("Audio cannot be casted to tensor. Cast it manually")

        window_size_samples = len(x[0]) if x.dim() == 2 else len(x)
        self.current_sample += window_size_samples

        speech_prob = self.model(x, self.sampling_rate).item()

        if (speech_prob >= self.threshold) and self.temp_end:
            self.temp_end = 0

        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            speech_start = self.current_sample - self.speech_pad_samples
            return {'start': int(speech_start) if not return_seconds else round(speech_start / self.sampling_rate, 1)}

        if (speech_prob < self.threshold - 0.15) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            else:
                speech_end = self.temp_end + self.speech_pad_samples
                self.temp_end = 0
                self.triggered = False
                return {'end': int(speech_end) if not return_seconds else round(speech_end / self.sampling_rate, 1)}

        return None


class FixedVADIterator(VADIterator):
    '''It fixes VADIterator by allowing to process any audio length, not only exactly 512 frames at once.
    If audio to be processed at once is long and multiple voiced segments detected, 
    then __call__ returns the start of the first segment, and end (or middle, which means no end) of the last segment. 
    '''

    def reset_states(self):
        super().reset_states()
        self.buffer = np.array([],dtype=np.float32)

    def __call__(self, x, return_seconds=False):
        self.buffer = np.append(self.buffer, x) 
        ret = None
        while len(self.buffer) >= 512:
            r = super().__call__(self.buffer[:512], return_seconds=return_seconds)
            self.buffer = self.buffer[512:]
            if ret is None:
                ret = r
            elif r is not None:
                if 'end' in r:
                    ret['end'] = r['end']  # the latter end
                if 'start' in r and 'end' in ret:  # there is an earlier start.
                    # Remove end, merging this segment with the previous one.
                    del ret['end']
        return ret if ret != {} else None

class AudioStream:
    def __init__(self, input_type="system"):
        self.CHUNK = 480  # 30ms at 16kHz
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.recording = True
        self.audio_chunks = []
        self.input_type = input_type
        
    async def record_audio(self):
        if self.input_type == "microphone":
            async for data in self._record_from_microphone():
                yield data
        elif self.input_type == "system":
            async for data in self._record_from_system():
                yield data
                
    async def _record_from_microphone(self):
        p = pyaudio.PyAudio()
        stream = p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        print("* Recording from microphone...")
        
        try:
            while self.recording:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                yield data
                await asyncio.sleep(0.001)
                    
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    async def _record_from_system(self):
        print("* Recording system audio...")
        
        # 创建一个队列来存储音频数据
        audio_queue = asyncio.Queue()
        
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"音频回调状态: {status}")
            # 将音频数据放入队列
            audio_data = (indata * 32767).astype(np.int16).tobytes()
            try:
                audio_queue.put_nowait(audio_data)
            except asyncio.QueueFull:
                print("音频队列已满，丢弃数据")

        # 查找 BlackHole 设备
        devices = sd.query_devices()
        blackhole_device = None
        
        print("\nAvailable audio devices:")
        for i, device in enumerate(devices):
            print(f"{i}: {device['name']} (in={device['max_input_channels']}, out={device['max_output_channels']})")
            if "BlackHole" in device['name'] and device['max_input_channels'] > 0:
                blackhole_device = i
        
        if blackhole_device is None:
            raise RuntimeError("BlackHole device not found. Please install BlackHole and set it as your system output.")
        
        print(f"\nUsing BlackHole device: {devices[blackhole_device]['name']}")
        
        # 创建音频流
        stream = sd.InputStream(
            device=blackhole_device,
            channels=self.CHANNELS,
            samplerate=self.RATE,
            dtype=np.int16,
            blocksize=self.CHUNK,
            callback=audio_callback,
            latency='low'
        )
        
        with stream:
            print("开始录制...")
            try:
                while self.recording:
                    try:
                        # 从队列中获取音频数据
                        audio_data = await audio_queue.get()
                        yield audio_data
                        await asyncio.sleep(0.001)
                    except asyncio.QueueEmpty:
                        continue
                    except Exception as e:
                        print(f"音频处理错误: {e}")
                    
            except Exception as e:
                print(f"录制系统音频时出错: {e}")
            finally:
                print("\n录制结束。")

class AudioPlayer:
    def __init__(self, buffer_size: int = 100, low_watermark: float = 0.01):
        """
        初始化音频播放器
        
        Args:
            buffer_size: 缓冲区大小（块数）
            low_watermark: 低水位线阈值（占总容量的比例）
        """
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = self.pyaudio_instance.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            output=True
        )
        
        self.buffer = Queue(maxsize=buffer_size)
        self.buffer_size = buffer_size
        self.low_watermark = int(buffer_size * low_watermark)
        self.is_playing = False
        self.play_thread = None
        self.current_channel = None  # 添加通道标记
        self._start_playback_thread()

    def clean_buffer(self):
        """清空缓冲区"""
        while not self.buffer.empty():
            try:
                self.buffer.get_nowait()
            except Empty:
                break
    
    def _start_playback_thread(self):
        """启动后台播放线程"""
        self.is_playing = True
        self.play_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.play_thread.start()
    
    async def _buffer_consumer(self):
        """异步消费缓冲区数据"""
        while self.is_playing:
            try:
                # 等待缓冲区达到低水位线
                while self.buffer.qsize() < self.low_watermark:
                    await asyncio.sleep(0.001)
                
                # 批量获取数据
                chunks = []
                while not self.buffer.empty():
                    try:
                        chunk = self.buffer.get_nowait()
                        chunks.append(chunk)
                    except Empty:
                        break
                
                if chunks:
                    audio_data = b''.join(chunks)
                    # 异步播放
                    await self._async_play(audio_data)
                    
            except Exception as e:
                print(f"播放错误: {e}")
                await asyncio.sleep(0.1)

    async def _async_play(self, audio_data: bytes):
        """异步播放音频"""
        try:
            # 尝试作为MP3播放
            try:
                audio_segment = AudioSegment.from_mp3(BytesIO(audio_data))
                self.stream.write(audio_segment.raw_data)
            except:
                # 直接播放原始数据
                self.stream.write(audio_data)
        except Exception as e:
            print(f"播放错误: {e}")

    def _playback_loop(self):
        """启动异步播放循环"""
        asyncio.run(self._buffer_consumer())
    
    def process_chunk(self, chunk: bytes, is_ping: bool = None):
        """
        处理单个音频数据块，支持通道切换
        
        Args:
            chunk: 音频数据块
            is_ping: 当前是否为ping通道
        
        Returns:
            bool: 是否成功加入缓冲区
        """
        try:
            # 如果通道发生变化，清空缓冲区
            if is_ping is not None and self.current_channel != is_ping:
                self.clean_buffer()
                self.current_channel = is_ping
                
            self.buffer.put_nowait(chunk)
            return True
        except queue.Full:
            print("警告：音频缓冲区已满")
            return False
    
    def close(self):
        """关闭音频播放器"""
        self.is_playing = False
        if self.play_thread:
            self.play_thread.join(timeout=1)
        # 直接清空缓冲区
        while not self.buffer.empty():
            try:
                self.buffer.get_nowait()
            except Empty:
                break
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()

class AudioStreamProcessor:
    def __init__(self):
        self.audio_stream = AudioStream(input_type="microphone")
        self.frame_rate = self.audio_stream.RATE
        self.vad = webrtcvad.Vad(3)  # 使用最高的激进程度
        # VAD参数
        self.frame_duration = 30  # 每帧时长(ms)
        # 噪声处理
        self.noise_floor = None
        self.noise_adapt_rate = 0.95

    def _update_noise_floor(self, energy):
        """更新环境噪声基准"""
        if self.noise_floor is None:
            self.noise_floor = energy
        else:
            self.noise_floor = self.noise_floor * self.noise_adapt_rate + energy * (1 - self.noise_adapt_rate)

    async def process_stream_by_webrtc(self):
        """使用WebRTC VAD处理音频流"""
        # 初始化状态
        is_speaking = False  # 当前是否在说话
        last_speech_time = 0  # 语音开始时间
        
        # 阈值设置
        energy_threshold = 0.01  # 能量阈值
        snr_threshold = 1.5  # 信噪比阈值
        min_speech_duration = 0.5  # 最小语音持续时间（秒）

        async for audio_data in self.audio_stream.record_audio():
            current_time = time.time()
            # 1. 基础音频分析
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            audio_energy = np.sqrt(np.mean(np.square(audio_array / 32768.0)))
            
            # 2. 更新噪声基准
            self._update_noise_floor(audio_energy)
            
            # 3. 计算信噪比
            snr = audio_energy / (self.noise_floor + 1e-10)
            
            # 4. VAD检测
            is_speech = self.vad.is_speech(audio_data, self.frame_rate)
            
            # 5. 综合判断（VAD结果 + 能量阈值 + 信噪比）
            is_valid_speech = is_speech and (audio_energy > energy_threshold) and (snr > snr_threshold)

            # 6. 状态更新和输出
            if is_valid_speech:
                if not is_speaking:
                    is_speaking = True
                last_speech_time = current_time
                # print("*", end="", flush=True)
                yield audio_data
            else:
                if is_speaking:
                    speech_duration = current_time - last_speech_time
                    if speech_duration >= min_speech_duration:
                        is_speaking = False
                        # print("[end]", end="", flush=True)
                        yield "[end]"
                    else:
                        # 未达到最小录音时长，继续录音
                        # print("*", end="", flush=True)
                        yield audio_data
                else:
                    # print(".", end="", flush=True)
                    yield "[pass]"

    async def process_stream_by_silero(self):
        # 状态管理
        speech_started = False
        last_speech_time = None
        
        async for audio_data in self.audio_stream.record_audio():
            current_time = time.time()
            # 将音频数据转换为张量
            audio_tensor = torch.frombuffer(audio_data, dtype=torch.int16).float() / 32768.0
            
            # 使用 VAD 进行检测
            vad_result = self.vad(audio_tensor, self.audio_stream.RATE)
            
            if vad_result:
                if 'start' in vad_result and not speech_started:
                    speech_started = True
                    start = int(vad_result['start'] * self.frame_rate)
                    yield audio_data[start:]
                if "end" in vad_result and speech_started:
                    speech_started = False
                    end = int(vad_result['end'] * self.frame_rate)
                    last_speech_time = current_time
                    yield audio_data[:end]
            else:
                if last_speech_time and current_time - last_speech_time > self.silence_threshold:
                    last_speech_time = False
                    yield "[end]"
                else:
                    yield "[pass]"

    async def get_stream_form_path(self, ):
        while True:
            with wave.open("./test_audios/test1.wav", "rb") as wf:
                data = wf.readframes(3200)
                if len(data) == 0:
                    yield "[end]"
                else:
                    yield data
            await asyncio.sleep(1)
                    
    async def process_stream(self):
        async for data in self.process_stream_by_webrtc():
            yield data



async def main():
    processor = AudioStreamProcessor()
    async for data in processor.process_stream():
        if isinstance(data, bytes):
            print("*", end=" ", flush=True)
        elif data == "[end]":
            print(data, end=" ", flush=True)
        else:
            print(".", end=" ", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
    # asyncio.run(test_system_audio())