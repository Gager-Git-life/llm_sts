from vosk import Model, KaldiRecognizer
import asyncio
import json

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.websocket_core import WebSocketServer
from core.audio_core import CAMPPlus_SV
from core.audio_core import Vosk_ASR
from utils.config import ASR_CONFIG
from utils.logger import logger

class AudioInputServer(WebSocketServer):

    def __init__(self, 
        asr_model_path: str=ASR_CONFIG["asr_model_path"],
        audio_sample_rate: int=ASR_CONFIG["sample_rate"],
        end_signal: str=ASR_CONFIG["end_signal"],

        speaker_model_path: str=ASR_CONFIG["speaker_model_path"],
        verification_audio_path: str=ASR_CONFIG["verification_audio_path"],
        feat_dim: int=ASR_CONFIG["feat_dim"],
        embedding_size: int=ASR_CONFIG["embedding_size"],

        host: str=ASR_CONFIG["host"], 
        port: int=ASR_CONFIG["port"]
    ):
        super().__init__(host, port)
        self.audio_sample_rate = audio_sample_rate
        self.asr_model_path = asr_model_path
        self.end_signal = end_signal

        self.speaker_model_path = speaker_model_path
        self.verification_audio_path = verification_audio_path
        self.feat_dim = feat_dim
        self.embedding_size = embedding_size

        self.audio_buffer = []
        self.is_speaking = False

        self.init()

    def init(self):
        """初始化ASR和说话人验证模型"""
        self.init_asr()
        self.init_speaker_verification()

    def init_asr(self):
        """初始化语音识别模型"""
        try:
            self.asr_model = Vosk_ASR(
                model_path=self.asr_model_path,
                sample_rate=self.audio_sample_rate,
                end_signal=self.end_signal
            )
        except Exception as e:
            logger.error(f"Failed to initialize Vosk model: {str(e)}")
            raise

    def init_speaker_verification(self):
        """初始化说话人验证模型"""
        self.sv_model = CAMPPlus_SV(
            model_path=self.speaker_model_path,
            feat_dim=self.feat_dim,
            embedding_size=self.embedding_size,
            sample_rate=self.audio_sample_rate,
            verification_audio_path=self.verification_audio_path
        )

    async def data_process(self, websocket, audio_data):
        """异步并行处理音频数据"""
        try:
            # 创建两个异步任务并行处理
            asr_task = asyncio.create_task(self.asr_model.process_audio_async(audio_data))
            sv_task = asyncio.create_task(self.sv_model.process_audio_async(audio_data))
            
            # 等待两个任务完成
            (asr_flag, text), (sv_flag, score) = await asyncio.gather(asr_task, sv_task)

            if asr_flag and text:
                if sv_flag:
                    logger.info(f"Final audio data: {text}, score: {score}")
                    await self.send_queues[websocket].put(text)
                else:
                    logger.warning(f"Final audio data: {text}, score: {score}")
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            return None

async def main():
    server = AudioInputServer()
    await server.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass 