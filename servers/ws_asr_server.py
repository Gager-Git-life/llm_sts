from vosk import Model, KaldiRecognizer
import asyncio
import json

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.websocket_core import WebSocketServer
from utils.config import ASR_CONFIG
from utils.logger import logger

class AudioInputServer(WebSocketServer):

    def __init__(self, 
        model_path: str=ASR_CONFIG["model_path"],
        audio_sample_rate: int=ASR_CONFIG["sample_rate"],
        host: str=ASR_CONFIG["host"], 
        port: int=ASR_CONFIG["port"]
    ):
        super().__init__(host, port)
        self.audio_sample_rate = audio_sample_rate
        self.model_path = model_path
        self.init()

    def init(self):
        """初始化语音识别模型"""
        try:
            model = Model(self.model_path)
            self.rec = KaldiRecognizer(model, self.audio_sample_rate)
            self.rec.SetWords(True)
        except Exception as e:
            logger.error(f"Failed to initialize Vosk model: {str(e)}")
            raise

    async def data_process(self, websocket, audio_data):
        """处理音频数据"""
        try:
            # logger.info(f"Received audio data of type: {type(audio_data)}")
            if isinstance(audio_data, bytes):
                if self.rec.AcceptWaveform(audio_data):
                    result = json.loads(self.rec.Result())
                    text = result.get("text", "")
                else:
                    partial = json.loads(self.rec.PartialResult())
                    text = partial.get("partial", "")
                if text:
                    logger.info(f"Received audio data: {text}")
                #     await self.send_queues[websocket].put(text)

            elif audio_data == "[end]":
                result = json.loads(self.rec.FinalResult())
                text = result.get("text", "")
                if text: 
                    logger.info(f"Final audio data: {text}")
                    await self.send_queues[websocket].put(text)
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