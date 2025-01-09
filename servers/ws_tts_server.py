import pyaudio
import asyncio
import edge_tts
import base64
import json

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.websocket_core import WebSocketServer
from utils.config import TTS_CONFIG
from utils.logger import logger

class AudioOutputServer(WebSocketServer):

    def __init__(self, 
        voice: str=TTS_CONFIG['voice'],
        host: str=TTS_CONFIG["host"], 
        port: int=TTS_CONFIG["port"]
    ):
        super().__init__(host, port)
        self.voice = voice

    async def data_process(self, websocket, data: str):
        """处理音频数据"""
        try:
            
            logger.info(f"text: {data}")
            communicate = edge_tts.Communicate(data, self.voice)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    # 直接发送音频数据块
                    await self.send_queues[websocket].put(chunk["data"])
                    await asyncio.sleep(0)

        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            return None

async def main():
    server = AudioOutputServer()
    await server.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass 