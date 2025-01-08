from pydub import AudioSegment
from io import BytesIO
import pyaudio
import asyncio

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.websocket_core import WebSocketClient
from utils.logger import logger

async def send_data_to_audio_server(client, message):
    """Connect to Service B and get response"""
    try:
        await client.send_message(message)
    except Exception as e:
        logger.error(f"Error send_to_audio_out_server: {str(e)}")

async def get_audio_from_audio_server(client):
    try:
        while True:
            response = await client.get_message()
            if response is None:
                continue
            if len(response) == 0:
                continue
            yield response
    except Exception as e:
        logger.error(f"Error get_from_audio_out_server: {str(e)}")
        raise

async def get_audio(message):
    """运行单个客户端，并作为异步迭代器返回转写结果"""
    client = WebSocketClient("ws://localhost:8763")

    try:
        # 启动客户端核心任务并等待连接建立
        client_task = asyncio.create_task(client.run())
        # 给予一些时间让连接建立
        await asyncio.sleep(0.5)
            
        # 发送消息
        await send_data_to_audio_server(client, message)
        
        # 直接异步迭代转写结果
        async for response in get_audio_from_audio_server(client):
            yield response
        
    except KeyboardInterrupt:
        logger.warning("Client shutting down...")
    except Exception as e:
        logger.error(f"Client error: {str(e)}")
    finally:
        # 确保在结束时关闭客户端
        await client.close()

def play_audio_chunks(chunks: list[bytes], stream: pyaudio.Stream) -> None:
    stream.write(AudioSegment.from_mp3(BytesIO(b''.join(chunks))).raw_data)

async def main():
    """主函数"""

    pyaudio_instance = pyaudio.PyAudio()
    audio_stream = pyaudio_instance.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)

    message = "希望在语音识别的学术研究和工业应用之间架起一座桥梁。通过发布工业级语音识别模型的训练和微调，研究人员和开发人员可以更方便地进行语音识别模型的研究和生产，并推动语音识别生态的发展。让语音识别更有趣！"

    try:
        audio_chunks = []
        async for response in get_audio(message):
            audio_chunks.append(response)
            if len(audio_chunks) >= 6:
                play_audio_chunks(audio_chunks, audio_stream)
                audio_chunks.clear()

    except KeyboardInterrupt:
        logger.warning("Shutting down...")
    except Exception as e:
        logger.error(f"Main error: {str(e)}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass 