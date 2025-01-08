import asyncio

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.websocket_core import WebSocketClient
from utils.logger import logger

async def send_stream_audio(client, audio_processor):
    """流式读取并发送音频数据"""
    try:
        async for chunk in audio_processor.process_stream_by_webrtc():
            await client.send_message(chunk)
            await asyncio.sleep(0.001)
                
    except Exception as e:
        logger.error(f"Error streaming audio: {str(e)}")

async def get_transcription(client):
    """处理服务器返回的消息"""
    while True:
        try:
            message = await client.get_message()
            yield message
        except Exception as e:
            logger.error(f"Error get_transcription: {str(e)}")
            break  # 在发生错误时退出循环

async def audio_transcription(audio_processor):
    """运行单个客户端，并作为异步迭代器返回转写结果"""
    client = WebSocketClient("ws://localhost:8765")

    try:
        # 启动客户端核心任务
        client_task = asyncio.create_task(client.run())
            
        # 启动音频流任务并等待其开始运行
        audio_task = asyncio.create_task(send_stream_audio(client, audio_processor))
        
        # 直接异步迭代转写结果
        async for transcription in get_transcription(client):
            yield transcription
        
    except KeyboardInterrupt:
        logger.warning("Client shutting down...")
    except Exception as e:
        logger.error(f"Client error: {str(e)}")

async def main():
    """主函数"""
    
    from core.audio_core import AudioStreamProcessor
    audio_processor = AudioStreamProcessor()

    try:
        async for transcription in audio_transcription(audio_processor):
            logger.info(f"Transcription: {transcription}")
    except KeyboardInterrupt:
        logger.warning("Shutting down...")
    except Exception as e:
        logger.error(f"Main error: {str(e)}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass 