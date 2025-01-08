from pydub import AudioSegment
from io import BytesIO
import pyaudio
import asyncio
import json

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.websocket_core import WebSocketClient
from core.audio_core import AudioStreamProcessor
from core.text_core import TextSegmenter
from core.audio_core import AudioPlayer
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
    """接收服务器返回的消息"""
    while True:
        try:
            message = await client.get_message()
            yield message
        except Exception as e:
            logger.error(f"Error get_transcription: {str(e)}")
            break  # 在发生错误时退出循环


async def send_to_llm_server(client, async_iterator, model_name, image_path=None):
    """发送文本数据到LLM服务器"""
    try:
        async for message in async_iterator:
            await client.send_message(json.dumps({
                    "model": model_name,
                    "content": message,
                    "image_path": image_path
                }))
            logger.info(f"User: {message}")
            await asyncio.sleep(0.001)

    except Exception as e:
        logger.error(f"Error sending message: {str(e)}")
        raise

async def get_from_llm_server(client):
    """接收LLM服务器返回的消息"""
    try:
        while True:
            response = await client.get_message()
            if response:
                yield response
    except Exception as e:
        logger.error(f"Error get_from_llm_server: {str(e)}")
        raise  # 在发生错误时退出循环

async def send_data_to_audio_server(client, async_iterator):
    """发送文本数据到音频服务器"""
    try:
        async for message in async_iterator:
            await client.send_message(message)
            await asyncio.sleep(0.001)
    except Exception as e:
        logger.error(f"Error send_to_audio_out_server: {str(e)}")

async def get_audio_from_audio_server(client):
    """接收音频服务器返回的音频数据"""
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


async def llm_sts_pipeline(audio_processor):
    """
    语音转写到LLM服务器的完整流水线处理
    Args:
        audio_processor: 音频处理器实例
    """
    try:
        # 初始化 WebSocket 客户端
        asr_client = WebSocketClient("ws://localhost:8765")  # ASR服务器
        llm_client = WebSocketClient("ws://localhost:8764")  # LLM服务器
        tts_client = WebSocketClient("ws://localhost:8763")  # 音频输出服务器

        # 启动所有客户端
        asr_client_task = asyncio.create_task(asr_client.run())
        llm_client_task = asyncio.create_task(llm_client.run())
        tts_client_task = asyncio.create_task(tts_client.run())
        
        # 启动音频流式处理任务
        asr_task = asyncio.create_task(send_stream_audio(asr_client, audio_processor))
        
        # 获取转写结果并发送到LLM服务器
        transcription_iterator = get_transcription(asr_client)
        llm_task = asyncio.create_task(send_to_llm_server(
            llm_client,
            transcription_iterator,
            model_name="qwen-long"
        ))

        # # 处理LLM服务器的响应
        # async for response in get_from_llm_server(llm_client):
        #     yield response

        # 处理LLM服务器的响应并发送到音频输出服务器
        segmenter = TextSegmenter()
        llm_response_iterator = segmenter.process(get_from_llm_server(llm_client))
        tts_task = asyncio.create_task(send_data_to_audio_server(
            tts_client,
            llm_response_iterator
        ))

        # 获取并处理音频服务器的响应
        async for audio_response in get_audio_from_audio_server(tts_client):
            yield audio_response
        
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        raise
    finally:
        # 关闭所有客户端连接
        await asr_client.close()
        await llm_client.close()
        await tts_client.close()

async def main():

    audio_processor = AudioStreamProcessor()
    audio_player = AudioPlayer()

    try:
        async for audio_response in llm_sts_pipeline(audio_processor):
            audio_player.process_chunk(audio_response)
            # await asyncio.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}")
    finally:
        # 确保资源被正确释放
        audio_player.close()
        logger.info("程序已退出")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass 
