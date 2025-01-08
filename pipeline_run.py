import asyncio
from typing import Optional
from core.websocket_core import WebSocketClient
from core.audio_core import AudioStreamProcessor, AudioPlayer
from core.text_core import TextSegmenter
from utils.logger import logger
import json

class PipelineManager:
    def __init__(self):
        # 初始化队列
        self.audio_input_queue = asyncio.Queue()    # 原始音频输入队列
        self.asr_output_queue = asyncio.Queue()     # ASR结果队列
        self.llm_input_queue = asyncio.Queue()      # LLM输入队列
        self.llm_output_queue = asyncio.Queue()     # LLM结果队列
        self.tts_input_queue = asyncio.Queue()      # TTS输入队列
        self.tts_output_queue = asyncio.Queue()     # TTS结果队列
        self.echo_reference_queue = asyncio.Queue()  # 回声参考音频队列
        
        # 初始化处理器
        self.audio_processor = AudioStreamProcessor()
        self.audio_player = AudioPlayer()
        self.text_segmenter = TextSegmenter()
        
        # 初始化WebSocket客户端
        self.asr_client = WebSocketClient("ws://localhost:8765")
        self.llm_client = WebSocketClient("ws://localhost:8764")
        self.tts_client = WebSocketClient("ws://localhost:8763")
        
        self.running = True

    async def start(self):
        """启动整个pipeline"""
        try:
            # 启动WebSocket连接
            tasks = [
                self.asr_client.run(),
                self.llm_client.run(),
                self.tts_client.run()
            ]
            
            # 启动各个处理模块
            tasks.extend([
                # 音频输入处理
                self.audio_input_processor(),
                # ASR模块
                self.asr_sender(),
                self.asr_receiver(),
                # LLM模块
                self.llm_sender(),
                self.llm_receiver(),
                # TTS模块
                self.tts_sender(),
                self.tts_receiver(),
                # 音频输出
                self.audio_output_processor()
            ])
            
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
        finally:
            await self.cleanup()

    async def audio_input_processor(self):
        """处理音频输入"""
        try:
            async for chunk in self.audio_processor.process_stream_by_webrtc():
                await self.audio_input_queue.put(chunk)
        except Exception as e:
            logger.error(f"Audio input processor error: {str(e)}")

    async def asr_sender(self):
        """发送音频数据到ASR服务器"""
        try:
            while self.running:
                chunk = await self.audio_input_queue.get()
                await self.asr_client.send_message(chunk)
        except Exception as e:
            logger.error(f"ASR sender error: {str(e)}")

    async def asr_receiver(self):
        """接收ASR服务器的转写结果并触发打断"""
        try:
            while self.running:
                text = await self.asr_client.get_message()
                if text:
                    # 发送新的文本到LLM输入队列
                    await self.llm_input_queue.put(text)
        except Exception as e:
            logger.error(f"ASR receiver error: {str(e)}")

    async def llm_sender(self):
        """发送文本到LLM服务器"""
        try:
            while self.running:
                data = await self.llm_input_queue.get()
     
                message = json.dumps({
                    "model": "qwen-long",
                    "content": data,
                    "image_path": None
                })
                await self.llm_client.send_message(message)
        except Exception as e:
            logger.error(f"LLM sender error: {str(e)}")

    async def llm_receiver(self):
        """接收LLM服务器的响应"""
        try:
            while self.running:
                response = await self.llm_client.get_message()
                if response:
                    try:
                        await self.tts_input_queue.put(response)
                    except json.JSONDecodeError:
                        logger.error("Invalid JSON response from LLM")
        except Exception as e:
            logger.error(f"LLM receiver error: {str(e)}")

    async def tts_sender(self):
        """发送文本到TTS服务器"""
        try:
            while self.running:
                result = await self.tts_input_queue.get()
                async for segment in self.text_segmenter.process_text(result):
                    await self.tts_client.send_message(segment)
                    await asyncio.sleep(0.05)
                
        except Exception as e:
            logger.error(f"TTS sender error: {str(e)}")

    async def tts_receiver(self):
        """接收TTS服务器的音频数据"""
        try:
            while self.running:
                response = await self.tts_client.get_message()
                if response and len(response) > 0:
                    await self.tts_output_queue.put(response)
                    await self.echo_reference_queue.put(response)
        except Exception as e:
            logger.error(f"TTS receiver error: {str(e)}")

    async def audio_output_processor(self):
        """处理音频输出"""
        try:
            while self.running:
                audio_data = await self.tts_output_queue.get()
                self.audio_player.process_chunk(audio_data)
        except Exception as e:
            logger.error(f"Audio output processor error: {str(e)}")

    async def cleanup(self):
        """清理资源"""
        self.running = False
        self.audio_player.close()
        await self.asr_client.close()
        await self.llm_client.close()
        await self.tts_client.close() 

if __name__ == "__main__":
    # 创建pipeline管理器实例
    pipeline = PipelineManager()
    
    # 运行事件循环
    try:
        asyncio.run(pipeline.start())
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}") 