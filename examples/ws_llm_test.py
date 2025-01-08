import asyncio
import json

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.websocket_core import WebSocketClient
from utils.logger import logger

async def send_to_llm_server(client, model_name, message, image_path=None):
    try:
        await client.send_message(json.dumps({
                "model": model_name,
                "content": message,
                "image_path": image_path
            }))
        await asyncio.sleep(0.001)

    except Exception as e:
        logger.error(f"Error sending message: {str(e)}")
        raise

async def get_from_llm_server(client):
    try:
        while True:
            response = await client.get_message()
            yield response

    except Exception as e:
        logger.error(f"Error get_from_llm_server: {str(e)}")
        raise  # 在发生错误时退出循环
    

async def llm_chat(model_name, message, image_path=None):
    """运行单个客户端，并作为异步迭代器返回转写结果"""
    client = WebSocketClient("ws://localhost:8764")

    try:
        # 启动客户端核心任务并等待连接建立
        client_task = asyncio.create_task(client.run())
        # 给予一些时间让连接建立
        await asyncio.sleep(0.5)
            
        # 发送消息
        await send_to_llm_server(client, model_name, message, image_path)
        
        # 直接异步迭代转写结果
        async for response in get_from_llm_server(client):
            yield response
        
    except KeyboardInterrupt:
        logger.warning("Client shutting down...")
    except Exception as e:
        logger.error(f"Client error: {str(e)}")
    finally:
        # 确保在结束时关闭客户端
        await client.close()

async def main():
    """主函数"""

    model_name = "qwen-long"
    message = "你好, 请你你能做些什么？"
    image_path = None

    try:
        async for response in llm_chat(model_name, message, image_path):
            logger.info(response, end="\n", flush=True)
    except KeyboardInterrupt:
        logger.warning("Shutting down...")
    except Exception as e:
        logger.error(f"Main error: {str(e)}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass 