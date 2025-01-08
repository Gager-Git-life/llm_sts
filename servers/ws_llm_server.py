import asyncio
import json

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.llm_core import QWenChat, ClaudeChat, OpenAIChat
from utils.config import LLM_CONFIG, MLLM_CONFIG
from core.websocket_core import WebSocketServer
from utils.logger import logger

class LLMServer(WebSocketServer):

    def __init__(self,
        llm_type: str=LLM_CONFIG["llm_type"],
        stream: bool=LLM_CONFIG["stream"],
        context: bool=LLM_CONFIG["context"],
        host: str=LLM_CONFIG["host"], 
        port: int=LLM_CONFIG["port"]
    ):
        super().__init__(host, port)
        self.llm_type = llm_type
        self.stream = stream
        self.context = context
        self.init()


    def init(self, ):
        self.base_llm_instance = self.create_llm_instance(
            self.llm_type, self.stream, self.context
        )
        self.llm_instances = {}

    def create_llm_instance(self, llm_type: str, stream: bool = None, context: bool = None):
        if llm_type not in MLLM_CONFIG:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
            
        config = MLLM_CONFIG[llm_type]
        if llm_type == "qwen":
            llm_instance = QWenChat(api_key=config["api_key"], 
                          base_url=config["base_url"], 
                          model_name=config["model_name"]
            )
        elif llm_type == "claude":
            llm_instance = ClaudeChat(api_key=config["api_key"], 
                            base_url=config["base_url"], 
                            model_name=config["model_name"]
            )
        elif llm_type == "openai":
            llm_instance = OpenAIChat(api_key=config["api_key"], 
                            base_url=config["base_url"], 
                            model_name=config["model_name"]
            )
        llm_instance.set_chat_options(stream=stream, context=context)
        return llm_instance

    async def data_process(self, websocket, data):

        try:
            if websocket not in self.llm_instances:
                llm_instance = self.create_llm_instance(
                    self.llm_type, self.stream, self.context
                )
                self.llm_instances[websocket] = llm_instance
            else:
                llm_instance = self.llm_instances[websocket]
            logger.info(len(self.llm_instances))
            llm_instance = self.llm_instances[websocket]
            
            # llm_instance = self.base_llm_instance
            message = json.loads(data)

            result = llm_instance.chat(
                model=message["model"],
                input_text=message["content"],
                input_image=message.get("image_path")
            )
            
            # 如果结果是迭代器，流式发送每个块
            logger.info("llm: ", end="")
            for chunk in result:
                logger.info(chunk, end="", flush=True)
                await self.send_queues[websocket].put(chunk)
                await asyncio.sleep(0.001)
            logger.info("\n", end="")


        except Exception as e:
            logger.error({"type": "error", "message": f"Error in response generation: {str(e)}"})
     

async def main():
    server = LLMServer()
    await server.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass 