from typing import List, Dict, Generator, Union, Optional
from contextlib import contextmanager
from abc import ABC, abstractmethod
from openai import OpenAI
from PIL import Image
import numpy as np
import anthropic
import base64
import torch
import json
import PIL
import cv2
import io

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.prompt import LLM_PROMPT

class Pipe:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args, **kwargs):
        if hasattr(self.function, '__self__'):
            # 如果是绑定方法，我们需要传递 self
            return self.function.__func__(self.function.__self__, *args, **kwargs)
        else:
            # 如果是普通函数，直接调用
            return self.function(*args, **kwargs)

    def __ror__(self, other):
        return self(other)

    def __or__(self, other):
        return other(self())

class PipeableMethod:
    def __init__(self, method):
        self.method = method
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        @Pipe
        def piped(*args, **kwargs):
            return self.method(obj, *args, **kwargs)
        piped.original_method = lambda *args, **kwargs: self.method(obj, *args, **kwargs)
        return piped

def pipeable(method):
    return PipeableMethod(method)

class LLMChatBase(ABC):
    def __init__(self, api_key: str, base_url: str, model_name: str=None, system_message: str=None):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.default_system_message = LLM_PROMPT["default_system_prompt"]
        self.system_message = system_message or LLM_PROMPT["default_system_prompt"]
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.default_system_message}
        ]
        self.client = self.initialize_client()  # 在初始化时就创建客户端
        self.default_stream = False  # 默认流式设置
        self.default_context = True   # 默认上下文设置
        self.max_tokens = 2000 # 默认最大token数，受到了qwen的限制

        self.reasoning = False

    @abstractmethod
    def initialize_client(self):
        """初始化客户端"""
        pass

    def _encode_image(self, image: Union[str, Image.Image, np.ndarray, torch.Tensor]) -> str:
        """将不同格式的图像编码为base64字符串"""
        if isinstance(image, str):
            with open(image, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        elif isinstance(image, Image.Image):
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        elif isinstance(image, np.ndarray):
            _, buffer = cv2.imencode('.jpg', image)
            return base64.b64encode(buffer).decode('utf-8')
        elif isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
            _, buffer = cv2.imencode('.jpg', image_np)
            return base64.b64encode(buffer).decode('utf-8')
        else:
            raise ValueError("Unsupported image type")

    def reset_chat(self, system_prompt: str=None) -> None:
        self.messages = [
            {"role": "system", "content": system_prompt or self.default_system_message}
        ]

    def set_system_prompt(self, system_prompt: str) -> None:
        if system_prompt:
            self.system_message = system_prompt 
        self.messages = [
            {"role": "system", "content": self.system_message}]

        self.reset_chat(system_prompt)

    def set_chat_options(self, stream: bool=None, context: bool=None) -> None:
        """全局设置chat的流式和上下文选项"""
        if stream is not None:
            self.default_stream = stream
        if context is not None:
            self.default_context = context

    def add_message(self, role: str, content: str) -> None:
        """添加一条消息到对话历史"""
        self.messages.append({"role": role, "content": content})

    def set_reasoning(self, text: str = None) -> None:
        self.reasoning = True
        self.set_system_prompt(LLM_PROMPT["reasoning_system_prompt"])
        self.set_chat_options(stream=False, context=True)
        if text:
            self.add_message("user", text)

    @pipeable
    def chat_image(
        self, 
        input_image: Union[str, Image.Image, np.ndarray, torch.Tensor],
        input_text: str, 
        model: str,
        **llm_kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """与模型进行对话，支持图像输入、流式输出和历史上下文选项"""
        if not model:
            model = self.model_name

        image_base64 = self._encode_image(input_image)
        
        content = [
            {
                "type": "text",
                "text": input_text
            },

            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            }
        ] 

        return self.chat(content, None, model, **llm_kwargs)

    @pipeable
    def chat(self, input_text: str=None, input_image: str=None, model: str=None, **llm_kwargs) -> Union[str, Generator[str, None, None]]:
        """与模型进行对话,支持流式输出、历史上下文选项和多步骤推理"""
        try:
            if not model:
                model = self.model_name

            if input_image:
                image_base64 = self._encode_image(input_image)
                
                content = [
                    {
                        "type": "text",
                        "text": input_text
                    },

                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            else:
                content = input_text

            if self.default_context:
                self.add_message("user", content)
                messages = self.messages
            else:
                messages = [
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": content}
                ]

            llm_kwargs["max_tokens"] = min(llm_kwargs.get("max_tokens", self.max_tokens), self.max_tokens)

            if self.reasoning:
                return self._reasoning_chat(model=model, messages=messages, **llm_kwargs)
            elif self.default_stream:
                return self._stream_chat(model=model, messages=messages, context=self.default_context, **llm_kwargs)
            else:
                return self._non_stream_chat(model=model, messages=messages, context=self.default_context, **llm_kwargs)
        except Exception as e:
            print(f"Error in chat: {str(e)}")
            return "抱歉,发生了错误。请稍后再试。"

    def _reasoning_chat(self, model: str, messages: List[Dict[str, str]], **llm_kwargs) -> str:
        """执行多步骤推理对话"""
        messages = messages.copy()
        messages.append({"role": "assistant", "content": LLM_PROMPT["step_by_step_reasoning_prompt"]})

        while True:
            print("###"*10)
            response_json = self._chat_with_parse(model=model, messages=messages, **llm_kwargs)
            step_number = response_json.get('step_number', '-')
            step_title = response_json.get('title', 'Untitled')
            step_content = response_json.get('content', '')
            step_msg = f"Step {step_number}: {step_title}, {step_content}"
            print(step_msg)
            
            messages.append({"role": "assistant", "content": json.dumps(response_json, ensure_ascii=False)})
            if response_json.get("next_action", "continue") == "final_answer":
                break

        messages.append({"role": "user", "content": LLM_PROMPT["reasoning_final_answer_prompt"]})
        final_response = self._non_stream_chat(model=model, messages=messages, context=False, **llm_kwargs)
        self.add_message("assistant", final_response)
        return final_response
    
    def _chat_with_parse(self, model, messages: List[Dict[str, str]], **llm_kwargs) -> str:
        """与模型进行对话,并解析为json"""
        for i in range(3):
            response_str = self._non_stream_chat(model=model, messages=messages, context=False, **llm_kwargs)
            response_json = self._parse_json_response(response_str)
            if isinstance(response_json, dict):
                return response_json
            elif i==2:
                return {
                        "step_number": -1,
                        "title": "Error", 
                        "content": f"Failed to generate answer after 3 attempts",
                        "next_action": "final_answer"
                    }

    def _parse_json_response(self, text):
        try:
            # 尝试将整个字符串解析为JSON
            return json.loads(text)
        except json.JSONDecodeError:
            # 如果整个字符串不是有效的JSON,尝试查找第一个JSON部分
            start = text.find('{')
            end = text.find('}', start)
            if start != -1 and end != -1:
                json_str = text[start:end+1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    return text
            else:
                return text

        
    @contextmanager
    def _create_chat_stream(self, model, messages: List[Dict[str, str]], **llm_kwargs):
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            **llm_kwargs
        )
        try:
            yield response
        finally:
            response.close()

    def _stream_chat(self, model, messages: List[Dict[str, str]], context: bool, **llm_kwargs) -> Generator[str, None, None]:
        full_response = ""
        with self._create_chat_stream(model, messages, **llm_kwargs) as stream:
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content
        if context:
            self.add_message("assistant", full_response)
        return full_response

    def _non_stream_chat(self, model, messages: List[Dict[str, str]], context: bool, **llm_kwargs) -> str:
        # 创建基本参数字典
        params = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        # 更新参数字典，允许任意参数输入
        params.update(llm_kwargs)
        
        response = self.client.chat.completions.create(**params)
        assistant_response = response.choices[0].message.content
        if context:
            self.add_message("assistant", assistant_response)
        return assistant_response

class OpenAIChat(LLMChatBase):
    def initialize_client(self):
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

class DeepSeekChat(OpenAIChat):
    def __init__(self, api_key: str, base_url: str, model_name: str):
        super().__init__(api_key, base_url, model_name)
        self.max_tokens = 4000

class QWenChat(OpenAIChat):
    def __init__(self, api_key: str, base_url: str, model_name: str):
        super().__init__(api_key, base_url, model_name)
        self.max_tokens = 2000

class ClaudeChat(LLMChatBase):
    def __init__(self, api_key: str, base_url: str, model_name: str):
        super().__init__(api_key, base_url, model_name)
        self.max_tokens = 1024

    def initialize_client(self):
        return anthropic.Anthropic(
            base_url=self.base_url,
            api_key=self.api_key
        )

    @pipeable
    def chat_image(
        self, 
        input_image: Union[str, PIL.Image.Image, np.ndarray, torch.Tensor],
        input_text: str, 
        model: str,
        **llm_kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """与模型进行对话，支持图像输入、流式输出和历史上下文选项"""
        
        image_base64 = self._encode_image(input_image)
        
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_base64,
                },
            },
            {
                "type": "text",
                "text": input_text
            }
        ]

        return self.chat(model, content, None, **llm_kwargs)
        
    @pipeable
    def chat(self, input_text: str=None, input_image: str=None, model: str=None, **llm_kwargs) -> Union[str, Generator[str, None, None]]:
        """与模型进行对话，支持流式和历史上下文选项"""
        try:
            if not model:
                model = self.model_name
            if input_image:
                image_base64 = self._encode_image(input_image)
                
                content = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": input_text
                    }
                ]
            else:
                content = input_text

            if self.default_context:
                self.add_message("user", content)
                messages = self.messages
            else:
                messages = [
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": content}
                ]

            llm_kwargs["max_tokens"] = min(llm_kwargs.get("max_tokens", self.max_tokens), self.max_tokens)

            if self.default_stream:
                return self._stream_chat(model=model, messages=messages, **llm_kwargs)
            else:
                return self._non_stream_chat(model=model, messages=messages, **llm_kwargs)
        except Exception as e:
            print(f"Error in Claude chat: {str(e)}")
            return "抱歉，发生了错误。请稍后再试。"

    def _stream_chat(self, model, messages: List[Dict[str, str]], **llm_kwargs: dict) -> Generator[str, None, None]:
        with self.client.messages.stream(
            model=model,
            messages=messages,
            **llm_kwargs
        ) as stream:
            full_response = ""
            for chunk in stream:
                if hasattr(chunk, 'type') and chunk.type == "text":
                    text = chunk.text
                    full_response += text
                    yield text
            self.add_message("assistant", full_response)

        return full_response

    def _non_stream_chat(self, model, messages: List[Dict[str, str]], **llm_kwargs: dict) -> str:
        response = self.client.messages.create(
            model=model,
            messages=messages,
            **llm_kwargs
        )
        assistant_response = response.content[0].text
        self.add_message("assistant", assistant_response)
        return assistant_response
