"""
配置文件
"""

# LLM配置
MLLM_CONFIG = {
    "qwen": {
        "api_key": "sk-822118097f144d3b8492d09901b80d2a",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_name": "qwen-long"
    },
    "claude": {
        "api_key": "your_claude_api_key",
        "base_url": "your_claude_base_url",
        "model_name": "claude-3"
    },
    "openai": {
        "api_key": "your_openai_api_key",
        "base_url": "your_openai_base_url",
        "model_name": "gpt-4"
    }
}


ASR_CONFIG = {
    "host": "0.0.0.0",
    "port": 8765,
    "sample_rate": 16000,
    "model_path": "./vosk-model-cn-0.22"
}

LLM_CONFIG = {
    "host": "0.0.0.0",
    "port": 8764,
    "llm_type": "qwen",
    "stream": True,
    "context": True
}

TTS_CONFIG = {
    "host": "0.0.0.0",
    "port": 8763,
    "voice": "zh-CN-XiaoxiaoNeural",
    "channel": 1
}
