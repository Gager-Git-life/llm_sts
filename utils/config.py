"""
配置文件
"""

# LLM配置
MLLM_CONFIG = {
    "qwen": {
        "api_key": "sk-5c669ac69ad2450d9dfcb430caf6233f",
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
    "asr_model_path": "./models/vosk-model-cn-0.22",
    "end_signal": "[end]",

    "speaker_model_path": "./models/campplus_cn_common.bin",
    "verification_audio_path": "./test_audios/gager.wav",
    "feat_dim": 80,
    "embedding_size": 192
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

SEARCH_CONFIG = {
    "host": "0.0.0.0",
    "port": 8762,
    
    "qa_host": "localhost",
    "qa_port": 8768,
    
    "api_key": "your-api-key",
    "base_url": "https://your-api-base-url",
    "model_name": "qwen-long"
}