<div align="center">
<h1>LLM-STS 🎯</h1>

</div>

一个基于WebSocket和LLM的实时语音对话系统，集成了语音识别(ASR)、大语言模型(LLM)和语音合成(TTS)功能。

## 项目简介 🌟

本项目的灵感来源于OpenAI的实时语音对话演示，以及科幻电影《Her》中描绘的人工智能语音助手场景。我们希望打造一个流畅的实时语音对话系统，让用户能够通过自然的语音方式与AI进行交互。

项目特点：
- 🚀 全程实时响应：从语音输入到语音输出的全链路流式处理
- ⚡ 低延迟：基于WebSocket的流式传输，实现实时语音对话
- 🔌 模块化设计：支持灵活替换不同的语音识别、大语言模型和语音合成组件
- 🤖 多模型支持：支持多种主流大语言模型（QWen、Claude、OpenAI等）

## TODO List 📋

### ASR 模块 🎤
- ✅ 实时音频流输入输出
- ✅ WebRTC VAD 语音活动检测
- ✅ CAM++ 说话人确认
- 🚧 AEC 声学回声消除 (开发中)
- ✅ Vosk 中文语音识别
- ✅ WebSocket 流式传输

### LLM 模块 🤖
- ✅ 流式响应输出
- ✅ 上下文对话支持
- ✅ 多模型推理功能
- 📦 已支持模型:
  - ✅ 阿里千问 (Qwen)
  - ✅ Anthropic Claude
  - ✅ OpenAI GPT
  - 🚧 Google Gemini (开发中)

### TTS 模块 🔊

- ✅ Edge TTS 集成
- ✅ 多音色支持
- ✅ 实时语音合成
- ✅ WebSocket 流式传输
- 💡 更多 TTS 引擎支持 (计划中)

### 系统层级 🏠

- 🚧  Agent功能如：浏览器查询、工具使用（开发中）

- 🚧 实时打断对话（开发中，暂时没有十分优雅的实现，欢迎pr🤤）



## 使用方法 📝

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 模型下载 📥:

   2.1.vosk模型下载：

   - [vosk-model-cn-0.22](https://alphacephei.com/vosk/models/vosk-model-cn-0.22.zip)
   - [vosk-model-small-cn-0.22](https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip)
   - [其它模型](https://alphacephei.com/vosk/models)

   2.2.cam++模型下载：

   - [CAM++说话人确认-中文-通用-200k-Spkrs](https://www.modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common)

3. 配置服务 ⚙️：
   修改 `utils/config.py` 中的配置参数：

   **A. LLM配置参数** 🤖
   - `api_key`: 模型服务的API密钥
   - `base_url`: API服务的接口地址
   - `model_name`: 使用的模型版本名称（如：qwen-long）
   - `stream`: 流式文本输出（必须为True）
   - `context`: 上下文关联响应（默认为True，不建议修改）

   **B. ASR配置参数** 🎤
   - `host`: 服务器监听地址，0.0.0.0表示允许所有IP访问
   - `port`: ASR服务端口号，默认8765
   - `sample_rate`: 音频采样率，Vosk要求16kHz
   - `asr_model_path`: Vosk中文模型路径，指向下载的模型目录
   - `speaker_model_path`: CAM++模型路径，指向下载的模型目录
   - `verification_audio_path`: 说话人语音文件，需要自己单独录制一段存放

   **C. TTS配置参数** 🔊
   - `host`: 服务器监听地址
   - `port`: TTS服务端口号，默认8763
   - `voice`: Edge TTS音色选项，如"zh-CN-XiaoxiaoNeural"
   - `channel`: 音频声道数，1为单声道


4. 启动服务 🚀：
```bash
# 启动ASR服务器 (端口8765)
python servers/ws_asr_server.py

# 启动LLM服务器 (端口8764)
python servers/ws_llm_server.py

# 启动TTS服务器 (端口8763)
python servers/ws_tts_server.py

# 运行主程序
python main.py
```

## 系统架构 🏗️

项目包含以下主要组件：

1. **语音识别服务** (ASR) 👂

   - 采用流式传输音频数据

   - 基于VAD进行语音活动检测

   - 基于Vosk实现中文语音识别
   - 基于CAM++实现说话人确认
   - 支持实时同步语音识别和说话人识别
   - WebSocket服务器端口：8765

2. **大语言模型服务** (LLM)

   - 兼容openai格式接口

   - 支持多种模型：QWen、deepseek、gemini等
   - 支持逻辑推理（自定义prompt）
   - 支持流式输出和上下文对话
   - WebSocket服务器端口：8764

3. **语音合成服务** (TTS)
   - 基于Edge TTS实现
   - 支持多种音色
   - WebSocket服务器端口：8763

4. 语音输出模块

   - 在audio_core中实现了AudioPlayer类用于音频输出

## 核心模块

1. **音频处理模块** (`core/audio_core.py`)
   - 实现音频流采集和处理
   - 支持系统音频和麦克风输入
   - 集成VAD（语音活动检测）
   - 支持说话人确认，防止无限触发
   - 提供音频流处理和噪声处理功能
2. **文本处理模块** (`core/text_core.py`)
   - 实现文本分段处理
   - 优化语音合成的文本输入
3. **LLM模块** (`core/llm_core.py`)
   - 封装多种大语言模型接口
   - 统一的模型调用接口
   - 支持流式输出处理
4. **WebSocket核心模块** (`core/websocket_core.py`)
   - 提供WebSocket客户端和服务器基类
   - 实现异步消息处理
   - 管理连接生命周期

## 工作流程

1. **音频采集和处理**
   - 通过 `AudioStreamProcessor` 采集音频输入
   - 使用WebRTC VAD进行语音活动检测
   - 将有效语音片段发送至ASR服务器

2. **语音识别**
   - ASR服务器接收音频流
   - 使用Vosk模型进行实时语音识别
   - 输出识别文本流

3. **对话处理**
   - LLM服务器接收识别文本
   - 调用配置的语言模型进行处理
   - 生成对话响应流

4. **语音合成**
   - 对LLM响应文本进行分段处理
   - TTS服务器接收文本并转换为语音
   - 实时播放合成的语音响应

## 数据流

```
音频输入 -> VAD处理 -> ASR服务器 -> 文本识别
    -> LLM服务器 -> 对话生成
    -> 文本分段 -> TTS服务器 -> 语音合成 -> 音频输出
```

## 目录结构

```
llm_sts_plus/
├── core/               # 核心功能模块
├── examples/           # 示例代码
├── servers/            # 服务器实现
│   ├── ws_asr_server.py   # ASR服务器
│   ├── ws_llm_server.py   # LLM服务器
│   └── ws_tts_server.py   # TTS服务器
├── utils/              # 工具函数
├── run_pipeline.py     # 主程序入口
└── requirements.txt    # 依赖包列表
```

## 注意事项

1. 使用前请确保已安装所有依赖包
2. 需要配置相应的API密钥才能使用LLM服务
3. 语音识别需要下载对应的Vosk模型
4. 确保网络连接正常，以便访问各种API服务

## 示例代码

可以参考 `examples` 目录下的示例代码来了解各个模块的使用方法。

## 技术栈

- Python 3.8+
- WebSocket
- Edge TTS
- Vosk
- PyAudio
- asyncio