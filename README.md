<div align="center">
<h1>LLM-STS 🎯</h1>

</div>

A real-time voice conversation system based on WebSocket and LLM, integrating speech recognition (ASR), large language model (LLM), and text-to-speech (TTS) functionalities.

## Project Introduction 🌟

This project is inspired by OpenAI's real-time voice conversation demo and the AI voice assistant scenario depicted in the sci-fi movie "Her". We aim to create a smooth real-time voice conversation system that allows users to interact with AI through natural voice.

Project Features:
- 🚀 Real-time response: Full-chain streaming processing from voice input to voice output
- ⚡ Low latency: Real-time voice conversation based on WebSocket streaming
- 🔌 Modular design: Supports flexible replacement of different speech recognition, large language model, and text-to-speech components
- 🤖 Multi-model support: Supports multiple mainstream large language models (QWen, Claude, OpenAI, etc.)

## TODO List 📋

### ASR Module 🎤
- ✅ Real-time audio stream input/output
- ✅ WebRTC VAD voice activity detection
- ✅ CAM++ speaker verification
- 🚧 AEC acoustic echo cancellation (in development)
- ✅ Vosk Chinese speech recognition
- ✅ WebSocket streaming

### LLM Module 🤖
- ✅ Streaming response output
- ✅ Contextual conversation support
- ✅ Multi-model inference functionality
- 📦 Supported models:
  - ✅ Alibaba Qwen
  - ✅ Anthropic Claude
  - ✅ OpenAI GPT
  - 🚧 Google Gemini (in development)

### TTS Module 🔊
- ✅ Edge TTS integration
- ✅ Multi-voice support
- ✅ Real-time speech synthesis
- ✅ WebSocket streaming
- 💡 More TTS engine support (planned)

### System Level 🏠
- 🚧 Agent functionalities like: browser query, tool usage (in development)
- 🚧 Real-time conversation interruption (in development, no elegant implementation yet, PRs welcome🤤)

## Usage 📝

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Model Download 📥:

    2.1 Vosk model download:

    - [vosk-model-cn-0.22](https://alphacephei.com/vosk/models/vosk-model-cn-0.22.zip)
    - [vosk-model-small-cn-0.22](https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip)
    - [Other models](https://alphacephei.com/vosk/models)

    2.2 CAM++ model download:

    - [CAM++ Speaker Verification - Chinese - General - 200k Speakers](https://www.modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common)

3. Configuration ⚙️：
    Modify the configuration parameters in `utils/config.py`:

    **A. LLM Configuration Parameters** 🤖
    - `api_key`: API key for model service
    - `base_url`: API service endpoint
    - `model_name`: Model version name (e.g., qwen-long)
    - `stream`: Streaming text output (must be True)
    - `context`: Contextual response (default is True, not recommended to modify)

    **B. ASR Configuration Parameters** 🎤
    - `host`: Server listening address, 0.0.0.0 allows all IP access
    - `port`: ASR service port, default 8765
    - `sample_rate`: Audio sample rate, Vosk requires 16kHz
    - `asr_model_path`: Vosk Chinese model path, pointing to the downloaded model directory
    - `speaker_model_path`: CAM++ model path, pointing to the downloaded model directory
    - `verification_audio_path`: Speaker voice file, need to record and store separately

    **C. TTS Configuration Parameters** 🔊
    - `host`: Server listening address
    - `port`: TTS service port, default 8763
    - `voice`: Edge TTS voice option, e.g., "zh-CN-XiaoxiaoNeural"
    - `channel`: Audio channel number, 1 for mono

4. Start Services 🚀：
```bash
# Start ASR server (port 8765)
python servers/ws_asr_server.py

# Start LLM server (port 8764)
python servers/ws_llm_server.py

# Start TTS server (port 8763)
python servers/ws_tts_server.py

# Run main program
python main.py
```

## System Architecture 🏗️

The project consists of the following main components:

1. **Speech Recognition Service** (ASR) 👂
   - Uses streaming audio data
   - Based on VAD for voice activity detection
   - Implements Chinese speech recognition using Vosk
   - Implements speaker verification using CAM++
   - Supports real-time synchronized speech recognition and speaker verification
   - WebSocket server port: 8765

2. **Large Language Model Service** (LLM)
   - Compatible with OpenAI format interface
   - Supports multiple models: QWen, deepseek, gemini, etc.
   - Supports logical reasoning (custom prompt)
   - Supports streaming output and contextual conversation
   - WebSocket server port: 8764

3. **Text-to-Speech Service** (TTS)
   - Implemented using Edge TTS
   - Supports multiple voices
   - WebSocket server port: 8763

4. Audio Output Module
   - Implements AudioPlayer class in audio_core for audio output

## Core Modules

1. **Audio Processing Module** (`core/audio_core.py`)
   - Implements audio stream capture and processing
   - Supports system audio and microphone input
   - Integrates VAD (Voice Activity Detection)
   - Supports speaker verification to prevent infinite triggering
   - Provides audio stream processing and noise handling

2. **Text Processing Module** (`core/text_core.py`)
   - Implements text segmentation
   - Optimizes text input for speech synthesis

3. **LLM Module** (`core/llm_core.py`)
   - Encapsulates multiple large language model interfaces
   - Unified model calling interface
   - Supports streaming output processing

4. **WebSocket Core Module** (`core/websocket_core.py`)
   - Provides WebSocket client and server base classes
   - Implements asynchronous message handling
   - Manages connection lifecycle

## Workflow

1. **Audio Capture and Processing**
   - Capture audio input through `AudioStreamProcessor`
   - Use WebRTC VAD for voice activity detection
   - Send valid voice segments to ASR server

2. **Speech Recognition**
   - ASR server receives audio stream
   - Use Vosk model for real-time speech recognition
   - Output recognized text stream

3. **Conversation Processing**
   - LLM server receives recognized text
   - Call configured language model for processing
   - Generate conversation response stream

4. **Speech Synthesis**
   - Segment LLM response text
   - TTS server receives text and converts to speech
   - Real-time playback of synthesized speech response

## Data Flow

```
Audio Input -> VAD Processing -> ASR Server -> Text Recognition
    -> LLM Server -> Conversation Generation
    -> Text Segmentation -> TTS Server -> Speech Synthesis -> Audio Output
```

## Directory Structure

```
llm_sts_plus/
├── core/               # Core functional modules
├── examples/           # Example code
├── servers/            # Server implementations
│   ├── ws_asr_server.py   # ASR server
│   ├── ws_llm_server.py   # LLM server
│   └── ws_tts_server.py   # TTS server
├── utils/              # Utility functions
├── run_pipeline.py     # Main program entry
└── requirements.txt    # Dependency list
```

## Notes

1. Make sure all dependencies are installed before use
2. Need to configure corresponding API keys to use LLM services
3. Speech recognition requires downloading corresponding Vosk models
4. Ensure network connection is normal to access various API services

## Example Code

Refer to the example code in the `examples` directory to understand how to use each module.

## Technology Stack

- Python 3.8+
- WebSocket
- Edge TTS
- Vosk
- PyAudio
- asyncio