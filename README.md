# chat-bot
[![PyPI](https://img.shields.io/pypi/v/achatbot)](https://pypi.org/project/achatbot/)

# Install
## pypi
```bash
python3 -m venv .venv_achatbot
source .venv_achatbot/bin/activate
pip install achatbot
# optional-dependencies e.g.
pip install "achatbot[fastapi_daily_bot_server]"
```

## local
```bash
git clone https://github.com/ai-bot-pro/chat-bot.git
cd chat-bot
python3 -m venv .venv_achatbot
source .venv_achatbot/bin/activate
bash scripts/pypi_achatbot.sh dev
# optional-dependencies e.g.
pip install "dist/achatbot-{$version}-py3-none-any.whl[fastapi_daily_bot_server]"
```

#  Run chat bots
## Run local chat bots
> [!NOTE]
> - run src code, replace achatbot to src, don't need set `ACHATBOT_PKG=1` e.g.:
>   ```
>   TQDM_DISABLE=True \
>        python -m achatbot.cmd.local-terminal-chat.generate_audio2audio > log/std_out.log
>    ```
> - PyAudio need install python3-pyaudio 
> e.g. ubuntu `apt-get install python3-pyaudio`, macos `brew install portaudio`
> see: https://pypi.org/project/PyAudio/
>
> - llm llama-cpp-python init use cpu Pre-built Wheel to install, 
> if want to use other lib(cuda), see: https://github.com/abetlen/llama-cpp-python#installation-configuration
>

1. run `pip install "achatbot[local_terminal_chat_bot]"` to install dependencies to run local terminal chat bot;
2. create achatbot data dir in `$HOME` dir `mkdir -p ~/.achatbot/{log,config,models,records,videos}`;
3. `cp .env.example .env`, and check `.env`, add key/value env params;
4. select a model ckpt to download:
    - vad model ckpt (default vad ckpt model use [silero vad](https://github.com/snakers4/silero-vad))
    ```
    # vad pyannote segmentation ckpt
    huggingface-cli download pyannote/segmentation-3.0  --local-dir ~/.achatbot/models/pyannote/segmentation-3.0 --local-dir-use-symlinks False
    ```
    - asr model ckpt (default whipser ckpt model use base size)
    ```
    # asr openai whisper ckpt
    wget https://openaipublic.azureedge.net/main/whisper/~/.achatbot/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt -O ~/.achatbot/models/base.pt

    # asr hf openai whisper ckpt for transformers pipeline to load
    huggingface-cli download openai/whisper-base  --local-dir ~/.achatbot/models/openai/whisper-base --local-dir-use-symlinks False

    # asr hf faster whisper (CTranslate2)
    huggingface-cli download Systran/faster-whisper-base  --local-dir ~/.achatbot/models/Systran/faster-whisper-base --local-dir-use-symlinks False

    # asr SenseVoice ckpt
    huggingface-cli download FunAudioLLM/SenseVoiceSmall  --local-dir ~/.achatbot/models/FunAudioLLM/SenseVoiceSmall --local-dir-use-symlinks False
    ```
    - llm model ckpt (default llamacpp ckpt(ggml) model use qwen-2 instruct 1.5B size)
    ```
    # llm llamacpp Qwen2-Instruct
    huggingface-cli download Qwen/Qwen2-1.5B-Instruct-GGUF qwen2-1_5b-instruct-q8_0.gguf  --local-dir ~/.achatbot/models --local-dir-use-symlinks False

    # llm llamacpp Qwen1.5-chat
    huggingface-cli download Qwen/Qwen1.5-7B-Chat-GGUF qwen1_5-7b-chat-q8_0.gguf  --local-dir ~/.achatbot/models --local-dir-use-symlinks False

    # llm llamacpp phi-3-mini-4k-instruct
    huggingface-cli download microsoft/Phi-3-mini-4k-instruct-gguf Phi-3-mini-4k-instruct-q4.gguf --local-dir ~/.achatbot/models --local-dir-use-symlinks False

    ```
    - tts model ckpt (default whipser ckpt model use base size)
    ```
    # tts chatTTS
    !huggingface-cli download 2Noise/ChatTTS  --local-dir ~/.achatbot/models/2Noise/ChatTTS --local-dir-use-symlinks False
    
    # tts coquiTTS
    !huggingface-cli download coqui/XTTS-v2  --local-dir ~/.achatbot/models/coqui/XTTS-v2 --local-dir-use-symlinks False
    
    # tts cosy voice
    git lfs install
    git clone https://www.modelscope.cn/iic/CosyVoice-300M.git ~/.achatbot/models/CosyVoice-300M
    git clone https://www.modelscope.cn/iic/CosyVoice-300M-SFT.git ~/.achatbot/models/CosyVoice-300M-SFT
    git clone https://www.modelscope.cn/iic/CosyVoice-300M-Instruct.git ~/.achatbot/models/CosyVoice-300M-Instruct
    #git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git ~/.achatbot/models/CosyVoice-ttsfrd
    
    ```

5. run local terminal chat bot with env; e.g. 
    - use dufault env params to run local chat bot
    ```
    ACHATBOT_PKG=1 TQDM_DISABLE=True \
        python -m achatbot.cmd.local-terminal-chat.generate_audio2audio > ~/.achatbot/log/std_out.log
    ```

## Run remote http fastapi daily chat bots
1. run `pip install "achatbot[fastapi_daily_bot_server]"` to install dependencies to run http fastapi daily chat bot; 

2. run below cmd to start http server, see api docs: http://0.0.0.0:4321/docs
    ```
    ACHATBOT_PKG=1 python -m achatbot.cmd.http.server.fastapi_daily_bot_serve
    ```
3. run chat bot processor, e.g. 
   - run a daily langchain rag bot api, with ui/educator-client
    > [!NOTE]
    > need process youtube audio save to local file with `pytube`, run `pip install "achatbot[pytube,deep_translator]"` to install dependencies
    > and transcribe/translate to text, then chunks to vector store, and run langchain rag bot api;
    > run data process: 
    > ```
    > ACHATBOT_PKG=1 python -m achatbot.cmd.bots.rag.data_process.youtube_audio_transcribe_to_tidb
    > ```
    > or download processed data from hf dataset [weege007/youtube_videos](https://huggingface.co/datasets/weege007/youtube_videos/tree/main/videos), then chunks to vector store .
   ```
   curl -XPOST "http://0.0.0.0:4321/bot_join/chat-bot/DailyLangchainRAGBot" \
    -H "Content-Type: application/json" \
    -d $'{"config":{"llm":{"model":"llama-3.1-70b-versatile","messages":[{"role":"system","content":""}],"language":"zh"},"tts":{"tag":"cartesia_tts_processor","args":{"voice_id":"eda5bbff-1ff1-4886-8ef1-4e69a77640a0","language":"zh"}},"asr":{"tag":"deepgram_asr_processor","args":{"language":"zh","model":"nova-2"}}}}' | jq .
   ```
   - run a daily rtvi chat bot api, with ui/rtvi-web-demo
   ```
   curl -XPOST "http://0.0.0.0:4321/bot_join/DailyRTVIBot" \
    -H "Content-Type: application/json" \
    -d $'{"config":{"llm":{"model":"llama-3.1-70b-versatile","messages":[{"role":"system","content":"You are ai assistant. Answer in 1-5 sentences. Be friendly, helpful and concise. Default to metric units when possible. Keep the conversation short and sweet. You only answer in raw text, no markdown format. Don\'t include links or any other extras"}]},"tts":{"voice":"2ee87190-8f84-4925-97da-e52547f9462c"}}}' | jq .

   curl -XPOST "http://0.0.0.0:4321/bot_join/DailyAsrRTVIBot" \
    -H "Content-Type: application/json" \
    -d $'{"config":{"llm":{"model":"llama-3.1-70b-versatile","messages":[{"role":"system","content":"你是一位很有帮助中文AI助理机器人。你的目标是用简洁的方式展示你的能力,请用中文简短回答，回答限制在1-5句话内。你的输出将转换为音频，所以不要在你的答案中包含特殊字符。以创造性和有帮助的方式回应用户说的话。"}]},"tts":{"voice":"2ee87190-8f84-4925-97da-e52547f9462c"}}}' | jq .
   ```

## Run remote rpc chat bot worker
1. run `pip install "achatbot[remote_rpc_chat_bot_be_worker]"` to install dependencies to run rpc chat bot BE worker; e.g. :
   - use dufault env params to run rpc chat bot BE worker
```
ACHATBOT_PKG=1 RUN_OP=be TQDM_DISABLE=True \
    TTS_TAG=tts_edge \
    python -m achatbot.cmd.grpc.terminal-chat.generate_audio2audio > ~/.achatbot/log/be_std_out.log
```
2. run `pip install "achatbot[remote_rpc_chat_bot_fe]"` to install dependencies to run rpc chat bot FE; 
```
ACHATBOT_PKG=1 RUN_OP=fe \
    TTS_TAG=tts_edge \
    python -m achatbot.cmd.grpc.terminal-chat.generate_audio2audio > ~/.achatbot/log/fe_std_out.log
```

## Run remote queue chat bot worker
1. run `pip install "achatbot[remote_queue_chat_bot_be_worker]"` to install dependencies to run queue chat bot worker; e.g.:
   - use default env params to run 
    ```
    ACHATBOT_PKG=1 REDIS_PASSWORD=$redis_pwd RUN_OP=be TQDM_DISABLE=True \
        python -m achatbot.cmd.remote-queue-chat.generate_audio2audio > ~/.achatbot/log/be_std_out.log
    ```
   - sense_voice (asr) -> qwen (llm) -> cosy_voice (tts)
   ```
    ACHATBOT_PKG=1 RUN_OP=be \
        TQDM_DISABLE=True \
        REDIS_PASSWORD=$redis_pwd \
        ASR_TAG=sense_voice_asr \
        ASR_LANG=zn \
        N_GPU_LAYERS=33 FLASH_ATTN=1 \
        LLM_MODEL_NAME=qwen \
        LLM_MODEL_PATH=./models/qwen1_5-7b-chat-q8_0.gguf \
        ASR_MODEL_NAME_OR_PATH=./models/FunAudioLLM/SenseVoiceSmall \
        TTS_TAG=tts_cosy_voice \
        python -m achatbot.cmd.remote-queue-chat.generate_audio2audio > ~/.achatbot/log/be_std_out.log
   ```
2. run `pip install "achatbot[remote_queue_chat_bot_fe]"` to install the required packages to run quueue chat bot frontend; e.g.:
   - use default env params to run 
    ```
    ACHATBOT_PKG=1 REDIS_PASSWORD=$redis_pwd RUN_OP=fe \
        python -m achatbot.cmd.remote-queue-chat.generate_audio2audio > ~/.achatbot/log/fe_std_out.log
    ```
   - with wake word
    ```
    ACHATBOT_PKG=1 REDIS_PASSWORD=$redis_pwd RUN_OP=fe \
        RECORDER_TAG=wakeword_rms_recorder \
        python -m achatbot.cmd.remote-queue-chat.generate_audio2audio > ~/.achatbot/log/fe_std_out.log
    ```

## Run remote grpc tts speaker bot
1. run `pip install "achatbot[remote_grpc_tts_server]"` to install dependencies to run grpc tts speaker bot server; 
```
ACHATBOT_PKG=1 python -m achatbot.cmd.grpc.speaker.server.serve
```
2. run `pip install "achatbot[remote_grpc_tts_client]"` to install dependencies to run grpc tts speaker bot client; 
```
ACHATBOT_PKG=1 TTS_TAG=tts_edge python -m achatbot.cmd.grpc.speaker.client
ACHATBOT_PKG=1 TTS_TAG=tts_g IS_RELOAD=1 python -m achatbot.cmd.grpc.speaker.client
ACHATBOT_PKG=1 TTS_TAG=tts_coqui IS_RELOAD=1 python -m achatbot.cmd.grpc.speaker.client
ACHATBOT_PKG=1 TTS_TAG=tts_chat IS_RELOAD=1 python -m achatbot.cmd.grpc.speaker.client
ACHATBOT_PKG=1 TTS_TAG=tts_cosy_voice IS_RELOAD=1 python -m achatbot.cmd.grpc.speaker.client
```


# Project Structure
![project-structure](https://github.com/user-attachments/assets/5bf7cebb-e590-4718-a78a-6b0c0b36ea28)


## audio (voice)
- stream-stt (realtime-recorder)
![audio-text](https://github.com/user-attachments/assets/44bcec7d-f0a1-47db-bd95-21feee43a361)

- audio-llm (multimode-chat)
![pipe](https://github.com/user-attachments/assets/9970cf18-9bbc-4109-a3c5-e3e3c88086af)
![queue](https://github.com/user-attachments/assets/30f2e880-f16d-4b62-8668-61bb97c57b2b)


- stream-tts (realtime-(clone)-speaker)
![text-audio](https://github.com/user-attachments/assets/676230a0-0a99-475b-9ef5-6afc95f044d8)
![audio-text text-audio](https://github.com/user-attachments/assets/cbcabf98-731e-4887-9f37-649ec81e37a0)


## vision (CV)
- stream-ocr (realtime-object-dectection)

## more
- Embodied Intelligence: Robots that touch the world, perceive and move
