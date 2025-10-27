## Run local chat bots

> [!NOTE]
>
> - run src code, replace achatbot to src, don't need set `ACHATBOT_PKG=1` e.g.:
>
>   ```
>   TQDM_DISABLE=True \
>        python -m src.cmd.local-terminal-chat.generate_audio2audio > log/std_out.log
>   ```
>
> - PyAudio need install python3-pyaudio 
>   e.g. ubuntu `apt-get install python3-pyaudio`, macos `brew install portaudio`
>   see: https://pypi.org/project/PyAudio/
>
> - llm llama-cpp-python init use cpu Pre-built Wheel to install, 
>   if want to use other lib(cuda), see: https://github.com/abetlen/llama-cpp-python#installation-configuration
>
> - install `pydub`  need install `ffmpeg` see: https://www.ffmpeg.org/download.html

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
   wget https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt -O ~/.achatbot/models/base.pt
   
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
   huggingface-cli download 2Noise/ChatTTS  --local-dir ~/.achatbot/models/2Noise/ChatTTS --local-dir-use-symlinks False
   
   # tts coquiTTS
   huggingface-cli download coqui/XTTS-v2  --local-dir ~/.achatbot/models/coqui/XTTS-v2 --local-dir-use-symlinks False
   
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

   