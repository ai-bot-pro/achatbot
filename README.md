# chat-bot
[![PyPI](https://img.shields.io/pypi/v/achatbot)](https://pypi.org/project/achatbot/)

# install
```
python3 -m venv .venv_achatbot
source .venv_achatbot/bin/activate
pip install achatbot
```

# run bots
[!NOTE]
PyAudio need install python3-pyaudio 
 e.g. ubuntu `apt-get install python3-pyaudio`, macos `brew install portaudio`
see: https://pypi.org/project/PyAudio/

1. run `pip install "achatbot[local_terminal_chat_bot]"` to install dependencies to run local terminal chat bot;
2. create achatbot data dir in `$HOME` dir `mkdir -p ~/.achatbot/{log,config,models,records,videos}`
3. `cp .env.example .env`, and check `.env`, add key/value
4. select a model ckpt to download 
    - asr model ckpt (default whipser ckpt model use base size)
```
# asr openai whisper ckpt
wget https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt -O ./models/base.pt

# asr hf openai whisper ckpt
huggingface-cli download openai/whisper-base  --local-dir ./models/openai/whisper-base --local-dir-use-symlinks False

# asr hf faster whisper
huggingface-cli download Systran/faster-whisper-base  --local-dir ./models/Systran/faster-whisper-base --local-dir-use-symlinks False

# asr SenseVoice ckpt
!huggingface-cli download FunAudioLLM/SenseVoiceSmall  --local-dir ./models/FunAudioLLM/SenseVoiceSmall --local-dir-use-symlinks False

```
5. run local terminal chat bot with env; e.g. 
```
ACHATBOT_PKG=1 TQDM_DISABLE=True \
    python -m achatbot.cmd.local-terminal-chat.generate_audio2audio > ~/.achatbot/log/std_out.log
```

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

# more
- Embodied Intelligence: Robots that touch the world and move
