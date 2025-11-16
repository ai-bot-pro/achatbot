<img width="1050" height="289" alt="image" src="https://github.com/user-attachments/assets/163aaf65-2080-4aea-91e2-fb0e248a6ee9" />

# achatbot
[![PyPI](https://img.shields.io/pypi/v/achatbot)](https://pypi.org/project/achatbot/)

achatbot factory, create chat bots with llm(tools), asr, tts, vad, ocr, detect object etc..

- [achatbot-go](https://github.com/ai-bot-pro/achatbot-go) (main/sub agent)

# Design
## apipeline design
- ⭐️ [pipeline design](https://github.com/ai-bot-pro/pipeline-py#design) ⭐️
## achatbot design
- [project structure](https://github.com/ai-bot-pro/achatbot/blob/main/docs/project_structure.md)


# Install
> [!NOTE]
> `python --version` >=3.10 with [asyncio-task](https://docs.python.org/3.10/library/asyncio-task.html)
> if install `achatbot[tts_openvoicev2]` need install melo-tts `pip install git+https://github.com/myshell-ai/MeloTTS.git`
>
> if some other nested loop code with achatbot lib, you need to add the following code:
>
> ```python
> import nest_asyncio
> 
> nest_asyncio.apply()
> ```

> [!TIP]
> use [uv](https://github.com/astral-sh/uv) + pip to run, install the required dependencies fastly, e.g.:
> `uv pip install achatbot`
> `uv pip install "achatbot[fastapi_bot_server]"`

## pypi
```bash
python3 -m venv .venv_achatbot
source .venv_achatbot/bin/activate
pip install achatbot
# optional-dependencies e.g.
pip install "achatbot[fastapi_bot_server]"
```

## local
```bash
git clone --recursive https://github.com/ai-bot-pro/chat-bot.git
cd chat-bot
python3 -m venv .venv_achatbot
source .venv_achatbot/bin/activate
bash scripts/pypi_achatbot.sh dev
# optional-dependencies e.g.
pip install "dist/achatbot-{$version}-py3-none-any.whl[fastapi_bot_server]"
```

## run local lite avatar chat bot
```shell
# install dependencies (replace $version) (if use cpu(default) install lite_avatar)
pip install "dist/achatbot-{$version}-py3-none-any.whl[fastapi_bot_server,livekit,livekit-api,daily,agora,silero_vad_analyzer,sense_voice_asr,openai_llm_processor,google_llm_processor,litellm_processor,together_ai,tts_edge,lite_avatar]"
# install dependencies (replace $version) (if use gpu(cuda) install lite_avatar_gpu)
pip install "dist/achatbot-{$version}-py3-none-any.whl[fastapi_bot_server,livekit,livekit-api,daily,agora,silero_vad_analyzer,sense_voice_asr,openai_llm_processor,google_llm_processor,litellm_processor,together_ai,tts_edge,lite_avatar_gpu]"
# download model weights
huggingface-cli download weege007/liteavatar --local-dir ./models/weege007/liteavatar
huggingface-cli download FunAudioLLM/SenseVoiceSmall --local-dir ./models/FunAudioLLM/SenseVoiceSmall
# run local lite-avatar chat bot
python -m src.cmd.bots.main -f config/bots/daily_liteavatar_echo_bot.json
python -m src.cmd.bots.main -f config/bots/daily_liteavatar_chat_bot.json
```
More details: https://github.com/ai-bot-pro/achatbot/pull/161

## run local lam_audio2expression avatar chat bot
```shell
# install dependencies (replace $version) 
pip install "dist/achatbot-{$version}-py3-none-any.whl[fastapi_bot_server,silero_vad_analyzer,sense_voice_asr,openai_llm_processor,google_llm_processor,litellm_processor,together_ai,tts_edge,lam_audio2expression_avatar]"
pip install spleeter==2.4.2
pip install typing_extensions==4.14.0 aiortc==1.13.0 transformers==4.36.2 protobuf==5.29.4
# download model weights
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/LAM_audio2exp_streaming.tar -P ./models/LAM_audio2exp/
tar -xzvf ./models/LAM_audio2exp/LAM_audio2exp_streaming.tar -C ./models/LAM_audio2exp && rm ./models/LAM_audio2exp/LAM_audio2exp_streaming.tar
git clone --depth 1 https://www.modelscope.cn/AI-ModelScope/wav2vec2-base-960h.git ./models/facebook/wav2vec2-base-960h
huggingface-cli download FunAudioLLM/SenseVoiceSmall  --local-dir ./models/FunAudioLLM/SenseVoiceSmall
# run http signaling service + webrtc + websocket local lam_audio2expression-avatar chat bot
python -m src.cmd.webrtc_websocket.fastapi_ws_signaling_bot_serve -f config/bots/small_webrtc_fastapi_websocket_avatar_echo_bot.json
python -m src.cmd.webrtc_websocket.fastapi_ws_signaling_bot_serve -f config/bots/small_webrtc_fastapi_websocket_avatar_chat_bot.json
# run http signaling service + webrtc + websocket voice avatar agent web ui
cd ui/webrtc_websocket/lam_audio2expression_avatar_ts && npm install && npm run dev
# run websocket signaling service + webrtc + websocket local lam_audio2expression-avatar chat bot
python -m src.cmd.webrtc_websocket.fastapi_ws_signaling_bot_serve_v2 -f config/bots/small_webrtc_fastapi_websocket_avatar_echo_bot.json
python -m src.cmd.webrtc_websocket.fastapi_ws_signaling_bot_serve_v2 -f config/bots/small_webrtc_fastapi_websocket_avatar_chat_bot.json
# run websocket signaling service + webrtc + websocket voice avatar agent web ui
cd ui/webrtc_websocket/lam_audio2expression_avatar_ts_v2 && npm install && npm run dev
```
More details: https://github.com/ai-bot-pro/achatbot/pull/164 and https://github.com/ai-bot-pro/achatbot/pull/206 | online lam_audio2expression avatar: https://avatar-2lm.pages.dev/

---

#  Run chat bots
- :memo: [Run chat bots with colab notebook](https://github.com/ai-bot-pro/achatbot/blob/main/docs/colab.md)




# License

achatbot is released under the [BSD 3 license](LICENSE). (Additional code in this distribution is covered by the MIT and Apache Open Source
licenses.) However you may have other legal obligations that govern your use of content, such as the terms of service for third-party models.
