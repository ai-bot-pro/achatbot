# achatbot
[![PyPI](https://img.shields.io/pypi/v/achatbot)](https://pypi.org/project/achatbot/)

achatbot factory, create chat bots with llm(tools), asr, tts, vad, ocr, detect object etc..

<details>
<summary>:evergreen_tree: Project Structure</summary>

# Project Structure
![project-structure](https://github.com/user-attachments/assets/5bf7cebb-e590-4718-a78a-6b0c0b36ea28)  

</details>

<details>
<summary>:herb: Features</summary>

# Features
- demo
  
  - [podcast](https://github.com/ai-bot-pro/achatbot/blob/main/demo/content_parser_tts.py)  AI Podcast：[https://podcast-997.pages.dev/](https://podcast-997.pages.dev/) :)
  
    ```shell
    # need GOOGLE_API_KEY in environment variables
    # default use language English
    
    # websit
    python -m demo.content_parser_tts instruct-content-tts \
        "https://en.wikipedia.org/wiki/Large_language_model"
    
    python -m demo.content_parser_tts instruct-content-tts \
        --role-tts-voices zh-CN-YunjianNeural \
        --role-tts-voices zh-CN-XiaoxiaoNeural \
        --language zh \
        "https://en.wikipedia.org/wiki/Large_language_model"
    
    # pdf
    # https://web.stanford.edu/~jurafsky/slp3/ed3bookaug20_2024.pdf 600 page is ok~ :)
    python -m demo.content_parser_tts instruct-content-tts \
        "/Users/wuyong/Desktop/Speech and Language Processing.pdf"
    
    python -m demo.content_parser_tts instruct-content-tts \
        --role-tts-voices zh-CN-YunjianNeural \
        --role-tts-voices zh-CN-XiaoxiaoNeural \
        --language zh \
        "/Users/wuyong/Desktop/Speech and Language Processing.pdf"
    ```
  
- cmd chat bots:

  - [local-terminal-chat](https://github.com/ai-bot-pro/achatbot/tree/main/src/cmd/local-terminal-chat)(be/fe)
  - [remote-queue-chat](https://github.com/ai-bot-pro/achatbot/tree/main/src/cmd/remote-queue-chat)(be/fe)
  - [grpc-terminal-chat](https://github.com/ai-bot-pro/achatbot/tree/main/src/cmd/grpc/terminal-chat)(be/fe)
  - [grpc-speaker](https://github.com/ai-bot-pro/achatbot/tree/main/src/cmd/grpc/speaker)
  - [http fastapi_daily_bot_serve](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/http/server/fastapi_daily_bot_serve.py) (with chat bots pipeline)
  - [**bots with config**](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/main.py)  see notebooks:
    - [Run chat bots with colab notebook](https://github.com/ai-bot-pro/achatbot?tab=readme-ov-file#run-chat-bots-with-colab-notebook)  🏃

- support transport connector: 
  - [x] pipe(UNIX socket), 
  - [x] grpc, 
  - [x] queue (redis),
  - [ ] websocket
  - [ ] TCP/IP socket

- chat bot processors: 
  - aggreators(llm use, assistant message), 
  - ai_frameworks
    - [x] [langchain](https://www.langchain.com/): RAG
    - [ ] [llamaindex](https://www.llamaindex.ai/): RAG
    - [ ] [autoagen](https://github.com/microsoft/autogen): multi Agents
  - realtime voice inference(RTVI),
  - transport: 
    - webRTC: (daily,livekit KISS)
      - [x] **[daily](https://github.com/ai-bot-pro/achatbot/blob/main/src/transports/daily.py)**: audio, video(image)
      - [x] **[livekit](https://github.com/ai-bot-pro/achatbot/blob/main/src/transports/livekit.py)**: audio, video(image)
      - [x] **[agora](https://github.com/ai-bot-pro/achatbot/blob/main/src/transports/agora.py)**: audio, video(image)
      - [x] **[small_webrtc](https://github.com/ai-bot-pro/achatbot/blob/main/src/transports/small_webrtc.py)**: audio, video(image)
    - [x] [Websocket server](https://github.com/ai-bot-pro/achatbot/blob/main/src/transports/websocket_server.py)
  - ai processor: llm, tts, asr etc..
    - llm_processor:
      - [x] [openai](https://github.com/ai-bot-pro/achatbot/blob/main/test/integration/processors/test_openai_llm_processor.py)(use openai sdk)
      - [x] [google gemini](https://github.com/ai-bot-pro/achatbot/blob/main/test/integration/processors/test_google_llm_processor.py)(use google-generativeai sdk)
      - [x] [litellm](https://github.com/ai-bot-pro/achatbot/blob/main/test/integration/processors/test_litellm_processor.py)(use openai input/output format proxy sdk) 
  
- core module:
  - local llm: 
    - [x] llama-cpp (support text,vision with function-call model)
      - [x] llm_llamacpp_generator
    - [x] fastdeploy:
      - [x] llm_fastdeploy_vision_ernie4v
      - [x] llm_fastdeploy_generator
    - [x] tensorrt_llm:
      - [x] llm_trtllm_generator
      - [x] llm_trtllm_runner_generator
    - [x] sglang:
      - [x] llm_sglang_generator
    - [x] vllm:
      - [x] llm_vllm_generator
    - [x] transformers(manual, pipeline) (support text; vision,vision+image; speech,voice; vision+voice)
      - [x] llm_transformers_manual_vision_llama
      - [x] llm_transformers_manual_vision_molmo
      - [x] llm_transformers_manual_vision_qwen
      - [x] llm_transformers_manual_vision_deepseek
      - [x] llm_transformers_manual_vision_janus_flow
      - [x] llm_transformers_manual_vision_janus
      - [x] llm_transformers_manual_vision_smolvlm
      - [x] llm_transformers_manual_vision_gemma
      - [x] llm_transformers_manual_vision_fastvlm
      - [x] llm_transformers_manual_vision_kimi
      - [x] llm_transformers_manual_vision_mimo
      - [x] llm_transformers_manual_vision_keye
      - [x] llm_transformers_manual_vision_glm4v
      - [x] llm_transformers_manual_image_janus_flow
      - [x] llm_transformers_manual_image_janus
      - [x] llm_transformers_manual_speech_llasa
      - [x] llm_transformers_manual_speech_step
      - [x] llm_transformers_manual_voice_glm
      - [x] llm_transformers_manual_vision_voice_minicpmo, llm_transformers_manual_voice_minicpmo,llm_transformers_manual_audio_minicpmo,llm_transformers_manual_text_speech_minicpmo,llm_transformers_manual_instruct_speech_minicpmo,llm_transformers_manual_vision_minicpmo
      - [x] llm_transformers_manual_qwen2_5omni, llm_transformers_manual_qwen2_5omni_audio_asr,llm_transformers_manual_qwen2_5omni_vision,llm_transformers_manual_qwen2_5omni_speech,llm_transformers_manual_qwen2_5omni_vision_voice,llm_transformers_manual_qwen2_5omni_text_voice,llm_transformers_manual_qwen2_5omni_audio_voice
      - [x] llm_transformers_manual_kimi_voice,llm_transformers_manual_kimi_audio_asr,llm_transformers_manual_kimi_text_voice
      - [x] llm_transformers_manual_vita_text llm_transformers_manual_vita_audio_asr llm_transformers_manual_vita_tts llm_transformers_manual_vita_text_voice llm_transformers_manual_vita_voice
      - [x] llm_transformers_manual_phi4_vision_speech,llm_transformers_manual_phi4_audio_asr,llm_transformers_manual_phi4_audio_translation,llm_transformers_manual_phi4_vision,llm_transformers_manual_phi4_audio_chat
  - remote api llm: personal-ai(like openai api, other ai provider)
  
- AI modules:
  - functions:
    - [x] search: search,search1,serper
    - [x] weather: openweathermap
  - speech:
    - [x] asr: 
      - [x] whisper_asr, whisper_timestamped_asr, whisper_faster_asr, whisper_transformers_asr, whisper_mlx_asr
      - [x] whisper_groq_asr
      - [x] sense_voice_asr
      - [x] minicpmo_asr (whisper)
      - [x] qwen2_5omni_asr (whisper)
      - [x] kimi_asr (whisper)
      - [x] vita_asr (sensevoice-small)
      - [x] phi4_asr (conformer)
    - [x] audio_stream: daily_room_audio_stream(in/out), pyaudio_stream(in/out)
    - [x] detector: porcupine_wakeword,pyannote_vad,webrtc_vad,silero_vad,webrtc_silero_vad,fsmn_vad
    - [x] player: stream_player
    - [x] recorder: rms_recorder, wakeword_rms_recorder, vad_recorder, wakeword_vad_recorder
    - [x] tts: 
      - [x] tts_edge
      - [x] tts_g
      - [x] tts_coqui
      - [x] tts_chat
      - [x] tts_cosy_voice,tts_cosy_voice2
      - [x] tts_f5
      - [x] tts_openvoicev2
      - [x] tts_kokoro,tts_onnx_kokoro
      - [x] tts_fishspeech
      - [x] tts_llasa
      - [x] tts_minicpmo
      - [x] tts_zonos
      - [x] tts_step
      - [x] tts_spark
      - [x] tts_orpheus
      - [x] tts_mega3
      - [x] tts_vita
    - [x] vad_analyzer: 
      - [x] daily_webrtc_vad_analyzer
      - [x] silero_vad_analyzer
  - vision
    - [x] OCR(*Optical Character Recognition*):
      - [ ] [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
      - [x]  [GOT](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)(*the General OCR Theory*)
    - [x] Detector:
      - [x] [YOLO](https://docs.ultralytics.com/) (*You Only Look Once*)
      - [ ] [RT-DETR v2](https://github.com/lyuwenyu/RT-DETR) (*RealTime End-to-End Object Detection with Transformers*)
  
- gen modules config(*.yaml, local/test/prod) from env with file: `.env`
   u also use HfArgumentParser this module's args to local cmd parse args

- deploy to cloud ☁️ serverless: 
  - vercel (frontend ui pages)
  - Cloudflare(frontend ui pages), personal ai workers 
  - [fastapi-daily-chat-bot](https://github.com/ai-bot-pro/achatbot/tree/main/deploy/cerebrium/fastapi-daily-chat-bot) on cerebrium (provider aws)
  - [fastapi-daily-chat-bot](https://github.com/ai-bot-pro/achatbot/tree/main/deploy/leptonai/fastapi-daily-chat-bot) on leptonai
  - [fastapi-daily-chat-bot](https://github.com/ai-bot-pro/achatbot/tree/main/deploy/modal) on modal
  - aws lambda + api Gateway
  - docker -> k8s/k3s
  - etc...

</details>

<details>
<summary>:sunflower: Service Deployment Architecture</summary>

# Service Deployment Architecture

## UI (easy to deploy with github like pages)
- [x] [ui/web-client-ui](https://github.com/ai-bot-pro/web-client-ui)
  deploy it to cloudflare page with vite, access https://chat-client-weedge.pages.dev/

- [x] [ui/educator-client](https://github.com/ai-bot-pro/educator-client)
  deploy it to cloudflare page with vite, access https://educator-client.pages.dev/

- [x] [chat-bot-rtvi-web-sandbox](https://github.com/ai-bot-pro/chat-bot-rtvi-client/tree/main/chat-bot-rtvi-web-sandbox)
  use this web sandbox to test config, actions with [DailyRTVIGeneralBot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/rtvi/daily_rtvi_general_bot.py)

- [x] [vite-react-rtvi-web-voice](https://github.com/ai-bot-pro/vite-react-rtvi-web-voice) rtvi web voice chat bots, diff cctv roles etc, u can diy your own role by change the system prompt with [DailyRTVIGeneralBot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/rtvi/daily_rtvi_general_bot.py)
  deploy it to cloudflare page with vite, access https://role-chat.pages.dev/

- [x] [vite-react-web-vision](https://github.com/ai-bot-pro/vite-react-web-vision) 
  deploy it to cloudflare page with vite, access https://vision-weedge.pages.dev/

- [x] [nextjs-react-web-storytelling](https://github.com/ai-bot-pro/nextjs-react-web-storytelling) 
  deploy it to cloudflare page worker with nextjs, access https://storytelling.pages.dev/ 

- [x] [websocket-demo](https://github.com/ai-bot-pro/achatbot/blob/main/ui/websocket/simple-demo): websocket audio chat bot demo

- [x] [webrtc-demo](https://github.com/ai-bot-pro/achatbot/blob/main/ui/webrtc/simple-demo): webrtc audio chat bot demo

- [x] [webrtc websocket voice avatar](https://github.com/ai-bot-pro/achatbot/tree/main/ui/webrtc_websocket):
  - [x] [webrtc+websocket lam audio2expression avatar bot demo intro](https://github.com/ai-bot-pro/achatbot/tree/main/ui/webrtc_websocket/lam_audio2expression_avatar): native js logic, get audio to play and print expression from websocket pb avatar_data_frames Message
  - [x] [lam_audio2expression_avatar_ts](https://github.com/ai-bot-pro/achatbot/tree/main/ui/webrtc_websocket/lam_audio2expression_avatar_ts_v2): **http signaling service** and use vite+ts+gaussian-splat-renderer-for-lam to play audio and render expression from websocket pb avatar_data_frames Message
  - [x] [**lam_audio2expression_avatar_ts_v2**](https://github.com/ai-bot-pro/achatbot/tree/main/ui/webrtc_websocket/lam_audio2expression_avatar_ts_v2): **websocket signaling service** and use vite+ts+gaussian-splat-renderer-for-lam to play audio and render expression from websocket pb avatar_data_frames Message, access https://avatar-2lm.pages.dev/ 
  
  



## Server Deploy (CD)
- [x] [deploy/modal](https://github.com/ai-bot-pro/achatbot/tree/main/deploy/modal)(KISS) 👍🏻 
- [x] [deploy/leptonai](https://github.com/ai-bot-pro/achatbot/tree/main/deploy/leptonai)(KISS)👍🏻
- [x] [deploy/cerebrium/fastapi-daily-chat-bot](https://github.com/ai-bot-pro/achatbot/tree/main/deploy/cerebrium/fastapi-daily-chat-bot) :)
- [x] [deploy/aws/fastapi-daily-chat-bot](https://github.com/ai-bot-pro/achatbot/tree/main/deploy/aws/fastapi-daily-chat-bot) :|
- [x] [deploy/docker/fastapi-daily-chat-bot](https://github.com/ai-bot-pro/achatbot/tree/main/deploy/docker) 🏃

</details>


# Install
> [!NOTE]
> `python --version` >=3.10 with [asyncio-task](https://docs.python.org/3.10/library/asyncio-task.html)
> if install `achatbot[tts_openvoicev2]` need install melo-tts `pip install git+https://github.com/myshell-ai/MeloTTS.git`

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

More details: https://github.com/ai-bot-pro/achatbot/pull/164 | online lam_audio2expression avatar: https://avatar-2lm.pages.dev/

---
HTTP signaling service +  webrtc + websocket transports I/O bridge:
<img width="1151" alt="image" src="https://github.com/user-attachments/assets/59e9eace-b27f-4f4c-b314-ee5988988335" />

Websocket signaling service +  webrtc + websocket transports I/O bridge:

<img width="1167" alt="image" src="https://github.com/user-attachments/assets/3963ff54-77ff-4c2f-a41f-7f9e9029d041" />

---
Websocket signaling service +  websocket + webrtc-queue transports I/O bridge:
<img width="1177" alt="image" src="https://github.com/user-attachments/assets/571d8d02-c161-4867-8f09-1d2777b27f0b" />


#  Run chat bots

## :memo: Run chat bots with colab notebook


|                           Chat Bot                           | optional-dependencies                                        | Colab                                                        | Device                                                       | Pipeline Desc                                                |
| :----------------------------------------------------------: | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [daily_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/daily_bot.py)<br />[livekit_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/livekit_bot.py)<br />[agora_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/agora_bot.py)<br /> | e.g.:<br />agora_channel_audio_stream\| daily_room_audio_stream \| livekit_room_audio_stream,<br />sense_voice_asr,<br />groq \| together api llm(text), <br />tts_edge | <a href="https://github.com/weedge/doraemon-nb/blob/main/webrtc_audio_bot.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | CPU (free, 2 cores)                                          | e.g.:<br />daily \| livekit room in stream<br />-> silero (vad)<br />-> sense_voice (asr) <br />-> groq \| together  (llm) <br />-> edge (tts)<br />-> daily \| livekit room out stream |
| [generate_audio2audio](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/remote-queue-chat/generate_audio2audio.py) | remote_queue_chat_bot_be_worker                              | <a href="https://github.com/weedge/doraemon-nb/blob/main/chat_bot_gpu_worker.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | T4(free)                                                     | e.g.:<br />pyaudio in stream<br />-> silero (vad)<br />-> sense_voice (asr) <br />-> qwen (llm) <br />-> cosy_voice (tts)<br />-> pyaudio out stream |
| [daily_describe_vision_tools_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/vision/daily_describe_vision_tools_bot.py)<br />[livekit_describe_vision_tools_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/vision/livekit_describe_vision_tools_bot.py)<br />[agora_describe_vision_tools_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/agora_describe_vision_tools_bot.py)<br /> | e.g.:<br />daily_room_audio_stream \|livekit_room_audio_stream<br />deepgram_asr,<br />goole_gemini,<br />tts_edge | <a href="https://github.com/weedge/doraemon-nb/blob/main/achatbot_describe_vision_tools_bot.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | CPU(free, 2 cores)                                           | e.g.:<br />daily \|livekit room in stream<br />-> silero (vad)<br />-> deepgram (asr) <br />-> google gemini  <br />-> edge (tts)<br />-> daily \|livekit room out stream |
| [daily_describe_vision_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/vision/daily_describe_vision_bot.py)<br />[livekit_describe_vision_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/vision/livekit_describe_vision_bot.py)<br />[agora_describe_vision_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/agora_describe_vision_bot.py)<br /> | e.g.:<br />daily_room_audio_stream \| livekit_room_audio_stream<br />sense_voice_asr,<br />llm_transformers_manual_vision_qwen,<br />tts_edge | achatbot_vision_qwen_vl.ipynb:<br /><a href="https://github.com/weedge/doraemon-nb/blob/main/achatbot_vision_qwen_vl.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a><br />achatbot_vision_janus.ipynb:<br /><a href="https://github.com/weedge/doraemon-nb/blob/main/achatbot_vision_janus.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a><br />achatbot_vision_minicpmo.ipynb:<br /><a href="https://github.com/weedge/doraemon-nb/blob/main/achat_miniCPMo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a><br />achatbot_kimivl.ipynb:<br /><a href="https://github.com/weedge/doraemon-nb/blob/main/achatbot_kimivl.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a><br />achatbot_phi4_multimodal.ipynb:<br /><a href="https://github.com/weedge/doraemon-nb/blob/main/achatbot_phi4_multimodal.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | - Qwen2-VL-2B-Instruct<br /> T4(free)<br />- Qwen2-VL-7B-Instruct<br />L4<br />- Llama-3.2-11B-Vision-Instruct<br />L4<br />- allenai/Molmo-7B-D-0924<br />A100 | e.g.:<br />daily \| livekit room in stream<br />-> silero (vad)<br />-> sense_voice (asr) <br />-> qwen-vl (llm) <br />-> edge (tts)<br />-> daily \| livekit room out stream |
| [daily_chat_vision_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/vision/daily_chat_vision_bot.py)<br />[livekit_chat_vision_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/vision/livekit_chat_vision_bot.py)<br />[agora_chat_vision_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/agora_chat_vision_bot.py)<br /> | e.g.:<br />daily_room_audio_stream \|livekit_room_audio_stream<br />sense_voice_asr,<br />llm_transformers_manual_vision_qwen,<br />tts_edge | <a href="https://github.com/weedge/doraemon-nb/blob/main/achatbot_daily_chat_vision_bot.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | - Qwen2-VL-2B-Instruct<br /> T4(free)<br />- Qwen2-VL-7B-Instruct<br />L4<br />- Ll<br/>ama-3.2-11B-Vision-Instruct<br />L4<br />- allenai/Molmo-7B-D-0924<br />A100 | e.g.:<br />daily \| livekit room in stream<br />-> silero (vad)<br />-> sense_voice (asr) <br />-> llm answer guide qwen-vl (llm) <br />-> edge (tts)<br />-> daily \| livekit room out stream |
| [daily_chat_tools_vision_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/vision/daily_chat_tools_vision_bot.py)<br />[livekit_chat_tools_vision_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/vision/livekit_chat_tools_vision_bot.py)<br />[agora_chat_tools_vision_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/vision/agora_chat_tools_vision_bot.py)<br /> | e.g.:<br />daily_room_audio_stream \| livekit_room_audio_stream<br />sense_voice_asr,<br />groq api llm(text), <br />tools:<br />- llm_transformers_manual_vision_qwen,<br />tts_edge | <a href="https://github.com/weedge/doraemon-nb/blob/main/achatbot_daily_chat_tools_vision_bot.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | - Qwen2-VL-2B-Instruct<br<br/> /> T4(free)<br />- Qwen2-VL-7B-Instruct<br />L4<br />- Llama-3.2-11B-Vision-Instruct<br />L4 <br />- allenai/Molmo-7B-D-0924<br />A100 | e.g.:<br />daily \| livekit room in stream<br />-> silero (vad)<br />-> sense_voice (asr) <br />->llm with tools qwen-vl  <br />-> edge (tts)<br />-> daily \| livekit room out stream |
| [daily_annotate_vision_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/vision/daily_annotate_vision_bot.py)<br />[livekit_annotate_vision_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/vision/livekit_annotate_vision_bot.py)<br />[agora_annotate_vision_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/vision/agora_annotate_vision_bot.py)<br /> | e.g.:<br />daily_room_audio_stream \| livekit_room_audio_stream<br />vision_yolo_detector<br />tts_edge | <a href="https://github.com/weedge/doraemon-nb/blob/main/daily_annotate_vision_bot.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | T4(free)                                                     | e.g.:<br />daily \| livekit room in stream<br />vision_yolo_detector<br />-> edge (tts)<br />-> daily \| livekit room out stream |
| [daily_detect_vision_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/vision/daily_detect_vision_bot.py)<br />[livekit_detect_vision_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/vision/livekit_detect_vision_bot.py)<br />[agora_detect_vision_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/vision/agora_detect_vision_bot.py)<br /> | e.g.:<br />daily_room_audio_stream \| livekit_room_audio_stream<br />vision_yolo_detector<br />tts_edge | <a href="https://github.com/weedge/doraemon-nb/blob/main/daily_detect_vision_bot.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | T4(free)                                                     | e.g.:<br />daily \| livekit room in stream<br />vision_yolo_detector<br />-> edge (tts)<br />-> daily \| livekit room out stream |
| [daily_ocr_vision_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/vision/daily_ocr_vision_bot.py)<br />[livekit_ocr_vision_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/vision/livekit_ocr_vision_bot.py)<br/>[agora_ocr_vision_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/vision/agora_ocr_vision_bot.py)<br/> | e.g.:<br />daily_room_audio_stream \| livekit_room_audio_stream<br />sense_voice_asr,<br />vision_transformers_got_ocr<br />tts_edge | <a href="https://github.com/weedge/doraemon-nb/blob/main/daily_ocr_vision_bot.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | T4(free)                                                     | e.g.:<br />daily \| livekit room in stream<br />-> silero (vad)<br />-> sense_voice (asr) <br />vision_transformers_got_ocr<br />-> edge (tts)<br />-> daily \| livekit room out stream |
| [daily_month_narration_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/image/daily_month_narration_bot.py) | e.g.:<br />daily_room_audio_stream <br />groq \|together api llm(text),<br />hf_sd, together api (image)<br />tts_edge | <a href="https://github.com/weedge/doraemon-nb/blob/main/achatbot_daily_month_narration_bot.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | when use sd model with diffusers<br />T4(free) cpu+cuda (slow)<br />L4 cpu+cuda<br/>A100 all cuda<br /> | e.g.:<br />daily room in stream<br />-> together  (llm) <br />-> hf sd gen image model<br />-> edge (tts)<br />-> daily  room out stream |
| [daily_storytelling_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/image/storytelling/daily_bot.py) | e.g.:<br />daily_room_audio_stream <br />groq \|together api llm(text),<br />hf_sd, together api (image)<br />tts_edge | <a href="https://github.com/weedge/doraemon-nb/blob/main/achatbot_daily_storytelling_bot.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | cpu (2 cores)<br />when use sd model with diffusers<br />T4(free) cpu+cuda (slow)<br />L4 cpu+cuda<br/>A100 all cuda<br /> | e.g.:<br />daily room in stream<br />-> together  (llm) <br />-> hf sd gen image model<br />-> edge (tts)<br />-> daily  room out stream |
| [websocket_server_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/websocket_server_bot.py)<br />[fastapi_websocket_server_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/fastapi_websocket_server_bot.py)<br /> | e.g.:<br /> websocket_server<br />sense_voice_asr,<br />groq \|together api llm(text),<br />tts_edge | <a href="https://github.com/weedge/doraemon-nb/blob/main/achatbot_websocket_server_bot.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | cpu(2 cores)                                                 | e.g.:<br />websocket protocol  in stream<br />-> silero (vad)<br />-> sense_voice (asr) <br />-> together  (llm) <br />-> edge (tts)<br />-> websocket protocol out stream |
| [daily_natural_conversation_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/nlp/daily_natural_conversation_bot.py) | e.g.:<br /> daily_room_audio_stream<br />sense_voice_asr,<br />groq \|together api llm(NLP task),<br />gemini-1.5-flash (chat)<br />tts_edge | <a href="https://github.com/weedge/doraemon-nb/blob/main/achat_natural_conversation_bot.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | cpu(2 cores)                                                 | e.g.:<br />daily room in stream<br />-> together  (llm NLP task) <br />->  gemini-1.5-flash model (chat)<br />-> edge (tts)<br />-> daily  room out stream |
| [fastapi_websocket_moshi_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/voice/fastapi_websocket_moshi_bot.py) | e.g.:<br /> websocket_server<br />moshi opus stream voice llm<br /> | <a href="https://github.com/weedge/doraemon-nb/blob/main/achatbot_moshi_bot.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | L4/A100                                                      | websocket protocol  in stream<br />-> silero (vad)<br />-> moshi opus stream voice llm<br />-> websocket protocol out stream |
| [daily_asr_glm_voice_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/voice/daily_asr_glm_voice_bot.py)<br>[daily_glm_voice_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/voice/daily_glm_voice_bot.py)<br /> | e.g.:<br /> daily_room_audio_stream<br />glm voice llm<br /> | <a href="https://github.com/weedge/doraemon-nb/blob/main/achatbot_glm_voice_bot.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | T4/L4/A100                                                   | e.g.:<br />daily room in stream<br />->glm4-voice<br />-> daily  room out stream |
| [daily_freeze_omni_voice_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/voice/daily_freeze_omni_voice_bot.py) | e.g.:<br /> daily_room_audio_stream<br />freezeOmni voice llm<br /> | <a href="https://github.com/weedge/doraemon-nb/blob/main/achatbot_freeze_omni_voice_bot.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | L4/A100                                                      | e.g.:<br />daily room in stream<br />->freezeOmni-voice<br />-> daily  room out stream |
| [daily_asr_minicpmo_voice_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/voice/daily_asr_minicpmo_voice_bot.py)<br/>[daily_minicpmo_voice_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/voice/daily_minicpmo_voice_bot.py)<br />[daily_minicpmo_vision_voice_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/omni/daily_minicpmo_vision_voice_bot.py)<br /> | e.g.:<br /> daily_room_audio_stream<br />minicpmo llm<br />  | <a href="https://github.com/weedge/doraemon-nb/blob/main/achat_miniCPMo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | T4: MiniCPM-o-2_6-int4<br />L4/A100: MiniCPM-o-2_6<br />     | e.g.:<br />daily room in stream<br />->minicpmo<br />-> daily  room out stream |
| [livekit_asr_qwen2_5omni_voice_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/voice/livekit_asr_qwen2_5omni_voice_bot.py)<br/>[livekit_qwen2_5omni_voice_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/voice/livekit_qwen2_5omni_voice_bot.py)<br />[livekit_qwen2_5omni_vision_voice_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/omni/livekit_qwen2_5omni_vision_voice_bot.py)<br /> | e.g.:<br /> livekit_room_audio_stream<br />qwen2.5omni llm<br /> | <a href="https://github.com/weedge/doraemon-nb/blob/main/achatbot_Qwen2_5_Omni.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | A100                                                         | e.g.:<br />livekit room in stream<br />->qwen2.5omni<br />-> livekit  room out stream |
| [livekit_asr_kimi_voice_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/voice/livekit_asr_kimi_voice_bot.py)<br/>[livekit_kimi_voice_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/voice/livekit_kimi_voice_bot.py)<br/> | e.g.:<br /> livekit_room_audio_stream<br />kimi audio llm<br /> | <a href="https://github.com/weedge/doraemon-nb/blob/main/achatbot_kimi_audio.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | A100                                                         | e.g.:<br />livekit room in stream<br />-> Kimi-Audio<br />-> livekit  room out stream |
| [livekit_asr_vita_voice_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/voice/livekit_asr_vita_voice_bot.py)<br/>[livekit_vita_voice_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/voice/livekit_vita_voice_bot.py)<br/> | e.g.:<br /> livekit_room_audio_stream<br />vita audio llm<br /> | <a href="https://github.com/weedge/doraemon-nb/blob/main/achatbot_vita_audio.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | L4/100                                                       | e.g.:<br />livekit room in stream<br />-> VITA-Audio<br />-> livekit  room out stream |
| [daily_phi4_voice_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/voice/daily_phi4_voice_bot.py)<br/>[daily_phi4_vision_speech_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/omni/daily_phi4_vision_speech_bot.py)<br/> | e.g.:<br /> daily_room_audio_stream<br />phi4-multimodal llm<br /> | <a href="https://github.com/weedge/doraemon-nb/blob/main/achatbot_phi4_multimodal.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | L4/100                                                       | e.g.:<br />daily room in stream<br />-> phi4-multimodal<br />-> edge (tts)<br />-> daily  room out stream |
| [daliy_multi_mcp_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/mcp/daily_multi_mcp_bot.py)<br />[livekit_multi_mcp_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/mcp/livekit_multi_mcp_bot.py)<br />[agora_multi_mcp_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/mcp/agora_multi_mcp_bot.py)<br /> | e.g.:<br />agora_channel_audio_stream \|daily_room_audio_stream \|livekit_room_audio_stream,<br />sense_voice_asr,<br />groq \|together api llm(text), <br />mcp <br />tts_edge | <a href="https://github.com/weedge/doraemon-nb/blob/main/achatbot_multiMCP_bot.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | CPU (free, 2 cores)                                          | e.g.:<br />agora \| daily \|livekit room in stream<br />-> silero (vad)<br />-> sense_voice (asr) <br />-> groq \|together  (llm) <br />-> mcp server tools<br />-> edge (tts)<br />-> daily \|livekit room out stream |
| [daily_liteavatar_chat_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/avatar/daily_liteavatar_chat_bot.py)<br />[daily_liteavatar_echo_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/avatar/daily_liteavatar_echo_bot.py)<br />[livekit_musetalk_chat_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/avatar/livekit_musetalk_chat_bot.py)<br />[livekit_musetalk_echo_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/avatar/livekit_musetalk_echo_bot.py)<br /> | e.g.:<br />agora_channel_audio_stream \|daily_room_audio_stream \|livekit_room_audio_stream,<br />sense_voice_asr,<br />groq \|together api llm(text), <br />tts_edge<br />avatar<br /> | achatbot_avatar_musetalk.ipynb:<br /><a href="https://github.com/weedge/doraemon-nb/blob/main/achatbot_avatar_musetalk.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | CPU/T4/L4                                                    | e.g.:<br />agora \|daily \|livekit room in stream<br />-> silero (vad)<br />-> sense_voice (asr) <br />-> groq \|together  (llm) <br />-> edge (tts)<br />-> avatar <br />-> daily \|livekit room out stream |
|                                                              |                                                              |                                                              |                                                              |                                                              |


<details>
<summary>:new_moon: Run local chat bots</summary>

## Run local chat bots

> [!NOTE]
>
> - run src code, replace achatbot to src, don't need set `ACHATBOT_PKG=1` e.g.:
>   ```
>   TQDM_DISABLE=True \
>        python -m src.cmd.local-terminal-chat.generate_audio2audio > log/std_out.log
>   ```
> - PyAudio need install python3-pyaudio 
> e.g. ubuntu `apt-get install python3-pyaudio`, macos `brew install portaudio`
> see: https://pypi.org/project/PyAudio/
>
> - llm llama-cpp-python init use cpu Pre-built Wheel to install, 
> if want to use other lib(cuda), see: https://github.com/abetlen/llama-cpp-python#installation-configuration
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
    </details>

    
<details>
<summary>:waxing_crescent_moon: Run remote http fastapi daily chat bots</summary>

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
   - run a simple daily chat bot api, with ui/web-client-ui (default language: zh)
   ```
   curl -XPOST "http://0.0.0.0:4321/bot_join/DailyBot" \
    -H "Content-Type: application/json" \
    -d '{}' | jq .
   ```
   </details>

<details>
<summary>:first_quarter_moon: Run remote rpc chat bot worker</summary>

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
</details>


<details>
<summary>:waxing_gibbous_moon: Run remote queue chat bot worker</summary>

## Run remote queue chat bot worker
1. run `pip install "achatbot[remote_queue_chat_bot_be_worker]"` to install dependencies to run queue chat bot worker; e.g.:
   - use default env params to run 
    ```
    ACHATBOT_PKG=1 REDIS_PASSWORD=$redis_pwd RUN_OP=be TQDM_DISABLE=True \
        python -m achatbot.cmd.remote-queue-chat.generate_audio2audio > ~/.achatbot/log/be_std_out.log
    ```
   - sense_voice(asr) -> qwen (llm) -> cosy_voice (tts)
   u can login [redislabs](https://app.redislabs.com/#/) create 30M free databases; set `REDIS_HOST`,`REDIS_PORT` and `REDIS_PASSWORD` to run, e.g.:
   ```
    ACHATBOT_PKG=1 RUN_OP=be \
      TQDM_DISABLE=True \
      REDIS_PASSWORD=$redis_pwd \
      REDIS_HOST=redis-14241.c256.us-east-1-2.ec2.redns.redis-cloud.com \
      REDIS_PORT=14241 \
      ASR_TAG=sense_voice_asr \
      ASR_LANG=zn \
      ASR_MODEL_NAME_OR_PATH=~/.achatbot/models/FunAudioLLM/SenseVoiceSmall \
      N_GPU_LAYERS=33 FLASH_ATTN=1 \
      LLM_MODEL_NAME=qwen \
      LLM_MODEL_PATH=~/.achatbot/models/qwen1_5-7b-chat-q8_0.gguf \
      TTS_TAG=tts_cosy_voice \
      python -m achatbot.cmd.remote-queue-chat.generate_audio2audio > ~/.achatbot/log/be_std_out.log
   ```
2. run `pip install "achatbot[remote_queue_chat_bot_fe]"` to install the required packages to run quueue chat bot frontend; e.g.:
   - use default env params to run (default vad_recorder)
    ```
    ACHATBOT_PKG=1 RUN_OP=fe \
        REDIS_PASSWORD=$redis_pwd \
        REDIS_HOST=redis-14241.c256.us-east-1-2.ec2.redns.redis-cloud.com \
        REDIS_PORT=14241 \
        python -m achatbot.cmd.remote-queue-chat.generate_audio2audio > ~/.achatbot/log/fe_std_out.log
    ```
   - with wake word
    ```
    ACHATBOT_PKG=1 RUN_OP=fe \
        REDIS_PASSWORD=$redis_pwd \
        REDIS_HOST=redis-14241.c256.us-east-1-2.ec2.redns.redis-cloud.com \
        REDIS_PORT=14241 \
        RECORDER_TAG=wakeword_rms_recorder \
        python -m achatbot.cmd.remote-queue-chat.generate_audio2audio > ~/.achatbot/log/fe_std_out.log
    ```
   - default pyaudio player stream with tts tag out sample info(rate,channels..), e.g.: (be use tts_cosy_voice out stream info)
   ```
    ACHATBOT_PKG=1 RUN_OP=fe \
        REDIS_PASSWORD=$redis_pwd \
        REDIS_HOST=redis-14241.c256.us-east-1-2.ec2.redns.redis-cloud.com \
        REDIS_PORT=14241 \
        RUN_OP=fe \
        TTS_TAG=tts_cosy_voice \
        python -m achatbot.cmd.remote-queue-chat.generate_audio2audio > ~/.achatbot/log/fe_std_out.log
   ```
   remote_queue_chat_bot_be_worker in colab examples :
   <a href="https://colab.research.google.com/github/weedge/doraemon-nb/blob/main/chat_bot_gpu_worker.ipynb" target="_parent">
   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
   
   - sense_voice(asr) -> qwen (llm) -> cosy_voice (tts)

</details>

<details>
<summary>:full_moon: Run remote grpc tts speaker bot</summary>

## Run remote grpc tts speaker bot
1. run `pip install "achatbot[remote_grpc_tts_server]"` to install dependencies to run grpc tts speaker bot server; 
```
ACHATBOT_PKG=1 python -m achatbot.cmd.grpc.speaker.server.serve
```
2. run `pip install "achatbot[remote_grpc_tts_client]"` to install dependencies to run grpc tts speaker bot client; 
```
ACHATBOT_PKG=1 TTS_TAG=tts_edge IS_RELOAD=1 python -m src.cmd.grpc.speaker.client
ACHATBOT_PKG=1 TTS_TAG=tts_g IS_RELOAD=1 python -m src.cmd.grpc.speaker.client
ACHATBOT_PKG=1 TTS_TAG=tts_coqui IS_RELOAD=1 python -m src.cmd.grpc.speaker.client
ACHATBOT_PKG=1 TTS_TAG=tts_chat IS_RELOAD=1 python -m src.cmd.grpc.speaker.client
ACHATBOT_PKG=1 TTS_TAG=tts_cosy_voice IS_RELOAD=1 python -m src.cmd.grpc.speaker.client
ACHATBOT_PKG=1 TTS_TAG=tts_fishspeech IS_RELOAD=1 python -m src.cmd.grpc.speaker.client
ACHATBOT_PKG=1 TTS_TAG=tts_f5 IS_RELOAD=1 python -m src.cmd.grpc.speaker.client
ACHATBOT_PKG=1 TTS_TAG=tts_openvoicev2 IS_RELOAD=1 python -m src.cmd.grpc.speaker.client
ACHATBOT_PKG=1 TTS_TAG=tts_kokoro IS_RELOAD=1 python -m src.cmd.grpc.speaker.client
ACHATBOT_PKG=1 TTS_TAG=tts_onnx_kokoro IS_RELOAD=1 KOKORO_ESPEAK_NG_LIB_PATH=/usr/local/lib/libespeak-ng.1.dylib KOKORO_LANGUAGE=cmn python -m src.cmd.grpc.speaker.client
ACHATBOT_PKG=1 TTS_TAG=tts_cosy_voice2 \
    COSY_VOICE_MODELS_DIR=./models/FunAudioLLM/CosyVoice2-0.5B \
    COSY_VOICE_REFERENCE_AUDIO_PATH=./test/audio_files/asr_example_zh.wav \
    IS_RELOAD=1 python -m src.cmd.grpc.speaker.client
```
</details>

<details>

<summary>:video_camera: Multimodal Interaction</summary>

# Multimodal Interaction
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
- stream-ocr (realtime-object-detection)

## more
- Embodied Intelligence: Robots that touch the world, perceive and move
</details>

# License

achatbot is released under the [BSD 3 license](LICENSE). (Additional code in this distribution is covered by the MIT and Apache Open Source
licenses.) However you may have other legal obligations that govern your use of content, such as the terms of service for third-party models.
