# achatbot
[![PyPI](https://img.shields.io/pypi/v/achatbot)](https://pypi.org/project/achatbot/)
<a href="https://app.commanddash.io/agent/github_ai-bot-pro_achatbot"><img src="https://img.shields.io/badge/AI-Code%20Agent-EB9FDA"></a>

achatbot factory, create chat bots with llm(tools), asr, tts, vad, ocr, detect object etc..

# Project Structure
![project-structure](https://github.com/user-attachments/assets/5bf7cebb-e590-4718-a78a-6b0c0b36ea28)

# Feature
- demo
  
  - [podcast](https://github.com/ai-bot-pro/achatbot/blob/main/demo/content_parser_tts.py)
  
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
    # https://www.apple.com/ios/ios-18/pdf/iOS_18_All_New_Features_Sept_2024.pdf
    python -m demo.content_parser_tts instruct-content-tts \
        "/Users/wuyong/Desktop/iOS_18_All_New_Features_Sept_2024.pdf"
    
    python -m demo.content_parser_tts instruct-content-tts \
        --role-tts-voices zh-CN-YunjianNeural \
        --role-tts-voices zh-CN-XiaoxiaoNeural \
        --language zh \
        "/Users/wuyong/Desktop/iOS_18_All_New_Features_Sept_2024.pdf"
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
    - webRTC: 
      - [x] **[daily](https://github.com/ai-bot-pro/achatbot/blob/main/src/transports/daily.py)**: audio, video(image)
      - [x] **[livekit](https://github.com/ai-bot-pro/achatbot/blob/main/src/transports/livekit.py)**: audio, video(image)
      - [ ] **[agora-audio](https://github.com/AgoraIO/agora-realtime-ai-api)**
    - [x] [Websocket server](https://github.com/ai-bot-pro/achatbot/blob/main/src/transports/websocket_server.py)
  - ai processor: llm, tts, asr etc..
    - llm_processor:
      - [x] [openai](https://github.com/ai-bot-pro/achatbot/blob/main/test/integration/processors/test_openai_llm_processor.py)(use openai sdk)
      - [x] [google gemini](https://github.com/ai-bot-pro/achatbot/blob/main/test/integration/processors/test_google_llm_processor.py)(use google-generativeai sdk)
      - [x] [litellm](https://github.com/ai-bot-pro/achatbot/blob/main/test/integration/processors/test_litellm_processor.py)(use openai input/output format proxy sdk) 

- core module:
  - local llm: 
    - [x] llama-cpp (support text,vision with function-call model)
    - [x] transformers(manual, pipeline) (support text,vision:🦙,Qwen2-vl,Molmo with function-call model)
    - [ ] mlx_lm 
  - remote api llm: personal-ai(like openai api, other ai provider)

- AI modules:
  - functions:
    - [x] search: search,search1,serper
    - [x] weather: openweathermap
  - speech:
    - [x] asr: sense_voice_asr, whisper_asr, whisper_timestamped_asr, whisper_faster_asr, whisper_transformers_asr, whisper_mlx_asr, lightning_whisper_mlx_asr(!TODO), whisper_groq_asr
    - [x] audio_stream: daily_room_audio_stream(in/out), pyaudio_stream(in/out)
    - [x] detector: porcupine_wakeword,pyannote_vad,webrtc_vad,silero_vad,webrtc_silero_vad
    - [x] player: stream_player
    - [x] recorder: rms_recorder, wakeword_rms_recorder, vad_recorder, wakeword_vad_recorder
    - [x] tts: tts_chat,tts_coqui,tts_cosy_voice,tts_edge,tts_g
    - [x] vad_analyzer: daily_webrtc_vad_analyzer,silero_vad_analyzer
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
  - aws lambda + api Gateway
  - docker -> k8s/k3s
  - etc...

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


## Server Deploy (CD)
- [x] [deploy/modal](https://github.com/ai-bot-pro/achatbot/tree/main/deploy/modal)(KISS) 👍🏻 
- [x] [deploy/leptonai](https://github.com/ai-bot-pro/achatbot/tree/main/deploy/leptonai)(KISS)👍🏻
- [x] [deploy/cerebrium/fastapi-daily-chat-bot](https://github.com/ai-bot-pro/achatbot/tree/main/deploy/cerebrium/fastapi-daily-chat-bot) :)
- [x] [deploy/aws/fastapi-daily-chat-bot](https://github.com/ai-bot-pro/achatbot/tree/main/deploy/aws/fastapi-daily-chat-bot) :|
- [x] [deploy/docker/fastapi-daily-chat-bot](https://github.com/ai-bot-pro/achatbot/tree/main/deploy/docker) 🏃


# Install
> [!NOTE]
> `python --version` >=3.10 with [asyncio-task](https://docs.python.org/3.11/library/asyncio-task.html)

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

#  Run chat bots
## Run chat bots with colab notebook

|                           Chat Bot                           | optional-dependencies                                        | Colab                                                        | Device                                                       | Pipeline Desc                                                |
| :----------------------------------------------------------: | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [daily_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/daily_bot.py)<br />[livekit_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/livekit_bot.py) | e.g.:<br />daily_room_audio_stream \| livekit_room_audio_stream,<br />sense_voice_asr,<br />groq \| together api llm(text), <br />tts_edge | <a href="https://github.com/weedge/doraemon-nb/blob/main/webrtc_audio_bot.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | CPU (free, 2 cores)                                          | e.g.:<br />daily \| livekit room in stream<br />-> silero (vad)<br />-> sense_voice (asr) <br />-> groq \| together  (llm) <br />-> edge (tts)<br />-> daily \| livekit room out stream |
| [generate_audio2audio](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/remote-queue-chat/generate_audio2audio.py) | remote_queue_chat_bot_be_worker                              | <a href="https://github.com/weedge/doraemon-nb/blob/main/chat_bot_gpu_worker.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | T4(free)                                                     | e.g.:<br />pyaudio in stream<br />-> silero (vad)<br />-> sense_voice (asr) <br />-> qwen (llm) <br />-> cosy_voice (tts)<br />-> pyaudio out stream |
| [daily_describe_vision_tools_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/vision/daily_describe_vision_tools_bot.py)<br />[livekit_describe_vision_tools_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/vision/livekit_describe_vision_tools_bot.py) | e.g.:<br />daily_room_audio_stream \|livekit_room_audio_stream<br />deepgram_asr,<br />goole_gemini,<br />tts_edge | <a href="https://github.com/weedge/doraemon-nb/blob/main/achatbot_describe_vision_tools_bot.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | CPU(free, 2 cores)                                           | e.g.:<br />daily \|livekit room in stream<br />-> silero (vad)<br />-> deepgram (asr) <br />-> google gemini  <br />-> edge (tts)<br />-> daily \|livekit room out stream |
| [daily_describe_vision_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/vision/daily_describe_vision_bot.py)<br />[livekit_describe_vision_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/vision/livekit_describe_vision_bot.py) | e.g.:<br />daily_room_audio_stream \| livekit_room_audio_stream<br />sense_voice_asr,<br />llm_transformers_manual_vision_qwen,<br />tts_edge | <a href="https://github.com/weedge/doraemon-nb/blob/main/achatbot_vision_qwen_vl.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | - Qwen2-VL-2B-Instruct<br /> T4(free)<br />- Qwen2-VL-7B-Instruct<br />L4<br />- Llama-3.2-11B-Vision-Instruct<br />L4<br />- allenai/Molmo-7B-D-0924<br />A100 | e.g.:<br />daily \| livekit room in stream<br />-> silero (vad)<br />-> sense_voice (asr) <br />-> qwen-vl (llm) <br />-> edge (tts)<br />-> daily \| livekit room out stream |
| [daily_chat_vision_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/vision/daily_chat_vision_bot.py)<br />[livekit_chat_vision_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/vision/livekit_chat_vision_bot.py) | e.g.:<br />daily_room_audio_stream \|livekit_room_audio_stream<br />sense_voice_asr,<br />llm_transformers_manual_vision_qwen,<br />tts_edge | <a href="https://github.com/weedge/doraemon-nb/blob/main/achatbot_daily_chat_vision_bot.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | - Qwen2-VL-2B-Instruct<br /> T4(free)<br />- Qwen2-VL-7B-Instruct<br />L4<br />- Ll<br/>ama-3.2-11B-Vision-Instruct<br />L4<br />- allenai/Molmo-7B-D-0924<br />A100 | e.g.:<br />daily \| livekit room in stream<br />-> silero (vad)<br />-> sense_voice (asr) <br />-> llm answer guide qwen-vl (llm) <br />-> edge (tts)<br />-> daily \| livekit room out stream |
| [daily_chat_tools_vision_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/vision/daily_chat_tools_vision_bot.py)<br />[livekit_chat_tools_vision_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/vision/livekit_chat_tools_vision_bot.py) | e.g.:<br />daily_room_audio_stream \| livekit_room_audio_stream<br />sense_voice_asr,<br />groq api llm(text), <br />tools:<br />- llm_transformers_manual_vision_qwen,<br />tts_edge | <a href="https://github.com/weedge/doraemon-nb/blob/main/achatbot_daily_chat_tools_vision_bot.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | - Qwen2-VL-2B-Instruct<br<br/> /> T4(free)<br />- Qwen2-VL-7B-Instruct<br />L4<br />- Llama-3.2-11B-Vision-Instruct<br />L4 <br />- allenai/Molmo-7B-D-0924<br />A100 | e.g.:<br />daily \| livekit room in stream<br />-> silero (vad)<br />-> sense_voice (asr) <br />->llm with tools qwen-vl  <br />-> edge (tts)<br />-> daily \| livekit room out stream |
| [daily_annotate_vision_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/vision/daily_annotate_vision_bot.py)<br />[livekit_annotate_vision_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/vision/livekit_annotate_vision_bot.py) | e.g.:<br />daily_room_audio_stream \| livekit_room_audio_stream<br />vision_yolo_detector<br />tts_edge | <a href="https://github.com/weedge/doraemon-nb/blob/main/daily_annotate_vision_bot.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | T4(free)                                                     | e.g.:<br />daily \| livekit room in stream<br />vision_yolo_detector<br />-> edge (tts)<br />-> daily \| livekit room out stream |
| [daily_detect_vision_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/vision/daily_detect_vision_bot.py)<br />[livekit_detect_vision_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/vision/livekit_detect_vision_bot.py) | e.g.:<br />daily_room_audio_stream \| livekit_room_audio_stream<br />vision_yolo_detector<br />tts_edge | <a href="https://github.com/weedge/doraemon-nb/blob/main/daily_detect_vision_bot.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | T4(free)                                                     | e.g.:<br />daily \| livekit room in stream<br />vision_yolo_detector<br />-> edge (tts)<br />-> daily \| livekit room out stream |
| [daily_ocr_vision_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/vision/daily_ocr_vision_bot.py)<br />[livekit_ocr_vision_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/vision/livekit_ocr_vision_bot.py) | e.g.:<br />daily_room_audio_stream \| livekit_room_audio_stream<br />sense_voice_asr,<br />vision_transformers_got_ocr<br />tts_edge | <a href="https://github.com/weedge/doraemon-nb/blob/main/daily_ocr_vision_bot.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | T4(free)                                                     | e.g.:<br />daily \| livekit room in stream<br />-> silero (vad)<br />-> sense_voice (asr) <br />vision_transformers_got_ocr<br />-> edge (tts)<br />-> daily \| livekit room out stream |
| [daily_month_narration_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/image/daily_month_narration_bot.py) | e.g.:<br />daily_room_audio_stream <br />groq \|together api llm(text),<br />hf_sd, together api (image)<br />tts_edge | <a href="https://github.com/weedge/doraemon-nb/blob/main/achatbot_daily_month_narration_bot.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | when use sd model with diffusers<br />T4(free) cpu+cuda (slow)<br />L4 cpu+cuda<br/>A100 all cuda<br /> | e.g.:<br />daily room in stream<br />-> together  (llm) <br />-> hf sd gen image model<br />-> edge (tts)<br />-> daily  room out stream |
| [daily_storytelling_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/image/storytelling/daily_bot.py) | e.g.:<br />daily_room_audio_stream <br />groq \|together api llm(text),<br />hf_sd, together api (image)<br />tts_edge | <a href="https://github.com/weedge/doraemon-nb/blob/main/achatbot_daily_storytelling_bot.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | cpu (2 cores)<br />when use sd model with diffusers<br />T4(free) cpu+cuda (slow)<br />L4 cpu+cuda<br/>A100 all cuda<br /> | e.g.:<br />daily room in stream<br />-> together  (llm) <br />-> hf sd gen image model<br />-> edge (tts)<br />-> daily  room out stream |
| [websocket_server_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/websocket_server_bot.py)<br />[fastapi_websocket_server_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/fastapi_websocket_server_bot.py) | e.g.:<br /> websocket_server<br />sense_voice_asr,<br />groq \|together api llm(text),<br />tts_edge | <a href="https://github.com/weedge/doraemon-nb/blob/main/achatbot_websocket_server_bot.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | cpu(2 cores)                                                 | e.g.:<br />websocket protocol  in stream<br />-> silero (vad)<br />-> sense_voice (asr) <br />-> together  (llm) <br />-> edge (tts)<br />-> websocket protocol out stream |
| [daily_natural_conversation_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/nlp/daily_natural_conversation_bot.py) | e.g.:<br /> daily_room_audio_stream<br />sense_voice_asr,<br />groq \|together api llm(NLP task),<br />gemini-1.5-flash (chat)<br />tts_edge | <a href="https://github.com/weedge/doraemon-nb/blob/main/achat_natural_conversation_bot.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | cpu(2 cores)                                                 | e.g.:<br />daily room in stream<br />-> together  (llm NLP task) <br />->  gemini-1.5-flash model (chat)<br />-> edge (tts)<br />-> daily  room out stream |
| [fastapi_websocket_moshi_bot](https://github.com/ai-bot-pro/achatbot/blob/main/src/cmd/bots/voice/fastapi_websocket_moshi_bot.py) | e.g.:<br /> websocket_server<br />moshi opus stream voice llm<br /> | <a href="https://github.com/weedge/doraemon-nb/blob/main/achatbot_moshi_bot.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | L4                                                           | websocket protocol  in stream<br />-> silero (vad)<br />-> moshi opus stream voice llm<br />-> websocket protocol out stream |
|                                                              |                                                              |                                                              |                                                              |                                                              |
|                                                              |                                                              |                                                              |                                                              |                                                              |



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
   
   - sense_voice(asr) -> qwen (llm) -> cosy_voice (tts): 

## Run remote grpc tts speaker bot
1. run `pip install "achatbot[remote_grpc_tts_server]"` to install dependencies to run grpc tts speaker bot server; 
```
ACHATBOT_PKG=1 python -m achatbot.cmd.grpc.speaker.server.serve
```
2. run `pip install "achatbot[remote_grpc_tts_client]"` to install dependencies to run grpc tts speaker bot client; 
```
ACHATBOT_PKG=1 TTS_TAG=tts_edge IS_RELOAD=1 python -m achatbot.cmd.grpc.speaker.client
ACHATBOT_PKG=1 TTS_TAG=tts_g IS_RELOAD=1 python -m achatbot.cmd.grpc.speaker.client
ACHATBOT_PKG=1 TTS_TAG=tts_coqui IS_RELOAD=1 python -m achatbot.cmd.grpc.speaker.client
ACHATBOT_PKG=1 TTS_TAG=tts_chat IS_RELOAD=1 python -m achatbot.cmd.grpc.speaker.client
ACHATBOT_PKG=1 TTS_TAG=tts_cosy_voice IS_RELOAD=1 python -m achatbot.cmd.grpc.speaker.client
```


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

# License

torchchat is released under the [BSD 3 license](LICENSE). (Additional code in this distribution is covered by the MIT and Apache Open Source
licenses.) However you may have other legal obligations that govern your use of content, such as the terms of service for third-party models.
