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

  - [x] pipe(UNIX socket)
  - [ ] TCP/IP socket
  - [x] gRPC
  - queue
    - [x] redis
    - [ ] zmq
    - [ ] rocketmq

- chat bot processors: 

  - aggreators(llm use, assistant message), 
  - ai_frameworks
    - [x] [langchain](https://www.langchain.com/): RAG
    - [ ] [llamaindex](https://www.llamaindex.ai/): RAG
    - [ ] [autoagen](https://github.com/microsoft/autogen): multi Agents
  - realtime voice inference(RTVI),
  - transport: 
    - WebRTC:
      - [x] **[daily](https://github.com/ai-bot-pro/achatbot/blob/main/src/transports/daily.py)**: audio, video(image)
      - [x] **[livekit](https://github.com/ai-bot-pro/achatbot/blob/main/src/transports/livekit.py)**: audio, video(image)
      - [x] **[agora](https://github.com/ai-bot-pro/achatbot/blob/main/src/transports/agora.py)**: audio, video(image)
      - [x] **[small_webrtc](https://github.com/ai-bot-pro/achatbot/blob/main/src/transports/small_webrtc.py)**: audio, video(image)
    - WebSocket:
      - [x] [Websocket server](https://github.com/ai-bot-pro/achatbot/blob/main/src/transports/websocket_server.py)
      - [x] [fastapi Websocket server](https://github.com/ai-bot-pro/achatbot/blob/main/src/transports/fastapi_websocket_server.py)
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
      - [x] llm_vllm_vision_skyworkr1v
      - [x] llm_vllm_deepseek_ocr, llm_office_vllm_deepseek_ocr
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
      - [x] llm_transformers_manual_vision_skyworkr1v
      - [x] llm_transformers_manual_vision_deepseek_ocr
      - [x] llm_transformers_manual_image_janus_flow
      - [x] llm_transformers_manual_vision_janus
      - [x] llm_transformers_manual_image_janus
      - [x] llm_transformers_manual_speech_llasa
      - [x] llm_transformers_manual_speech_step
      - [x] llm_transformers_manual_voice_glm
      - [x] llm_transformers_manual_vision_voice_minicpmo, llm_transformers_manual_voice_minicpmo,llm_transformers_manual_audio_minicpmo,llm_transformers_manual_text_speech_minicpmo,llm_transformers_manual_instruct_speech_minicpmo,llm_transformers_manual_vision_minicpmo
      - [x] llm_transformers_manual_qwen2_5omni, llm_transformers_manual_qwen2_5omni_audio_asr,llm_transformers_manual_qwen2_5omni_vision,llm_transformers_manual_qwen2_5omni_speech,llm_transformers_manual_qwen2_5omni_vision_voice,llm_transformers_manual_qwen2_5omni_text_voice,llm_transformers_manual_qwen2_5omni_audio_voice
      - [x] llm_transformers_manual_kimi_voice,llm_transformers_manual_kimi_audio_asr,llm_transformers_manual_kimi_text_voice
      - [x] llm_transformers_manual_vita_text llm_transformers_manual_vita_audio_asr llm_transformers_manual_vita_tts llm_transformers_manual_vita_text_voice llm_transformers_manual_vita_voice
      - [x] llm_transformers_manual_phi4_vision_speech,llm_transformers_manual_phi4_audio_asr,llm_transformers_manual_phi4_audio_translation,llm_transformers_manual_phi4_vision,llm_transformers_manual_phi4_audio_chat
      - [x] llm_transformers_manual_vision_speech_gemma3n,llm_transformers_manual_vision_gemma3n,llm_transformers_manual_gemma3n_audio_asr,llm_transformers_manual_gemma3n_audio_translation
      - [x] llm_transformers_manual_voice_step2, llm_vllm_client_step_audio2, llm_vllm_client_step_audio2_mock
      - [x] llm_transformers_manual_qwen3omni, llm_transformers_manual_qwen3omni_vision_voice
  - remote api llm: personal-ai(like openai api, other ai provider)

- AI modules:

  - functions:
    - [x] search: search,search1,serper
    - [x] weather: openweathermap
  - punctuation: 
    - [x] punc_ct_tranformerm, punc_ct_tranformer_offline, punc_ct_tranformer_onnx, punc_ct_tranformer_onnx_offline
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
      - [x] gemma3n_asr (matformer)
    - [x] asr_live:
      - [x] asr_streaming_sensevoice
    - [x] speech enhancement:
      - [x] enhancer_ans_rnnoise
      - [x] enhancer_ans_dfsmn
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
      - [x] ten_vad_analyzer
    - [x] turn_analyzer
      - [x] v2_smart_turn_analyzer
  - vision
    - [x] OCR(*Optical Character Recognition*):
      - [ ] [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
      - [x] [GOT](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)(*the General OCR Theory*)
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