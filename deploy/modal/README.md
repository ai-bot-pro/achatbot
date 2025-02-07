# intro
the Starter plan with $30.00 included compute credits per month, for development, so nice~

# modal
> [!NOTE] 
> in deploy/modal dir to run shell
```shell
pip install modal 

modal setup

# create worksapce achatbot environment
modal environment create achatbot

# Created Volume 'bot_config' in environment 'achatbot'
# for remote local bot config
modal volume create -e achatbot bot_config
```

## modal run 
- run serverless remote function call like local to dev, so nice~ (auto schedule)
- crob jobs to run schedule task

## modal serve
- use fastapi(ASGI modal app) to run serverless web service
- web server endpoint: `https://{profile}-{environment}--{app_name}-{class_name}-app-dev.modal.run` (if app_name + class_name is so long, hash tag name to replace)
```shell
# just a simple serve, don't to run bot
modal serve -e achatbot src/fastapi_serve.py
```

### webrtc_audio_bot
- run webrtc_audio_bot serve
```shell
# bot serve e.g.:
# webrtc_audio_bot serve on default pip image
# need create .env.example to modal Secrets
IMAGE_NAME=default modal serve -e achatbot src/fastapi_webrtc_audio_bot_serve.py
```
- curl api to run chat room bot with webrtc (daily/livekit/agora)
`https://weedge-achatbot--fastapi-webrtc-audio-bot-srv-app-dev.modal.run/docs` see api docs
```shell
curl --location 'https://weedge-achatbot--fastapi-webrtc-audio-bot-srv-app-dev.modal.run/bot_join/chat-room/DailyBot' \
--header 'Authorization: Bearer xxx' \
--header 'Content-Type: application/json' \
--data '{
    "chat_bot_name": "DailyBot",
    "room_name": "chat-room",
    "room_url": "",
    "token": "",
    "room_manager": {
        "tag": "daily_room",
        "args": {
            "privacy": "public"
        }
    },
    "services": {
        "pipeline": "achatbot",
        "vad": "silero",
        "asr": "sense_voice",
        "llm": "groq",
        "tts": "edge"
    },
    "config": {
        "vad": {
            "tag": "silero_vad_analyzer",
            "args": {
                "stop_secs": 0.7
            }
        },
        "asr": {
            "tag": "sense_voice_asr",
            "args": {
                "language": "zn",
                "model_name_or_path": "/root/.achatbot/models/FunAudioLLM/SenseVoiceSmall"
            }
        },
        "llm": {
            "tag": "openai_llm_processor",
            "base_url": "https://api.together.xyz/v1",
            "model": "Qwen/Qwen2-72B-Instruct",
            "language": "zh",
            "messages": [
                {
                    "role": "system",
                    "content": "你是一名叫奥利给的智能助理。保持回答简短和清晰。请用中文回答。"
                }
            ]
        },
        "tts": {
            "tag": "tts_edge",
            "args": {
                "voice_name": "zh-CN-YunjianNeural",
                "language": "zh",
                "gender": "Male"
            }
        }
    },
    "config_list": []
}'
```


### ws_moshi_voice_bot
- run ws_moshi_voice_bot serve
```shell
# put local config to modal volume bot_config / dir
modal volume put -e achatbot bot_config  ../../config/bots/fastapi_websocket_moshi_voice_bot.json / -f
# ws_moshi_voice_bot serve on default pip image
IMAGE_NAME=default modal serve -e achatbot src/fastapi_ws_moshi_voice_bot_serve.py
```
- run moshi_opus_stream_ws_pb_client to chat with moshi in CLI
```shell
# run moshi_opus_stream_ws_pb_client to chat with moshi in CLI
python -m achatbot.cmd.websocket.moshi_opus_stream_ws_pb_client --endpoint https://weedge-achatbot--fastapi-ws-moshi-voice-bot-srv-app-dev.modal.run/
```
> [!TIPS] 
> process frame(size:1920, 25ms; sample_rate:24000/s, sample_width:2, channels:1) cost: 53.0ms 
> (opus audio format) speech mimi encoder encode -> gen lm(moshi) -> text|speech tokens -> text BPE tokenizer decode|speech mimi decoder decode -> text|opus audio format with pb serialize

### webrtc_vision_bot
- run webrtc_vision_bot serve with task queue(redis)
```shell
# webrtc_vision_bot serve on default pip image
IMAGE_NAME=default IMAGE_CONCURRENT_CN=100 modal serve -e achatbot src/fastapi_webrtc_vision_bot_serve.py
# webrtc_vision_bot serve on qwen vision llm pip image
IMAGE_NAME=qwen IMAGE_CONCURRENT_CN=1 IMAGE_GPU=T4 modal serve -e achatbot src/fastapi_webrtc_vision_bot_serve.py
IMAGE_NAME=qwen IMAGE_CONCURRENT_CN=1 IMAGE_GPU=L4 LLM_MODEL_NAME_OR_PATH=Qwen/Qwen2-VL-7B-Instruct modal serve -e achatbot src/fastapi_webrtc_vision_bot_serve.py
# webrtc_vision_bot serve on llama vision llm pip image
IMAGE_NAME=llama IMAGE_CONCURRENT_CN=1 IMAGE_GPU=L4 modal serve -e achatbot src/fastapi_webrtc_vision_bot_serve.py
# webrtc_vision_bot serve on janus vision llm pip image
# https://www.nvidia.com/en-us/data-center/tesla-t4/ 16G
IMAGE_NAME=janus IMAGE_CONCURRENT_CN=1 IMAGE_GPU=T4 modal serve -e achatbot src/fastapi_webrtc_vision_bot_serve.py
IMAGE_NAME=janus IMAGE_CONCURRENT_CN=1 IMAGE_GPU=L4 LLM_MODEL_NAME_OR_PATH=deepseek-ai/Janus-Pro-7B modal serve -e achatbot src/fastapi_webrtc_vision_bot_serve.py
# webrtc_vision_bot serve on deepseekvl2 vision llm pip image
# https://www.nvidia.com/en-us/data-center/l4/ 24GB 
IMAGE_NAME=deepseekvl2 IMAGE_CONCURRENT_CN=1 IMAGE_GPU=L4 modal serve -e achatbot src/fastapi_webrtc_vision_bot_serve.py
# https://www.nvidia.com/en-us/data-center/l40s/ 48G
IMAGE_NAME=deepseekvl2 IMAGE_CONCURRENT_CN=1 IMAGE_GPU=L40S LLM_MODEL_NAME_OR_PATH=deepseek-ai/deepseek-vl2-small modal serve -e achatbot src/fastapi_webrtc_vision_bot_serve.py
# https://www.nvidia.com/en-us/data-center/a100/ 40G
IMAGE_NAME=deepseekvl2 IMAGE_CONCURRENT_CN=1 IMAGE_GPU=A100 LLM_MODEL_NAME_OR_PATH=deepseek-ai/deepseek-vl2-small modal serve -e achatbot src/fastapi_webrtc_vision_bot_serve.py
# https://www.nvidia.com/en-us/data-center/a100/ 80G
IMAGE_NAME=deepseekvl2 IMAGE_CONCURRENT_CN=1 IMAGE_GPU=A100-80GB LLM_MODEL_NAME_OR_PATH=deepseek-ai/deepseek-vl2 modal serve -e achatbot src/fastapi_webrtc_vision_bot_serve.py
# https://www.nvidia.com/en-us/data-center/h100/ 80G
IMAGE_NAME=deepseekvl2 IMAGE_CONCURRENT_CN=1 IMAGE_GPU=H100 LLM_MODEL_NAME_OR_PATH=deepseek-ai/deepseek-vl2 modal serve -e achatbot src/fastapi_webrtc_vision_bot_serve.py
```
- curl api to run chat room bot with webrtc (daily/livekit/agora)
```shell
curl --location 'https://weedge-achatbot--fastapi-webrtc-vision-qwen-bot-srv-app-dev.modal.run/bot_join/chat-room/DailyDescribeVisionBot' \
--header 'Content-Type: application/json' \
--data '{
  "chat_bot_name": "DailyDescribeVisionBot",
  "room_name": "chat-bot",
  "room_url": "",
  "token": "",
  "services": {
    "pipeline": "achatbot",
    "vad": "silero",
    "asr": "sense_voice",
    "llm": "transformers_manual_vision_qwen",
    "tts": "edge"
  },
  "config": {
    "vad": {
      "tag": "silero_vad_analyzer",
      "args": { "stop_secs": 0.7 }
    },
    "asr": {
      "tag": "sense_voice_asr",
      "args": {
        "language": "zn",
        "model_name_or_path": "/root/.achatbot/models/FunAudioLLM/SenseVoiceSmall"
      }
    },
    "llm": {
      "tag":"llm_transformers_manual_vision_qwen",
      "args":{
        "lm_device":"cuda",
        "lm_model_name_or_path":"/root/.achatbot/models/Qwen/Qwen2-VL-2B-Instruct",
        "chat_history_size": 0,
        "init_chat_prompt":"请用中文交流",
        "model_type":"chat_completion"
      },
      "language": "zh"
    },
    "tts": {
      "tag": "tts_edge",
      "args": {
        "voice_name": "zh-CN-YunjianNeural",
        "language": "zh",
        "gender": "Male"
      }
    }
  },
  "config_list": []
}'
```

### webrtc_glm_voice_bot
- run webrtc_glm_voice_bot serve with task queue(redis)
```shell
# webrtc_audio_bot serve on default pip image
# need create .env.example to modal Secrets for webrtc key
IMAGE_NAME=default IMAGE_CONCURRENT_CN=1 IMAGE_GPU=T4 modal serve -e achatbot src/fastapi_webrtc_glm_voice_bot_serve.py
```
- curl api to run chat room bot with webrtc (daily/livekit/agora)
```shell
curl --location 'https://weedge-achatbot--fastapi-webrtc-glm-voice-bot-srv-app-dev.modal.run/bot_join/chat-room/DailyGLMVoiceBot' \
--header 'Content-Type: application/json' \
--data '{
  "chat_bot_name": "DailyGLMVoiceBot",
  "room_name": "chat-room",
  "room_url": "",
  "token": "",
  "room_manager": {
    "tag": "daily_room",
    "args": {
      "privacy": "public"
    }
  },
  "services": {
    "pipeline": "achatbot",
    "vad": "silero",
    "voice_llm": "glm"
  },
  "config": {
    "vad": {
      "tag": "silero_vad_analyzer",
      "args": { "stop_secs": 0.7 }
    },
    "voice_llm": {
      "tag": "glm_voice_processor",
      "args": {
        "lm_gen_args": {
          "temperature": 0.2,
          "top_p": 0.8,
          "max_new_token": 2000
        },
        "voice_out_args": {
          "audio_sample_rate": 22050,
          "audio_channels": 1
        },
        "voice_tokenizer_path": "/root/.achatbot/models/THUDM/glm-4-voice-tokenizer",
        "model_path": "/root/.achatbot/models/THUDM/glm-4-voice-9b",
        "voice_decoder_path": "/root/.achatbot/models/THUDM/glm-4-voice-decoder",
        "torch_dtype": "auto",
        "bnb_quant_type": "int4",
        "device": "cuda"
      }
    }
  },
  "config_list": []
}'

curl --location 'https://weedge-achatbot--fastapi-webrtc-glm-voice-bot-srv-app-dev.modal.run/bot_join/chat-room/DailyAsrGLMVoiceBot' \
--header 'Content-Type: application/json' \
--data '{
  "chat_bot_name": "DailyAsrGLMVoiceBot",
  "room_name": "chat-room",
  "room_url": "",
  "token": "",
  "room_manager": {
    "tag": "daily_room",
    "args": {
      "privacy": "public"
    }
  },
  "services": {
    "pipeline": "achatbot",
    "vad": "silero",
    "asr": "sense_voice",
    "voice_llm": "glm"
  },
  "config": {
    "vad": {
      "tag": "silero_vad_analyzer",
      "args": { "stop_secs": 0.7 }
    },
    "asr": {
      "tag": "sense_voice_asr",
      "args": {
        "language": "zn",
        "model_name_or_path": "/root/.achatbot/models/FunAudioLLM/SenseVoiceSmall"
      }
    },
    "voice_llm": {
      "tag": "text_glm_voice_processor",
      "args": {
        "lm_gen_args": {
          "temperature": 0.2,
          "top_p": 0.8,
          "max_new_token": 2000
        },
        "voice_out_args": {
          "audio_sample_rate": 22050,
          "audio_channels": 1
        },
        "voice_tokenizer_path": "/root/.achatbot/models/THUDM/glm-4-voice-tokenizer",
        "model_path": "/root/.achatbot/models/THUDM/glm-4-voice-9b",
        "voice_decoder_path": "/root/.achatbot/models/THUDM/glm-4-voice-decoder",
        "device": "cuda"
      }
    }
  },
  "config_list": []
}'
```

### webrtc_freeze_omni_voice_bot
- run webrtc_freeze_omni_voice_bot serve with task queue(redis)
```shell
# webrtc_audio_bot serve on default pip image
# need create .env.example to modal Secrets for webrtc key
IMAGE_NAME=default IMAGE_CONCURRENT_CN=1 IMAGE_GPU=L4 modal serve -e achatbot src/fastapi_webrtc_freeze_omni_voice_bot_serve.py
```
- curl api to run chat room bot with webrtc (daily/livekit/agora)
```shell
curl --location 'https://weedge-achatbot--fastapi-webrtc-freeze-omni-voice-bo-4b7458-dev.modal.run/bot_join/chat-room/DailyFreezeOmniVoiceBot' \
--header 'Content-Type: application/json' \
--data '{
  "chat_bot_name": "DailyFreezeOmniVoiceBot",
  "room_name": "chat-room",
  "room_url": "",
  "token": "",
  "room_manager": {
    "tag": "daily_room",
    "args": {
      "privacy": "public"
    }
  },
  "services": {
    "pipeline": "achatbot",
    "vad": "silero",
    "voice_llm": "freeze_omni"
  },
  "config": {
    "vad": {
      "tag": "silero_vad_analyzer",
      "args": { "stop_secs": 0.7 }
    },
    "voice_llm": {
      "tag": "freeze_omni_voice_processor",
      "args": {
        "args": {
          "llm_path": "/root/.achatbot/models/Qwen/Qwen2-7B-Instruct",
          "model_path": "/root/.achatbot/models/VITA-MLLM/Freeze-Omni/checkpoints"
        }
      }
    }
  },
  "config_list": []
}'
```

## modal deploy (online)
- deploy webrtc_audio_bot serve
```shell
IMAGE_NAME=default modal deploy -e achatbot src/fastapi_webrtc_audio_bot_serve.py
```
endpoint: https://weedge-achatbot--fastapi-webrtc-audio-bot-srv-app.modal.run/

- deploy ws_moshi_voice_bot serve
```shell
IMAGE_NAME=default modal deploy -e achatbot src/fastapi_ws_moshi_voice_bot_serve.py
```
endpoint: https://weedge-achatbot--fastapi-ws-moshi-voice-bot-srv-app.modal.run

- deploy webrtc_vision_bot serve
```shell
IMAGE_NAME=default modal deploy -e achatbot src/fastapi_ws_moshi_voice_bot_serve.py
```
endpoint: https://weedge-achatbot--fastapi-webrtc-vision-bot-srv-app.modal.run/

```shell
IMAGE_NAME=qwen IMAGE_CONCURRENT_CN=1 IMAGE_GPU=T4 modal deploy -e achatbot src/fastapi_webrtc_vision_bot_serve.py
```
endpoint: https://weedge-achatbot--fastapi-webrtc-vision-qwen-bot-srv-app.modal.run/

```shell
IMAGE_NAME=llama IMAGE_CONCURRENT_CN=1 IMAGE_GPU=L4 modal serve -e achatbot src/fastapi_webrtc_vision_bot_serve.py
```
endpoint: https://weedge-achatbot--fastapi-webrtc-vision-llama-bot-srv-app.modal.run/

```shell
IMAGE_NAME=janus IMAGE_CONCURRENT_CN=1 IMAGE_GPU=T4 modal serve -e achatbot src/fastapi_webrtc_vision_bot_serve.py
```
endpoint: https://weedge-achatbot--fastapi-webrtc-vision-janus-bot-srv-app.modal.run/

```shell
IMAGE_NAME=deepseekvl2 IMAGE_CONCURRENT_CN=1 IMAGE_GPU=L4 modal serve -e achatbot src/fastapi_webrtc_vision_bot_serve.py
```
endpoint: https://weedge-achatbot--fastapi-webrtc-vision-deepseekvl2-bot-srv-app.modal.run/

- deploy webrtc_glm_voice_bot serve
```shell
IMAGE_NAME=default IMAGE_CONCURRENT_CN=1 IMAGE_GPU=T4 modal deploy -e achatbot src/fastapi_webrtc_glm_voice_bot_serve.py
```
endpoint: https://weedge-achatbot--fastapi-webrtc-glm-voice-bot-srv-app.modal.run/

- deploy webrtc_freeze_omni_voice_bot serve
```shell
IMAGE_NAME=default IMAGE_CONCURRENT_CN=1 IMAGE_GPU=L4 modal deploy -e achatbot src/fastapi_webrtc_freeze_omni_voice_bot_serve.py
```
endpoint: https://weedge-achatbot--fastapi-webrtc-freeze_omni-voice-bot-srv-app.modal.run/

# references (nice docs) 👍 @modal
- https://modal.com/docs/guide
- https://modal.com/docs/guide/gpu
- https://modal.com/docs/guide/cuda
- https://modal.com/docs/guide/volumes
- https://modal.com/docs/examples/basic_web
- https://modal.com/docs/guide/cron
- https://github.com/modal-labs/modal-examples