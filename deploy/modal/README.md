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

## dowload models and assets
- download models and assets to modal volume
  - modal run -e achatbot src/download_models.py --repo-id $repo_id
  - modal run -e achatbot src/download_assets.py --asset-urls $asset_urls

e.g.:
```
modal run -e achatbot src/download_models.py --repo-ids "Qwen/Qwen2.5-0.5B"
modal run -e achatbot src/download_models.py --repo-ids "FunAudioLLM/SenseVoiceSmall"

modal run -e achatbot src/download_assets.py --asset-urls "https://raw.githubusercontent.com/bytedance/MegaTTS3/refs/heads/main/tts/utils/text_utils/dict.json"

modal run -e achatbot src/download_assets.py --asset-urls "https://raw.githubusercontent.com/ai-bot-pro/achatbot/refs/heads/main/src/cmd/bots/image/storytelling/assets/book1.png,https://raw.githubusercontent.com/ai-bot-pro/achatbot/refs/heads/main/src/cmd/bots/image/storytelling/assets/book2.png,https://raw.githubusercontent.com/ai-bot-pro/achatbot/refs/heads/main/src/cmd/bots/image/storytelling/assets/ding.wav,https://raw.githubusercontent.com/ai-bot-pro/achatbot/refs/heads/main/src/cmd/bots/image/storytelling/assets/listening.wav,https://raw.githubusercontent.com/ai-bot-pro/achatbot/refs/heads/main/src/cmd/bots/image/storytelling/assets/talking.wav"
```

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

# put local config to modal volume bot_config / dir
modal volume put -e achatbot bot_config ../../config/bots/fastapi_websocket_moshi_hibiki_voice_bot.json / -f
# ws_moshi_hibiki_voice_bot serve on default pip image
IMAGE_NAME=default IMAGE_GPU=L4 BOT_CONFIG_NAME=fastapi_websocket_moshi_hibiki_voice_bot MODEL_NAME=kyutai/hibiki-1b-pytorch-bf16 modal serve -e achatbot src/fastapi_ws_moshi_voice_bot_serve.py
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
# single Node multi GPU cards
# deepseek-ai/deepseek-vl2-small use 2xL4
IMAGE_NAME=deepseekvl2 IMAGE_CONCURRENT_CN=1 IMAGE_GPU=L4:2 LLM_MODEL_NAME_OR_PATH=deepseek-ai/deepseek-vl2-small modal serve -e achatbot src/fastapi_webrtc_vision_bot_serve.py
# deepseek-ai/deepseek-vl2 use 4xL4
IMAGE_NAME=deepseekvl2 IMAGE_CONCURRENT_CN=1 IMAGE_GPU=L4:4 LLM_MODEL_NAME_OR_PATH=deepseek-ai/deepseek-vl2 modal serve -e achatbot src/fastapi_webrtc_vision_bot_serve.py

# webrtc_vision_bot serve on minicpmo vision llm pip image
# https://www.nvidia.com/en-us/data-center/l4/ 24GB 
IMAGE_NAME=minicpmo IMAGE_CONCURRENT_CN=1 IMAGE_GPU=L4 modal serve -e achatbot src/fastapi_webrtc_vision_bot_serve.py

# moonshotai/Kimi-VL-A3B-Instruct (or Thinking) use 2xL4 like deepseek-ai/deepseek-vl2-small
IMAGE_NAME=kimi IMAGE_CONCURRENT_CN=1 IMAGE_GPU=L4:2 modal serve -e achatbot src/fastapi_webrtc_vision_bot_serve.py

# webrtc_vision_bot serve on qwen2.5omni vision llm pip image
IMAGE_NAME=qwen2.5omni IMAGE_CONCURRENT_CN=1 IMAGE_GPU=L4 modal serve -e achatbot src/fastapi_webrtc_vision_bot_serve.py
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

### webrtc_minicpmo_vision_voice_bot
- run webrtc_minicpmo_vision_voice_bot serve
```shell
# webrtc_audio_bot serve on default pip image
# need create .env.example to modal Secrets for webrtc key
IMAGE_NAME=default IMAGE_CONCURRENT_CN=1 IMAGE_GPU=L4 modal serve -e achatbot src/fastapi_webrtc_minicpmo_vision_voice_bot_serve.py
# use gptq int4 ckpt
IMAGE_NAME=default IMAGE_CONCURRENT_CN=1 IMAGE_GPU=T4 USE_GPTQ_CKPT=1 modal serve -e achatbot src/fastapi_webrtc_minicpmo_vision_voice_bot_serve.py
```
- curl api to run chat room bot with webrtc (daily)
```shell
# DailyMiniCPMoVoiceBot no ref audio, use sdpa attn impl
curl --location 'https://weedge-achatbot--fastapi-webrtc-minicpmo-omni-bot-srv-app-dev.modal.run/bot_join/chat-room/DailyMiniCPMoVoiceBot' \
--header 'Content-Type: application/json' \
--data '{
  "chat_bot_name": "DailyMiniCPMoVoiceBot",
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
    "voice_llm": "minicpmo"
  },
  "config": {
    "vad": {
      "tag": "silero_vad_analyzer",
      "args": { "stop_secs": 0.7 }
    },
    "voice_llm": {
      "tag": "minicpmo_voice_processor",
      "args": {
        "lm_device": "cuda",
        "lm_torch_dtype": "bfloat16",
        "lm_attn_impl": "sdpa",
        "warmup_steps": 1,
        "lm_gen_temperature": 0.5,
        "lm_model_name_or_path": "/root/.achatbot/models/openbmb/MiniCPM-o-2_6"
      }
    }
  },
  "config_list": []
}'
# DailyAsrMiniCPMoVoiceBot with ref audio, use sdpa attn impl
curl --location 'https://weedge-achatbot--fastapi-webrtc-minicpmo-omni-bot-srv-app-dev.modal.run/bot_join/chat-room/DailyAsrMiniCPMoVoiceBot' \
--header 'Content-Type: application/json' \
--data '{
    "chat_bot_name": "DailyAsrMiniCPMoVoiceBot",
    "config": {
        "asr": {
            "args": {
                "language": "zn",
                "model_name_or_path": "/root/.achatbot/models/FunAudioLLM/SenseVoiceSmall"
            },
            "tag": "sense_voice_asr"
        },
        "vad": {
            "args": {
                "stop_secs": 0.7
            },
            "tag": "silero_vad_analyzer"
        },
        "voice_llm": {
            "args": {
                "lm_attn_impl": "sdpa",
                "lm_device": "cuda",
                "lm_gen_temperature": 0.5,
                "lm_model_name_or_path": "/root/.achatbot/models/openbmb/MiniCPM-o-2_6",
                "lm_torch_dtype": "bfloat16",
                "ref_audio_path": "/root/.achatbot/assets/asr_example_zh.wav",
                "warmup_steps": 1
            },
            "tag": "asr_minicpmo_voice_processor"
        }
    },
    "config_list": [],
    "room_manager": {
        "args": {
            "privacy": "public"
        },
        "tag": "daily_room"
    },
    "room_name": "chat-room",
    "room_url": "",
    "services": {
        "asr": "sense_voice",
        "pipeline": "achatbot",
        "vad": "silero",
        "voice_llm": "minicpmo"
    },
    "token": ""
}'
# DailyMiniCPMoVoiceBot no ref audio, use flash_attention_2 attn impl
curl --location 'https://weedge-achatbot--fastapi-webrtc-minicpmo-omni-bot-srv-app-dev.modal.run/bot_join/chat-room/DailyMiniCPMoVisionVoiceBot' \
--header 'Content-Type: application/json' \
--data '{
  "chat_bot_name": "DailyMiniCPMoVisionVoiceBot",
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
    "omni_llm": "minicpmo"
  },
  "config": {
    "vad": {
      "tag": "silero_vad_analyzer",
      "args": { "stop_secs": 0.7 }
    },
    "omni_llm": {
      "tag": "minicpmo_vision_voice_processor",
      "args": {
        "lm_device_map": "auto",
        "lm_torch_dtype": "bfloat16",
        "lm_attn_impl": "flash_attention_2",
        "warmup_steps": 1,
        "lm_gen_temperature": 0.5,
        "ref_audio_path": "/root/.achatbot/assets/asr_example_zh.wav",
        "lm_model_name_or_path": "/root/.achatbot/models/openbmb/MiniCPM-o-2_6"
      }
    }
  },
  "config_list": []
}'
# DailyMiniCPMoVoiceBot with ref audio, use flash_attention_2 attn impl, use gptq int4 ckpt 
curl --location 'https://weedge-achatbot--fastapi-webrtc-minicpmo-omni-bot-srv-app-dev.modal.run/bot_join/chat-room/DailyMiniCPMoVisionVoiceBot' \
--header 'Content-Type: application/json' \
--data '{
  "chat_bot_name": "DailyMiniCPMoVisionVoiceBot",
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
    "omni_llm": "minicpmo"
  },
  "config": {
    "vad": {
      "tag": "silero_vad_analyzer",
      "args": { "stop_secs": 0.7 }
    },
    "omni_llm": {
      "tag": "minicpmo_vision_voice_processor",
      "args": {
        "lm_device_map": "auto",
        "lm_torch_dtype": "bfloat16",
        "lm_attn_impl": "flash_attention_2",
        "warmup_steps": 1,
        "lm_gen_temperature": 0.5,
        "ref_audio_path": "/root/.achatbot/assets/asr_example_zh.wav",
        "lm_model_name_or_path": "/root/.achatbot/models/openbmb/MiniCPM-o-2_6-int4"
      }
    }
  },
  "config_list": []
}'
```
### webrtc_qwen2_5omni_vision_voice_bot
- run webrtc_qwen2_5omni_vision_voice_bot serve with webrtc
```shell
# webrtc_audio_bot serve on default pip image
# need create .env.example to modal Secrets for webrtc key
IMAGE_CONCURRENT_CN=1 IMAGE_GPU=L40s modal serve -e achatbot src/fastapi_webrtc_qwen2_5omni_vision_voice_bot_serve.py
```
- curl api to run chat room bot with webrtc (livekit_room)
```shell
# thinker gen chunk token and hidden states -> talker gen vq codes token -> code2wav gen chunk wav | don't use_sliding_window_code2wav
curl --location 'https://weedge-achatbot--fastapi-webrtc-qwen2-5omni-bot-srv-app-dev.modal.run/bot_join/chat-room/LivekitQwen2_5OmniVisionVoiceBot' \
--header 'Content-Type: application/json' \
--data '{
  "chat_bot_name": "LivekitQwen2_5OmniVisionVoiceBot",
  "room_name": "chat-room",
  "room_url": "",
  "token": "",
  "room_manager": {
    "tag": "livekit_room",
    "args": {
      "bot_name": "LivekitQwen2_5OmniVisionVoiceBot",
      "is_common_session": false
    }
  },
  "services": {
    "pipeline": "achatbot",
    "vad": "silero",
    "omni_llm": "llm_transformers_manual_qwen2_5omni_vision_voice"
  },
  "config": {
    "vad": {
      "tag": "silero_vad_analyzer",
      "args": { "stop_secs": 0.7 }
    },
    "omni_llm": {
      "tag": "llm_transformers_manual_qwen2_5omni_vision_voice",
      "args": {
        "lm_device": "cuda",
        "lm_torch_dtype": "bfloat16",
        "lm_attn_impl": "flash_attention_2",
        "warmup_steps": 1,
        "chat_history_size": 0,
        "thinker_eos_token_ids": [151644, 151645],
        "thinker_args": {
          "lm_gen_temperature": 0.95,
          "lm_gen_top_k": 20,
          "lm_gen_top_p": 0.9,
          "lm_gen_min_new_tokens": 1,
          "lm_gen_max_new_tokens": 1024,
          "lm_gen_max_tokens_per_step": 10,
          "lm_gen_repetition_penalty": 1.1
        },
        "talker_args": {
          "lm_gen_temperature": 0.95,
          "lm_gen_top_k": 20,
          "lm_gen_top_p": 0.9,
          "lm_gen_min_new_tokens": 1,
          "lm_gen_max_new_tokens": 2048,
          "lm_gen_repetition_penalty": 1.1
        },
        "talker_skip_thinker_token_ids": [],
        "talker_eos_token_ids": [8292, 8294],
        "code2wav_args": {
          "model_path": "/root/.achatbot/models/Qwen/Qwen2.5-Omni-7B",
          "enable_torch_compile": false,
          "enable_torch_compile_first_chunk": false,
          "odeint_method": "euler",
          "odeint_method_relaxed": false,
          "batched_chunk": 3,
          "frequency": "50hz",
          "device": "cuda",
          "num_steps": 10,
          "guidance_scale": 0.5,
          "sway_coefficient": -1.0,
          "code2wav_dynamic_batch": false
        },
        "speaker": "Chelsie",
        "is_use_sliding_window_code2wav": false,
        "lm_model_name_or_path": "/root/.achatbot/models/Qwen/Qwen2.5-Omni-7B"
      }
    }
  },
  "config_list": []
}
'
# thinker gen chunk token and hidden states -> talker gen vq codes token -> code2wav gen chunk wav | use_sliding_window_code2wav | no torch.compile
curl --location 'https://weedge-achatbot--fastapi-webrtc-qwen2-5omni-bot-srv-app-dev.modal.run/bot_join/chat-room/LivekitQwen2_5OmniVisionVoiceBot' \
--header 'Content-Type: application/json' \
--data '{
  "chat_bot_name": "LivekitQwen2_5OmniVisionVoiceBot",
  "room_name": "chat-room",
  "room_url": "",
  "token": "",
  "room_manager": {
    "tag": "livekit_room",
    "args": {
      "bot_name": "LivekitQwen2_5OmniVisionVoiceBot",
      "is_common_session": false
    }
  },
  "services": {
    "pipeline": "achatbot",
    "vad": "silero",
    "omni_llm": "llm_transformers_manual_qwen2_5omni_vision_voice"
  },
  "config": {
    "vad": {
      "tag": "silero_vad_analyzer",
      "args": { "stop_secs": 0.7 }
    },
    "omni_llm": {
      "tag": "llm_transformers_manual_qwen2_5omni_vision_voice",
      "args": {
        "lm_device": "cuda",
        "lm_torch_dtype": "bfloat16",
        "lm_attn_impl": "flash_attention_2",
        "warmup_steps": 1,
        "chat_history_size": 0,
        "thinker_eos_token_ids": [151644, 151645],
        "thinker_args": {
          "lm_gen_temperature": 0.95,
          "lm_gen_top_k": 20,
          "lm_gen_top_p": 0.9,
          "lm_gen_min_new_tokens": 1,
          "lm_gen_max_new_tokens": 1024,
          "lm_gen_max_tokens_per_step": 10,
          "lm_gen_repetition_penalty": 1.1
        },
        "talker_args": {
          "lm_gen_temperature": 0.95,
          "lm_gen_top_k": 20,
          "lm_gen_top_p": 0.9,
          "lm_gen_min_new_tokens": 1,
          "lm_gen_max_new_tokens": 2048,
          "lm_gen_repetition_penalty": 1.1
        },
        "talker_skip_thinker_token_ids": [],
        "talker_eos_token_ids": [8292, 8294],
        "code2wav_args": {
          "model_path": "/root/.achatbot/models/Qwen/Qwen2.5-Omni-7B",
          "enable_torch_compile": false,
          "enable_torch_compile_first_chunk": false,
          "odeint_method": "euler",
          "odeint_method_relaxed": false,
          "batched_chunk": 3,
          "frequency": "50hz",
          "device": "cuda",
          "num_steps": 10,
          "guidance_scale": 0.5,
          "sway_coefficient": -1.0,
          "code2wav_dynamic_batch": false
        },
        "speaker": "Chelsie",
        "is_use_sliding_window_code2wav": true,
        "lm_model_name_or_path": "/root/.achatbot/models/Qwen/Qwen2.5-Omni-7B"
      }
    }
  },
  "config_list": []
}
'
```

### webrtc_step_voice_bot
- run webrtc_step_voice_bot serve with task queue(redis)
```shell
# webrtc_audio_bot serve on default pip image
# need create .env.example to modal Secrets for webrtc key
IMAGE_NAME=default IMAGE_CONCURRENT_CN=1 IMAGE_GPU=L40S:8 modal serve -e achatbot src/fastapi_webrtc_step_voice_bot_serve.py
```
- curl api to run chat room bot with webrtc (daily/livekit/agora)
```shell
curl --location 'https://weedge-achatbot--fastapi-webrtc-step-voice-bot-srv-app-dev.modal.run/bot_join/chat-room/DailyStepVoiceBot' \
--header 'Content-Type: application/json' \
--data '{
    "chat_bot_name": "DailyStepVoiceBot",
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
        "voice_llm": "step"
    },
    "config": {
        "vad": {
            "tag": "silero_vad_analyzer",
            "args": {
                "stop_secs": 0.7
            }
        },
        "voice_llm": {
            "tag": "step_voice_processor",
            "args": {
                "lm_gen_args": {
                    "lm_device_map": "auto",
                    "lm_torch_dtype": "bfloat16",
                    "warmup_steps": 1,
                    "lm_gen_temperature": 0.5,
                    "lm_model_name_or_path": "/root/.achatbot/models/stepfun-ai/Step-Audio-Chat"
                },
                "voice_out_args": {
                    "audio_sample_rate": 22050,
                    "audio_channels": 1
                },
                "voice_tokenizer_path": "/root/.achatbot/models/stepfun-ai/Step-Audio-Tokenizer",
                "voice_decoder_path": "/root/.achatbot/models/stepfun-ai/Step-Audio-TTS-3B",
                "torch_dtype": "auto",
                "device": "cuda"
            }
        }
    },
    "config_list": []
}'
```
### webrtc_kimi_voice_bot
- run webrtc_kimi_voice_bot serve with task queue(redis)
```shell
ACHATBOT_VERSION=0.0.10 IMAGE_CONCURRENT_CN=1 IMAGE_GPU=L40s modal serve -e achatbot src/fastapi_webrtc_kimi_voice_bot_serve.py
```
- curl api to run chat room bot with webrtc (daily/livekit/agora)
```shell
curl --location 'https://weedge-achatbot--fastapi-webrtc-kimi-voice-bot-srv-app-dev.modal.run/bot_join/chat-room/LivekitKimiVoiceBot' \
--header 'Content-Type: application/json' \
--data '{
    "chat_bot_name": "LivekitKimiVoiceBot",
    "room_name": "chat-room",
    "room_url": "",
    "token": "",
    "room_manager": {
        "tag": "livekit_room",
        "args": {
            "bot_name": "LivekitKimiVoiceBot",
            "is_common_session": false
        }
    },
    "services": {
        "pipeline": "achatbot",
        "vad": "silero",
        "voice_llm": "llm_transformers_manual_kimi_voice"
    },
    "config": {
        "vad": {
            "tag": "silero_vad_analyzer",
            "args": {
                "stop_secs": 0.7
            }
        },
        "voice_llm": {
            "tag": "llm_transformers_manual_kimi_voice",
            "args": {
                "no_stream_sleep_time": 0.5,
                "lm_device": "cuda",
                "lm_torch_dtype": "bfloat16",
                "lm_attn_impl": "flash_attention_2",
                "warmup_steps": 1,
                "chat_history_size": 0,
                "code2wav_args": {
                    "max_prompt_chunk": 10,
                    "look_ahead_tokens": 12,
                    "max_kv_cache_tokens": 900,
                    "use_cfg": false,
                    "use_cfg_rescale": true,
                    "cfg_init": 1.5,
                    "cfg_scale": 7.5,
                    "cfg_schedule": "linear",
                    "device": "cuda"
                },
                "lm_model_name_or_path": "/root/.achatbot/models/moonshotai/Kimi-Audio-7B-Instruct"
            }
        }
    },
    "config_list": []
}'
```
### webrtc_vita_voice_bot
- run webrtc_vita_voice_bot serve with task queue(redis)
```shell
ACHATBOT_VERSION=0.0.11 IMAGE_CONCURRENT_CN=1 IMAGE_GPU=L4 modal serve -e achatbot src/fastapi_webrtc_vita_voice_bot_serve.py
```
- curl api to run chat room bot with webrtc (daily/livekit/agora)
```shell
curl --location 'https://weedge-achatbot--fastapi-webrtc-vita-voice-bot-srv-app-dev.modal.run/bot_join/chat-room/LivekitVITAVoiceBot' \
--header 'Content-Type: application/json' \
--data '{
    "chat_bot_name": "LivekitVITAVoiceBot",
    "room_name": "chat-room",
    "room_url": "",
    "token": "",
    "room_manager": {
        "tag": "livekit_room",
        "args": {
            "bot_name": "LivekitVITAVoiceBot",
            "is_common_session": false
        }
    },
    "services": {
        "pipeline": "achatbot",
        "vad": "silero",
        "voice_llm": "llm_transformers_manual_vita_voice"
    },
    "config": {
        "vad": {
            "tag": "silero_vad_analyzer",
            "args": {
                "stop_secs": 0.7
            }
        },
        "voice_llm": {
            "tag": "llm_transformers_manual_vita_voice",
            "args": {
                "no_stream_sleep_time": 0.5,
                "lm_device": "cuda",
                "lm_torch_dtype": "bfloat16",
                "lm_attn_impl": "flash_attention_2",
                "warmup_steps": 1,
                "chat_history_size": 0,
                "audio_tokenizer_type": "sensevoice_glm4voice",
                "audio_tokenizer_model_path": null,
                "sense_voice_model_path": "/root/.achatbot/models/FunAudioLLM/SenseVoiceSmall",
                "flow_path": "/root/.achatbot/models/THUDM/glm-4-voice-decoder",
                "audio_tokenizer_rank": 0,
                "lm_model_name_or_path": "/root/.achatbot/models/VITA-MLLM/VITA-Audio-Plus-Vanilla"
            }
        }
    },
    "config_list": []
}'
```
### webrtc_lite_avatar_chat_bot
- run local bot
```shell
python -m src.cmd.bots.main -f config/bots/daily_liteavatar_echo_bot.json
python -m src.cmd.bots.main -f config/bots/daily_liteavatar_chat_bot.json
```

- download model weights
```shell
modal run src/download_models.py --repo-ids "FunAudioLLM/SenseVoiceSmall"
modal run src/download_models.py --repo-ids "weege007/liteavatar"
```

- run webrtc_avatar_chat_bot serve
```shell
ACHATBOT_VERSION=0.0.18 IMAGE_CONCURRENT_CN=1 IMAGE_GPU=T4 modal serve src/fastapi_webrtc_avatar_bot_serve.py
```
- curl api to run chat room bot with webrtc (daily)
```shell
curl --location 'https://weedge--fastapi-webrtc-avatar-bot-srv-app-dev.modal.run/bot_join/chat-room/DailyAvatarChatBot' \
--header 'Content-Type: application/json' \
--data '{
    "chat_bot_name": "DailyAvatarChatBot",
    "room_name": "chat-room",
    "room_url": "",
    "token": "",
    "room_manager": {
        "tag": "daily_room",
        "args": {
            "privacy": "public"
        }
    },
    "is_background": false,
    "services": {
        "pipeline": "achatbot",
        "vad": "silero",
        "asr": "sense_voice",
        "llm": "together",
        "tts": "edge",
        "avatar": "liteavatar"
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
                    "content": "你是一名医疗专家。保持回答简短和清晰。请用中文回答。"
                }
            ]
        },
        "tts": {
            "tag": "tts_edge",
            "args": {
                "voice_name": "zh-CN-XiaoxiaoNeural",
                "language": "zh",
                "gender": "Female"
            }
        },
        "avatar": {
            "args": {
                "avatar_name": "20250612/P1-64AzfrJY037WpS69RiUMw",
                "is_show_video_debug_text": true,
                "use_gpu": true,
                "weight_dir": "/root/.achatbot/models/weege007/liteavatar",
                "is_flip": false
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

```shell
IMAGE_NAME=kimi IMAGE_CONCURRENT_CN=1 IMAGE_GPU=L40s modal serve -e achatbot src/fastapi_webrtc_vision_bot_serve.py
```
endpoint: https://weedge-achatbot--fastapi-webrtc-vision-kimi-bot-srv-app.modal.run/

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

- deploy webrtc_minicpmo_vision_voice_bot serve
```shell
IMAGE_NAME=default IMAGE_CONCURRENT_CN=1 IMAGE_GPU=L4 modal deploy -e achatbot src/fastapi_webrtc_minicpmo_vision_voice_bot_serve.py
```
endpoint: https://weedge-achatbot--fastapi-webrtc-minicpmo_omni-bot-srv-app.modal.run/

# references (nice docs) 👍 @modal
- https://modal.com/docs/guide
- https://modal.com/docs/guide/gpu
- https://modal.com/docs/guide/cuda
- https://modal.com/docs/guide/volumes
- https://modal.com/docs/examples/basic_web
- https://modal.com/docs/guide/cron
- https://github.com/modal-labs/modal-examples