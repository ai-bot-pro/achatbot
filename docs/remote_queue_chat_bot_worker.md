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

