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
    >
    > ```
    > ACHATBOT_PKG=1 python -m achatbot.cmd.bots.rag.data_process.youtube_audio_transcribe_to_tidb
    > ```
    >
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

   