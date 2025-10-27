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

