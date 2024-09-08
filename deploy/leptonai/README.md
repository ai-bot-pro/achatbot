# app

- [fastapi-daily-chat-bot](https://github.com/ai-bot-pro/achatbot/tree/main/deploy/leptonai/fastapi-daily-chat-bot)

## deploy

use shell script to deploy, simply run, like run on local machine. :) yes!

```
cd fastapi-daily-chat-bot

pip install leptonai

# create
lep photon create -n fastapi-daily-chat-bot -m photon.py

# local run
lep photon runlocal -n fastapi-daily-chat-bot


# deploy
lep login
lep photon push -n fastapi-daily-chat-bot

# run
lep photon run -n fastapi-daily-chat-bot -dn fastapi-daily-chat-bot

# status
lep deployment status -n fastapi-daily-chat-bot

# test
```

# references
- https://www.lepton.ai/docs