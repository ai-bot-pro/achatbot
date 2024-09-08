# app

- [fastapi-daily-chat-bot](https://github.com/ai-bot-pro/achatbot/tree/main/deploy/leptonai/fastapi-daily-chat-bot)

## deploy

use shell script to deploy, simply run, like run on local machine. :) yes!

> [!NOTE]
> if you just change the achatbot code, don't need to do this, just re-run deploy. so nice~, but need smooth update to switch to new deployment.

> [!NOTE]
> if rerun deployment, need to add secret again. (secret hope is golab~)
> rerun deployment, e.g.:
> `lep photon run --rerun -n fastapi-daily-chat-bot -dn fastapi-daily-chat-bot --secret DAILY_API_KEY --secret DEEPGRAM_API_KEY --secret OPENAI_API_KEY --secret GROQ_API_KEY --secret TOGETHER_API_KEY`

### new deployment
```
cd fastapi-daily-chat-bot

pip install leptonai

# create
lep photon create -n fastapi-daily-chat-bot -m photon.py

# local run
lep photon runlocal -n fastapi-daily-chat-bot

# test
curl http://0.0.0.0:8080/metrics


# deploy
lep login
lep photon push -n fastapi-daily-chat-bot

# run
lep photon run -n fastapi-daily-chat-bot -dn fastapi-daily-chat-bot

# status
lep deployment status -n fastapi-daily-chat-bot

# test
curl http://0.0.0.0:8080/metrics

```

```
# rerun deploy
lep photon run --rerun -n fastapi-daily-chat-bot -dn fastapi-daily-chat-bot
```

## CD (when change photon.py)
continue to deploy change, just for change the photon.py, and push again.
```
# list local photon
lep photon list -l
# remove local photon (if you update photon.py, want to re-create, then push again)
lep photon remove -n fastapi-daily-chat-bot -l
lep photon create -n fastapi-daily-chat-bot -m photon.py
lep photon push -n fastapi-daily-chat-bot
# list remote deployment
lep photon list

# run new deployment(latest photon)
lep photon run -n fastapi-daily-chat-bot -dn fastapi-daily-chat-bot-***
# run new deployment(defined ID photon)
lep photon run -n fastapi-daily-chat-bot -i fastapi-daily-chat-bot-*** -dn fastapi-daily-chat-bot-***
# remove remove old photon(CI version)
lep deployment remove fastapi-daily-chat-bot -i fastapi-daily-chat-bot-***
```

# references
- https://www.lepton.ai/docs
