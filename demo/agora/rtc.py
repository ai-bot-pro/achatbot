# the first run need down load agora sdk core lib(c++)

import os
from dotenv import load_dotenv

from agora_realtime_ai_api.rtc import RtcEngine, RtcOptions

PCM_SAMPLE_RATE = 24000
PCM_CHANNELS = 1

load_dotenv(override=True)
app_id = os.environ.get("AGORA_APP_ID")
app_cert = os.environ.get("AGORA_APP_CERT")

if not app_id:
    raise ValueError("AGORA_APP_ID must be set in the environment.")


engine = RtcEngine(appid=app_id, appcert=app_cert)
print(engine)

channel_name = os.environ.get("AGORA_CHANNEL_NAME")
uid = os.environ.get("AGORA_UID")
options = RtcOptions(
    channel_name=channel_name,
    uid=uid,
    sample_rate=PCM_SAMPLE_RATE,
    channels=PCM_CHANNELS,
    enable_pcm_dump=os.environ.get("WRITE_RTC_PCM", "false") == "true"
)
channel = engine.create_channel(options)
print(channel)

