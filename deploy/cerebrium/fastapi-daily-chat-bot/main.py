import os
import logging

from achatbot.cmd.http.server import fastapi_daily_bot_serve as serve
from cerebrium import get_secret

os.environ["LLM_CHAT_SYSTEM"] = get_secret("LLM_CHAT_SYSTEM")
os.environ["OPENAI_API_KEY"] = get_secret("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = get_secret("GROQ_API_KEY")
os.environ["TOGETHER_API_KEY"] = get_secret("TOGETHER_API_KEY")
os.environ["DAILY_API_KEY"] = get_secret("DAILY_API_KEY")
os.environ["CARTESIA_API_KEY"] = get_secret("CARTESIA_API_KEY")
os.environ["DEEPGRAM_API_KEY"] = get_secret("DEEPGRAM_API_KEY")
os.environ["ELEVENLABS_API_KEY"] = get_secret("ELEVENLABS_API_KEY")
os.environ["JINA_API_KEY"] = get_secret("JINA_API_KEY")

os.environ["VAD_ANALYZER_TAG"] = get_secret("VAD_ANALYZER_TAG")
os.environ["SILERO_MODEL_SOURCE"] = get_secret("SILERO_MODEL_SOURCE")
os.environ["SILERO_REPO_OR_DIR"] = get_secret("SILERO_REPO_OR_DIR")
os.environ["SILERO_MODEL"] = get_secret("SILERO_MODEL")

os.environ["ASR_TAG"] = get_secret("ASR_TAG")
os.environ["ASR_LANG"] = get_secret("ASR_LANG")
os.environ["ASR_MODEL_NAME_OR_PATH"] = get_secret("ASR_MODEL_NAME_OR_PATH")

os.environ["LLM_OPENAI_BASE_URL"] = get_secret("LLM_OPENAI_BASE_URL")
os.environ["LLM_OPENAI_MODEL"] = get_secret("LLM_OPENAI_MODEL")
os.environ["LLM_LANG"] = get_secret("LLM_LANG")

os.environ["TTS_TAG"] = get_secret("TTS_TAG")
os.environ["TTS_LANG"] = get_secret("TTS_LANG")
os.environ["TTS_VOICE"] = get_secret("TTS_VOICE")

os.environ["TIDB_HOST"] = get_secret("TIDB_HOST")
os.environ["TIDB_PORT"] = get_secret("TIDB_PORT")
os.environ["TIDB_SSL_CA"] = get_secret("TIDB_SSL_CA")
os.environ["TIDB_USERNAME"] = get_secret("TIDB_USERNAME")
os.environ["TIDB_PASSWORD"] = get_secret("TIDB_PASSWORD")
os.environ["TIDB_DATABASE"] = get_secret("TIDB_DATABASE")
os.environ["TIDB_VSS_DISTANCE_STRATEGY"] = get_secret("TIDB_VSS_DISTANCE_STRATEGY")


# !TIPS:
# cerebrum wrap fastapi, use HTTP POST method api
# so use achatbot serve APIResponse dict[str, Any] to return
# KISS, have a nice code :)
# register_bot = serve.register_bot
app_status = serve.app_status
create_random_room = serve.create_random_room
bot_join = serve.bot_join
bot_join_room = serve.bot_join_room
get_bot_proc_status = serve.get_status
get_num_bots = serve.get_num_bots
get_room_bots = serve.get_room_bots
