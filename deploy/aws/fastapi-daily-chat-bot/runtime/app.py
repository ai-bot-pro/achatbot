from achatbot.cmd.http.server.fastapi_daily_bot_serve import app
from mangum import Mangum

handler = Mangum(app)
