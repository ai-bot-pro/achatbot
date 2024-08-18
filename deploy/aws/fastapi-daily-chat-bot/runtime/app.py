try:
    import unzip_requirements
except ImportError:
    pass

from achatbot.cmd.http.server.fastapi_daily_bot_serve import app
from mangum import Mangum

handler = Mangum(app)
