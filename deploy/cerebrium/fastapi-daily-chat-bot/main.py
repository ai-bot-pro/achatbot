from achatbot.cmd.http.server import fastapi_daily_bot_serve as serve

# from cerebrium import get_secret


def status():
    return {"status_code": 200}


# !TIPS:
# cerebrum wrap fastapi, use HTTP POST method api
# so use achatbot serve APIResponse dict[str, Any] to return
# KISS, have a nice code :)

create_random_room = serve.create_random_room
register_bot = serve.register_bot
bot_join = serve.bot_join
bot_join_room = serve.bot_join_room
create_random_room = serve.create_random_room
get_bot_proc_status = serve.get_status
get_num_bots = serve.get_num_bots
get_room_bots = serve.get_room_bots
