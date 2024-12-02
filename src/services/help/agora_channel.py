import os
import time
import logging
import uuid

from src.common.const import *
from src.common.types import AgoraChannelArgs, GeneralRoomInfo
from src.common.interface import IRoomManager
from src.common.factory import EngineClass

from agora_realtime_ai_api.rtc import RtcEngine, RtcOptions, Channel
from agora_realtime_ai_api.token_builder.realtimekit_token_builder import RealtimekitTokenBuilder
from agora_realtime_ai_api.token_builder.AccessToken2 import AccessToken

from dotenv import load_dotenv
load_dotenv(override=True)

class TokenPaser():
    """
    https://github.com/AgoraIO/Tools/blob/master/DynamicKey/AgoraDynamicKey/parse.py
    """
    service = {
        1: {
            'name': 'Rtc',
            'privilege': {
                1: {'name': 'joinChannel'},
                2: {'name': 'publishAudioStream'},
                3: {'name': 'publishVideoStream'},
                4: {'name': 'publishDataStream'}
            }
        },
        2: {
            'name': 'Rtm',
            'privilege': {
                1: {'name': 'login'}
            }
        },
        4: {
            'name': 'Fpa',
            'privilege': {
                1: {'name': 'login'}
            }
        },
        5: {
            'name': 'Chat',
            'privilege': {
                1: {'name': 'user'},
                2: {'name': 'app'}
            }
        },
        7: {
            'name': 'Apaas',
            'privilege': {
                1: {'name': 'roomUser'},
                2: {'name': 'user'},
                3: {'name': 'app'}
            }
        }
    }


    @staticmethod
    def check_expire(expire):
        remain = expire - int(time.time())
        return remain < 0, remain


    @staticmethod
    def get_expire_msg(expire):
        is_expired, remain = TokenPaser.check_expire(expire)
        if is_expired:
            return 'expired'
        return 'will expire in %d seconds' % remain

    @staticmethod
    def valid_token(token):
        if len(token)<=3:
            raise ValueError('Invalid token')

        if token[:3] != '007':
            raise ValueError('Not support, just for parsing token version 007!')

        access_token = AccessToken()
        try:
            access_token.from_string(token)
        except Exception as e:
            raise ValueError(f'Parse token failed! err: {e}')
        
        return True
    @staticmethod
    def parse_token(token):
        res = '\nToken is %s \n\n' % token

        if token[:3] != '007':
            return res + 'Not support, just for parsing token version 007!'

        access_token = AccessToken()
        try:
            access_token.from_string(token)
        except Exception as e:
            res += 'Parse token failed! %s \n' % e
            return res

        res += 'Parse token success! \n'
        res += 'Token information, %s. \n    - app_id:%s, issue_ts:%d, expire:%d, salt:%d \n' % (
            TokenPaser.get_expire_msg(access_token._AccessToken__issue_ts + access_token._AccessToken__expire), access_token._AccessToken__app_id.decode(), access_token._AccessToken__issue_ts, access_token._AccessToken__expire, access_token._AccessToken__salt)

        for _, item in access_token._AccessToken__service.items():
            for key, serviceItem in item.__dict__.items():
                if key == '_Service__type':
                    if item._Service__type not in access_token.service:
                        res += 'service type not existed, type:%d \n' % item._Service__type
                        continue
                    res += '- Service information, type:%d (%s) \n' % (
                        item._Service__type, access_token.service[item._Service__type]['name'])
                elif key == '_Service__privileges':
                    for privilege, privilegeExpire in item._Service__privileges.items():
                        res += '    - privilege:%d(%s), expire:%d (%s) \n' % (
                            privilege, access_token.service[item._Service__type]['privilege'][privilege]['name'], privilegeExpire,
                            access_token.get_expire_msg(access_token._AccessToken__issue_ts + privilegeExpire))
                else:
                    res += '    - {}:{} \n'.format(key.replace('_Service%s__' % access_token.service[item._Service__type]['name'], ''),
                                                   serviceItem.decode() if type(serviceItem) == bytes else serviceItem)

        return res



class AgoraChannel(EngineClass, IRoomManager):
    """
    no room, just use channel name as room name
    """
    TAG = "agora_channel"

    def __init__(self, **kwargs) -> None:
        self.args = AgoraChannelArgs(**kwargs)
        self.app_id = os.environ.get("AGORA_APP_ID")
        self.app_cert = os.environ.get("AGORA_APP_CERT")
        if not self.app_id:
            raise ValueError("AGORA_APP_ID must be set in the environment.")
        engine = RtcEngine(appid=self.app_id, appcert=self.app_cert)

    async def close_session(self):
        # no http rest api session, don't do anything
        pass

    async def create_room(
            self,
            room_name: str | None = None,
            exp_time_s: int = ROOM_EXPIRE_TIME) -> GeneralRoomInfo:
        return GeneralRoomInfo(
            name=room_name,
        )

    async def create_random_room(
            self,
            exp_time_s: int = RANDOM_ROOM_EXPIRE_TIME) -> GeneralRoomInfo:
        return GeneralRoomInfo(
            name=str(uuid.uuid4()),
        )

    async def gen_token(self, room_name: str, exp_time_s: int = ROOM_TOKEN_EXPIRE_TIME) -> str:
        token = RealtimekitTokenBuilder.build_token(
            self.app_id, self.app_cert, room_name, self.uid, expiration_in_seconds=exp_time_s
        )
        logging.debug(f"token:{token}")
        return token

    async def get_room(self, room_name: str) -> GeneralRoomInfo:
        g_room = GeneralRoomInfo(
            name=room_name,
        )
        return g_room

    async def check_valid_room(self, room_name: str, token: str) -> bool:
        if not room_name or not token:
            return False

        try:
            TokenPaser.valid_token(token)
        except Exception as e:
            logging.error(f"check_valid_room error: {e}")
            return False

        room = await self.get_room(room_name)
        if not room:
            return False

        return True
