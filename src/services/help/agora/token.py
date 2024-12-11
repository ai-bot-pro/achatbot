from dataclasses import dataclass, field
import time

from agora_realtime_ai_api.token_builder.AccessToken2 import AccessToken


@dataclass
class RtcGrants:
    """
        RTC is a SDK for you to do real time video/audio communication.
    https://docs.agora.io/en/Interactive%20Broadcast/product_live
    """

    joinChannel: bool = False
    publishAudioStream: bool = False
    publishVideoStream: bool = False
    publishDataStream: bool = False
    channel_name: str = ""
    uid: str = ""


@dataclass
class SignalingRtmGrants:
    """
    RTM aka Real Time Messaging is a SDK for you to send instant signaling messages. https://docs.agora.io/en/Real-time-Messaging/product_rtm
    """

    login: bool = False
    user_id: str = ""


@dataclass
class FpaGrants:
    login: bool = False


@dataclass
class ChatGrants:
    """
    https://docs.agora.io/en/agora-chat/overview/product-overview
    """

    user: bool = False
    app: bool = False
    user_id: str = ""


@dataclass
class ApaasGrants:
    """
    https://docs.agora.io/en/flexible-classroom/overview/product-overview
    """

    roomUser: bool = False
    user: bool = False
    app: bool = False
    room_uuid: str = ""
    user_uuid: str = ""
    role: int = 0


@dataclass
class TokenClaims:
    token: str = ""
    app_id: str = ""
    issue_ts: int = 0
    expire: int = 0
    salt: int = 0
    rtc: RtcGrants = field(default_factory=RtcGrants)
    signaling_rtm: SignalingRtmGrants = field(default_factory=SignalingRtmGrants)
    chat_msg: ChatGrants = field(default_factory=ChatGrants)
    apaas_room: ApaasGrants = field(default_factory=ApaasGrants)


class TokenPaser:
    """
    https://github.com/AgoraIO/Tools/blob/master/DynamicKey/AgoraDynamicKey/parse.py
    """

    service = {
        1: {
            "name": "Rtc",
            "privilege": {
                1: {"name": "joinChannel"},
                2: {"name": "publishAudioStream"},
                3: {"name": "publishVideoStream"},
                4: {"name": "publishDataStream"},
            },
        },
        2: {"name": "Rtm", "privilege": {1: {"name": "login"}}},
        4: {"name": "Fpa", "privilege": {1: {"name": "login"}}},
        5: {"name": "Chat", "privilege": {1: {"name": "user"}, 2: {"name": "app"}}},
        7: {
            "name": "Apaas",
            "privilege": {1: {"name": "roomUser"}, 2: {"name": "user"}, 3: {"name": "app"}},
        },
    }

    @staticmethod
    def check_expire(expire):
        remain = expire - int(time.time())
        return remain < 0, remain

    @staticmethod
    def get_expire_msg(expire):
        is_expired, remain = TokenPaser.check_expire(expire)
        if is_expired:
            return "expired"
        return "will expire in %d seconds" % remain

    @staticmethod
    def _access_token(token):
        if len(token) <= 3:
            raise ValueError("Invalid token version")

        if token[:3] != "007":
            raise ValueError("Not support, just for parsing token version 007!")

        # so ugly token design, need use jwt
        access_token = AccessToken()
        try:
            access_token.from_string(token)
        except Exception as e:
            raise ValueError(f"Parse token failed! err: {e}")

        return access_token

    @staticmethod
    def is_expired(token):
        try:
            access_token = TokenPaser._access_token(token)
        except Exception as e:
            raise e

        is_expired, _ = TokenPaser.check_expire(
            access_token._AccessToken__issue_ts + access_token._AccessToken__expire
        )
        return is_expired

    @staticmethod
    def parse_claims(token) -> TokenClaims:
        try:
            access_token = TokenPaser._access_token(token)
        except Exception as e:
            raise e
        tokenClaims = TokenClaims(
            token=token,
            app_id=access_token._AccessToken__app_id.decode(),
            issue_ts=access_token._AccessToken__issue_ts,
            expire=access_token._AccessToken__expire,
            salt=access_token._AccessToken__salt,
        )

        for _, item in access_token._AccessToken__service.items():
            for key, serviceItem in item.__dict__.items():
                if key == "_Service__type":
                    sType = item._Service__type
                    if item._Service__type not in TokenPaser.service:
                        continue
                elif key == "_Service__privileges":
                    for privilege, _ in item._Service__privileges.items():
                        match sType:
                            case 1:
                                if privilege == 1:
                                    tokenClaims.rtc.joinChannel = True
                                if privilege == 2:
                                    tokenClaims.rtc.publishAudioStream = True
                                if privilege == 3:
                                    tokenClaims.rtc.publishVideoStream = True
                                if privilege == 4:
                                    tokenClaims.rtc.publishDataStream = True
                            case 2:
                                if privilege == 1:
                                    tokenClaims.signaling_rtm.login = True
                            case 5:
                                if privilege == 1:
                                    tokenClaims.chat_msg.user = True
                                if privilege == 2:
                                    tokenClaims.chat_msg.app = True
                            case 7:
                                if privilege == 1:
                                    tokenClaims.apaas_room.roomUser = True
                                if privilege == 2:
                                    tokenClaims.apaas_room.user = True
                                if privilege == 3:
                                    tokenClaims.apaas_room.app = True
                else:
                    key_name = key.replace(
                        "_Service%s__" % TokenPaser.service[item._Service__type]["name"], ""
                    )
                    val = serviceItem.decode() if isinstance(serviceItem, bytes) else serviceItem
                    match sType:
                        case 1:
                            if key_name == "channel_name":
                                tokenClaims.rtc.channel_name = val
                            if key_name == "uid":
                                tokenClaims.rtc.uid = val
                        case 2:
                            if key_name == "user_id":
                                tokenClaims.signaling_rtm.user_id = val
                        case 5:
                            if key_name == "user_id":
                                tokenClaims.chat_msg.user_id = val
                        case 7:
                            if key_name == "room_uuid":
                                tokenClaims.apaas_room.room_uuid = val
                            if key_name == "user_uuid":
                                tokenClaims.apaas_room.user_uuid = val
                            if key_name == "role":
                                tokenClaims.apaas_room.role = val

        return tokenClaims

    @staticmethod
    def parse_detail(token):
        res = "\nToken is %s \n\n" % token
        try:
            access_token = TokenPaser._access_token(token)
        except Exception as e:
            res += "Parse token failed! %s \n" % e
            return res

        res += "Parse token success! \n"
        res += "Token information, %s. \n    - app_id:%s, issue_ts:%d, expire:%d, salt:%d \n" % (
            TokenPaser.get_expire_msg(
                access_token._AccessToken__issue_ts + access_token._AccessToken__expire
            ),
            access_token._AccessToken__app_id.decode(),
            access_token._AccessToken__issue_ts,
            access_token._AccessToken__expire,
            access_token._AccessToken__salt,
        )

        for _, item in access_token._AccessToken__service.items():
            for key, serviceItem in item.__dict__.items():
                if key == "_Service__type":
                    if item._Service__type not in TokenPaser.service:
                        res += "service type not existed, type:%d \n" % item._Service__type
                        continue
                    res += "- Service information, type:%d (%s) \n" % (
                        item._Service__type,
                        TokenPaser.service[item._Service__type]["name"],
                    )
                elif key == "_Service__privileges":
                    for privilege, privilegeExpire in item._Service__privileges.items():
                        res += "    - privilege:%d(%s), expire:%d (%s) \n" % (
                            privilege,
                            TokenPaser.service[item._Service__type]["privilege"][privilege]["name"],
                            privilegeExpire,
                            TokenPaser.get_expire_msg(
                                access_token._AccessToken__issue_ts + privilegeExpire
                            ),
                        )
                else:
                    res += "    - {}:{} \n".format(
                        key.replace(
                            "_Service%s__" % TokenPaser.service[item._Service__type]["name"], ""
                        ),
                        serviceItem.decode() if isinstance(serviceItem, bytes) else serviceItem,
                    )

        return res


if __name__ == "__main__":
    """
    python -m src.services.help.agora.token
    python -m src.services.help.agora.token TOKEN
    """
    import sys

    if len(sys.argv) > 1:
        token = sys.argv[1]
    else:
        token = "007eJxTYNi/pqL4zazPf+P2/HDX+9fA/KLX+6oIz5O5Wzw2vzTSPdqtwGBpbuDsaGyakmpmkGxiYmZimpSUmGqRaGRoamBmmGRs/P87S7IAHwOD/mEfBlYGRgYWIAbxmcAkM5hkAZMKDOYp5kbGZqapSZYWxiYWpsaW5qnGqcZplikmZgZJKSmJXAxGFhZGxiaGRubGTEBzICYhi7LARVkZmFBsQlbFDrQX0xV8DEX5+bnxpaWZKfElqcUlfAylxalFCP7//wBAqz0L"

    print(TokenPaser.parse_detail(token))

    print(TokenPaser.parse_claims(token))
