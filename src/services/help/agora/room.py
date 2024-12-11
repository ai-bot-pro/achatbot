from src.services.help.agora.channel import AgoraChannel


class AgoraAPaaSRoom(AgoraChannel):
    """
    type: App PaaS (room)
    create class room
    e.g.: https://solutions-apaas.agora.io/apaas/demo/index.html#/
    """

    TAG = "agora_appaas_room"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
