import logging
from typing import Any, Mapping

from apipeline.frames.control_frames import EndFrame

from src.cmd.bots.base import AIRoomBot
from src.transports.daily import DailyTransport


class DailyRoomBot(AIRoomBot):
    async def on_first_participant_joined(
        self,
        transport: DailyTransport,
        participant: Mapping[str, Any],
    ):
        self.session.set_client_id(participant["id"])
        logging.info(f"First participant {participant['id']} joined")

    async def on_participant_left(
        self,
        transport: DailyTransport,
        participant: Mapping[str, Any],
        reason: str,
    ):
        if self.task is not None:
            await self.task.queue_frame(EndFrame())
        logging.info(f"Partcipant {participant} left. Exiting.")

    async def on_call_state_updated(
        self,
        transport: DailyTransport,
        state: str,
    ):
        logging.info("Call state %s " % state)
        if state == "left" and self.task is not None:
            await self.task.queue_frame(EndFrame())
