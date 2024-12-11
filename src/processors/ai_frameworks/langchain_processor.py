import logging
from typing import Union

from src.types.frames.control_frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame
from src.types.frames.data_frames import Frame, LLMMessagesFrame, TextFrame
from apipeline.processors.frame_processor import FrameDirection, FrameProcessor


try:
    from langchain_core.messages import AIMessageChunk
    from langchain_core.runnables import Runnable
except ModuleNotFoundError as e:
    logging.exception("In order to use Langchain, you need to `pip install langchain`. ")
    raise Exception(f"Missing module: {e}")


class LangchainProcessor(FrameProcessor):
    def __init__(self, chain: Runnable, transcript_key: str = "input"):
        super().__init__()
        self._chain = chain
        self._transcript_key = transcript_key
        self._participant_id: str | None = None

    def set_participant_id(self, participant_id: str):
        self._participant_id = participant_id

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMMessagesFrame):
            # Messages are accumulated by the `LLMUserResponseAggregator` in a list of messages.
            # The last one by the human is the one we want to send to the LLM.
            logging.debug(f"Got transcription frame {frame}")
            text: str = frame.messages[-1]["content"]

            await self._ainvoke(text.strip())
        else:
            await self.push_frame(frame, direction)

    def _get_token_value(self, text: Union[str, AIMessageChunk]) -> str:
        match text:
            case str():
                return text
            case AIMessageChunk():
                return text.content
            case dict() as d if "answer" in d:
                return d["answer"]
            case _:
                return ""

    async def _ainvoke(self, text: str):
        logging.debug(f"Invoking chain with {text}")
        await self.push_frame(LLMFullResponseStartFrame())
        try:
            async for token in self._chain.astream(
                {self._transcript_key: text},
                config={"configurable": {"session_id": self._participant_id}},
            ):
                await self.push_frame(TextFrame(self._get_token_value(token)))
        except GeneratorExit:
            logging.warning(f"{self} generator was closed prematurely")
        except Exception as e:
            logging.exception(f"{self} an unknown error occurred: {e}")
        finally:
            await self.push_frame(LLMFullResponseEndFrame())
