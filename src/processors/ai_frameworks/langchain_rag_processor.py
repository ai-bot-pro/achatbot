import logging
from typing import Union

from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import Runnable
from apipeline.frames.data_frames import TextFrame

from src.processors.ai_frameworks.langchain_processor import LangchainProcessor
from src.types.frames.control_frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMResponseEndFrame,
    LLMResponseStartFrame,
)


class LangchainRAGProcessor(LangchainProcessor):
    def __init__(self, chain: Runnable, transcript_key: str = "input"):
        super().__init__(chain, transcript_key)
        self._chain = chain
        self._transcript_key = transcript_key

    async def _ainvoke(self, text: str):
        logging.debug(f"Invoking chain with {text}")
        targetPhrases = [
            "you can continue with the lecture",
            "continue with the lecture",
            "you can continue with lecture",
            "continue with lecture",
            "play the video",
            "继续讲课",
            "播放视频",
        ]

        # Simple fuzzy matching by checking if the target phrase is included in the transcript text
        matchFound = any(phrase in text for phrase in targetPhrases)
        if matchFound:
            logging.info("Fuzzy match found for the phrase: 'You can continue with the lecture'")
            return

        await self.push_frame(LLMFullResponseStartFrame())
        try:
            async for token in self._chain.astream(
                {self._transcript_key: text},
                config={"configurable": {"session_id": self._participant_id}},
            ):
                await self.push_frame(LLMResponseStartFrame())
                await self.push_frame(TextFrame(self._get_token_value(token)))
                await self.push_frame(LLMResponseEndFrame())
        except GeneratorExit:
            logging.warning(f"{self} generator was closed prematurely")
        except Exception as e:
            logging.exception(f"{self} an unknown error occurred: {e}")
        finally:
            await self.push_frame(LLMFullResponseEndFrame())
