from dataclasses import dataclass
from typing import List


from apipeline.frames.sys_frames import SystemFrame

from src.types.metrics.metrics import MetricsData


@dataclass
class BotInterruptionFrame(SystemFrame):
    """Emitted by when the bot should be interrupted. This will mainly cause the
    same actions as if the user interrupted except that the
    UserStartedSpeakingFrame and UserStoppedSpeakingFrame won't be generated.

    """

    pass


@dataclass
class FunctionCallInProgressFrame(SystemFrame):
    """A frame signaling that a function call is in progress."""

    function_name: str
    tool_call_id: str
    arguments: str


@dataclass
class MetricsFrame(SystemFrame):
    """Frame containing performance metrics data.

    Emitted by processors that can compute metrics like latencies.

    Parameters:
        data: List of metrics data collected by the processor.
    """

    data: List[MetricsData]