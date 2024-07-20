from typing import Mapping
from dataclasses import dataclass


from .base import Frame

#
# System frames
#


@dataclass
class SystemFrame(Frame):
    pass


@dataclass
class StartFrame(SystemFrame):
    """This is the first frame that should be pushed down a pipeline."""
    allow_interruptions: bool = False
    enable_metrics: bool = False
    report_only_initial_ttfb: bool = False


@dataclass
class CancelFrame(SystemFrame):
    """Indicates that a pipeline needs to stop right away."""
    pass


@dataclass
class ErrorFrame(SystemFrame):
    """This is used notify upstream that an error has occurred downstream the
    pipeline."""
    error: str | None

    def __str__(self):
        return f"{self.name}(error: {self.error})"


@dataclass
class StopTaskFrame(SystemFrame):
    """Indicates that a pipeline task should be stopped. This should inform the
    pipeline processors that they should stop pushing frames but that they
    should be kept in a running state.

    """
    pass


@dataclass
class StartInterruptionFrame(SystemFrame):
    """Emitted by VAD to indicate that a user has started speaking (i.e. is
    interruption). This is similar to UserStartedSpeakingFrame except that it
    should be pushed concurrently with other frames (so the order is not
    guaranteed).

    """
    pass


@dataclass
class StopInterruptionFrame(SystemFrame):
    """Emitted by VAD to indicate that a user has stopped speaking (i.e. no more
    interruptions). This is similar to UserStoppedSpeakingFrame except that it
    should be pushed concurrently with other frames (so the order is not
    guaranteed).

    """
    pass


@dataclass
class MetricsFrame(SystemFrame):
    """Emitted by processor that can compute metrics like latencies.
    """
    ttfb: Mapping[str, float]
