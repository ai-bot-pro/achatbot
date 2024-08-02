from dataclasses import dataclass


from apipeline.frames.sys_frames import SystemFrame


@dataclass
class BotInterruptionFrame(SystemFrame):
    """Emitted by when the bot should be interrupted. This will mainly cause the
    same actions as if the user interrupted except that the
    UserStartedSpeakingFrame and UserStoppedSpeakingFrame won't be generated.

    """
    pass
