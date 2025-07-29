from enum import Enum


class EndOfTurnState(Enum):
    """State enumeration for end-of-turn analysis results.

    Parameters:
        COMPLETE: The user has finished their turn and stopped speaking.
        INCOMPLETE: The user is still speaking or may continue speaking.
    """

    COMPLETE = 1
    INCOMPLETE = 2
