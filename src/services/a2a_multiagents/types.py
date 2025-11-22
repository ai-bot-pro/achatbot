from pydantic import BaseModel, Field
from a2a.types import Message


class Conversation(BaseModel):
    conversation_id: str
    is_active: bool
    name: str = ""
    task_ids: list[str] = Field(default_factory=list)
    messages: list[Message] = Field(default_factory=list)


class Event(BaseModel):
    id: str
    actor: str = ""
    # TODO: Extend to support internal concepts for models, like function calls.
    content: Message
    timestamp: float
