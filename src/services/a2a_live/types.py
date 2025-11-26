from typing import List, Literal, Optional

from pydantic import BaseModel, Field
from google.genai import types


class LiveMultiModalInputMessage(BaseModel):
    context_id: Optional[str] = Field(default=None, description="The context ID for the message.")
    message_id: Optional[str] = Field(default=None, description="message ID")
    task_id: Optional[str] = Field(default=None, description="task ID for the message")
    kind: Literal["text", "audio", "images", "text_images", "audio_images"] = Field(
        default="audio_images", description="the task kind of the input message"
    )
    text_content: Optional[types.Content] = Field(
        default=None,
        description="""Optional. text content when kind is text""",
    )
    audio_blob: Optional[types.Blob] = Field(
        default=None,
        description="""Optional. audio binary blob when kind is audio or audio_images""",
    )
    image_blob_list: Optional[List[types.Blob]] = Field(
        default=None,
        description="""Optional. image binary blob list when kind is text_images or audio_images""",
    )


class LiveMultiModalOutputMessage(BaseModel):
    context_id: Optional[str] = Field(default=None, description="The context ID for the message.")
    message_id: Optional[str] = Field(default=None, description="message ID")
    task_id: Optional[str] = Field(default=None, description="task ID for the message")
    kind: Literal["transcription", "text", "audio", "images", "interrupted"] = Field(
        default="text", description="the task kind of gen output"
    )
    is_first_text: bool = False
    text_content: Optional[types.Content] = Field(
        default=None,
        description="""Optional. text content when kind is transcription or text""",
    )
    is_first_audio_chunk: bool = False
    audio_blob: Optional[types.Blob] = Field(
        default=None,
        description="""Optional. audio binary blob when kind is audio""",
    )
    image_blob_list: Optional[List[types.Blob]] = Field(
        default=None,  # NOTE: gemini2.5 live don't gen image, mayge feature gen
        description="""Optional. image binary blob list when kind is images""",
    )
    interrupted: Optional[bool] = Field(
        default=None,
        description="""Flag indicating that LLM was interrupted when generating the content.
    Usually it's due to user interruption during a bidi streaming.""",
    )
