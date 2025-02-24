import logging

from src.core.llm.transformers.manual_speech_step import TransformersManualSpeechStep


class TransformersManualVoiceStep(TransformersManualSpeechStep):
    """
    system prompt + (one short: text->speech(audio vq code(<audio_*>)) prompt) + chat prompt(text/speech((<audio_*>))) -> tokenizer encode -> token ids -> StepForCausalLM -> audio vq tokens
    with TransformersLMArgs
    """

    TAG = "llm_transformers_manual_voice_step"
    DEFAULT_SYS_PROMPT = """You are an AI designed for conversation, currently unable to connect to the internet.
when you need to sing or rap, start your response with (RAP). when you need to speak fast, you start your response with (fast). when you need to speak fast, you start your response with (slow)
Now, you need to listen to the user's voice content and respond with politely, concise, conversational text. Respond in accordance with the user's language."""
    CHINESE_SYS_PROMPT = """你是一个为对话而设计的人工智能模型，目前无法连接到互联网。
当你需要唱歌或说唱时，请以（RAP）开头。当你需要快速说话时，请以（快速）开头。当你需要慢速说话时，请以（慢速）开头。
现在，你需要倾听用户的语音内容，并以礼貌、简洁、口语化的文本进行回复。你需要尽量用户的语种进行回复。"""

    def __init__(self, **args):
        """
        use the same LM (step1) as TransformersManualSpeechStep
        """
        super().__init__(**args)
