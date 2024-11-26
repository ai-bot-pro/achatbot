import os
import logging
import sys
from typing import AsyncGenerator, Generator
import uuid

import torch
from apipeline.frames import *
from transformers import WhisperFeatureExtractor, AutoTokenizer

from deps.GLM4Voice.speech_tokenizer.modeling_whisper import WhisperVQEncoder
from deps.GLM4Voice.audio_process import AudioStreamProcessor
from deps.GLM4Voice.flow_inference import AudioDecoder
from deps.GLM4Voice.speech_tokenizer.utils import extract_speech_token

from src.core.llm.transformers.manual_voice_glm import TransformersManualVoicGLM
from src.processors.voice.base import VoiceProcessorBase
from src.types.llm.lmgen import *
from src.types.llm.transformers import TransformersLMArgs
from src.common.session import Session
from src.common.types import SessionCtx
from src.common.utils.audio_utils import bytes2TorchTensorWith16


class GLMVoiceBaseProcessor(VoiceProcessorBase):
    """
    use Voice-Tokenizer (whisper) +  GLM4-Voice-9B(text/audio) + Voice-Decoder (CosyVoice)
    - T1A1-T2A2: (text/speech)-to-(tokens) (GLM4(text)/whisper(speech encoder)) -> GLM4(llm) -- text|speech tokens --> GLM4(text decoder)|CosyVoice(speech decoder(mel->waveform)))

    Model Architecture:
    - GLM-4-Voice-Tokenizer: Trained by adding vector quantization to the encoder part of Whisper, converting continuous speech input into discrete tokens.  Each second of audio is converted into 12.5 discrete tokens.
    - GLM-4-Voice-9B: Pre-trained and aligned on speech modality based on GLM-4-9B, enabling understanding and generation of discretized speech.
    - GLM-4-Voice-Decoder: A speech decoder supporting streaming inference, retrained based on CosyVoice, converting discrete speech tokens into continuous speech output. Generation can start with as few as 10 audio tokens, reducing conversation latency.
    """

    DEFAULT_SYS_PROMPT = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. "

    def __init__(self,
                 *,
                 voice_in_args: GLMVoiceInArgs | dict = GLMVoiceInArgs(),
                 lm_gen_args: GLMInferenceArgs | dict = GLMInferenceArgs(),
                 voice_out_args: GLMVoiceOutArgs | dict = GLMVoiceOutArgs(),
                 system_prompt: str = "",
                 voice_tokenizer_path: str | None = None,  # audio encoder/ft extractor
                 model_path: str | None = None,  # gen lm and text tokenizer
                 voice_decoder_path: str | None = None,  # audio decoder
                 device: str = "cuda",
                 torch_dtype: str = "auto",  # auto,float16,bfloat16,float32
                 bnb_quant_type: str = "int4",
                 session: Session | None = None,
                 **kwargs):
        super().__init__(**kwargs)

        cur_dir = os.path.dirname(__file__)
        if bool(os.getenv("ACHATBOT_PKG", "")):
            sys.path.insert(1, os.path.join(cur_dir, '../../GLM4Voice'))
            sys.path.insert(2, os.path.join(cur_dir, '../../GLM4Voice/third_party/Matcha-TTS'))
        else:
            sys.path.insert(1, os.path.join(cur_dir, '../../../deps/GLM4Voice'))
            sys.path.insert(2, os.path.join(cur_dir,
                                            '../../../deps/GLM4Voice/third_party/Matcha-TTS'))

        self._voice_in_args = voice_in_args
        if isinstance(voice_in_args, dict):
            self._voice_in_args = GLMVoiceInArgs(**voice_in_args)
        self._lm_gen_args = lm_gen_args
        if isinstance(lm_gen_args, dict):
            self._lm_gen_args = GLMInferenceArgs(**lm_gen_args)
        self._voice_out_args = voice_out_args
        if isinstance(voice_out_args, dict):
            self._voice_out_args = GLMVoiceOutArgs(**voice_out_args)

        self._sys_prompt = system_prompt or self.DEFAULT_SYS_PROMPT
        self._voice_tokenizer_path = voice_tokenizer_path
        self._model_path = model_path
        self._voice_decoder_path = voice_decoder_path
        self._torch_dtype = torch_dtype
        self._bnb_quant_type = bnb_quant_type
        self._device = device

        self._session = session or Session(**SessionCtx(uuid.uuid4()).__dict__)

        self.reset()
        self.load_models()

    @property
    def stream_info(self) -> dict:
        """Return dict out stream info"""
        return {
            "sample_rate": self._voice_out_args.audio_sample_rate,
            "channels": self._voice_out_args.audio_channels,
        }

    def reset(self):
        # input_texts + completion_texts (audio tokenid special tag)
        self._history_texts = ""

    def load_models(self):
        logging.info("loading model weights")

        # Speech tokenizer(whisper vq encoder)/feature_extractor
        self._whisper_model = WhisperVQEncoder.from_pretrained(
            self._voice_tokenizer_path).eval().to(self._device)
        self._feature_extractor = WhisperFeatureExtractor.from_pretrained(
            self._voice_tokenizer_path)
        logging.info("speech whisper vq encoder and feature_extractor model state weight load")

        # Flow & Hift decoder with config, fixed sample rate 22050
        flow_config = os.path.join(self._voice_decoder_path, "config.yaml")
        flow_checkpoint = os.path.join(self._voice_decoder_path, 'flow.pt')
        hift_checkpoint = os.path.join(self._voice_decoder_path, 'hift.pt')
        self._audio_decoder = AudioDecoder(
            config_path=flow_config,
            flow_ckpt_path=flow_checkpoint,
            hift_ckpt_path=hift_checkpoint,
            device=self._device,
        )
        logging.info("speech audio Flow & Hift decoder model state weight load")

        # gen lm text tokenizer
        self._glm_tokenizer = AutoTokenizer.from_pretrained(
            self._model_path, trust_remote_code=True)
        logging.info("gen lm text tokenizer load")

        # gen lm
        self._glm_model = TransformersManualVoicGLM(**TransformersLMArgs(
            lm_gen_temperature=self._lm_gen_args.temperature,
            lm_gen_top_p=self._lm_gen_args.top_p,
            lm_gen_max_new_tokens=self._lm_gen_args.max_new_token,
            lm_torch_dtype=self._torch_dtype,
            lm_bnb_quant_type=self._bnb_quant_type,
            lm_device=self._device,
        ).__dict__)
        self._glm_model.warmup()
        logging.info("gen lm model state weight load and warnup")

        logging.info("model weights loaded")

    async def start(self, frame: StartFrame):
        await super().start(frame)

        self._create_push_task()

        logging.info("start done")

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        logging.info("stop done")

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        logging.info("cancel done")

    async def run_voice(self, frame: AudioRawFrame) -> AsyncGenerator[Frame, None]:
        audio_tensor = bytes2TorchTensorWith16(frame.audio)
        # default 16khz sample rate encode/extract -> audio tokens
        audio_tokens = extract_speech_token(
            self._whisper_model,
            self._feature_extractor,
            [(audio_tensor, self._voice_in_args.audio_sample_rate)]
        )[0]
        if len(audio_tokens) == 0:
            yield ErrorFrame("No audio tokens extracted")
            return

        audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
        audio_tokens = "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>"
        user_input = audio_tokens

        # history
        if "<|system|>" not in self._history_texts:
            self._history_texts += f"<|system|>\n{self._sys_prompt}"
        self._history_texts += f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n"

        self._session.ctx.state["prompt"] = self._history_texts
        iter_tokens = self._glm_model.generate(self._session)

        await self.tokens_decode_out(iter_tokens)

        yield None

    async def tokens_decode_out(self, iter_tokens):
        text_tokens, audio_tokens = [], []
        audio_offset = self._glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')
        end_token_id = self._glm_tokenizer.convert_tokens_to_ids('<|user|>')
        complete_tokens = []
        prompt_speech_feat = torch.zeros(1, 0, 80).to(self._device)
        flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64).to(self._device)
        this_uuid = str(uuid.uuid4())
        tts_speechs = []
        tts_mels = []
        prev_mel = None
        is_finalize = False

        block_size_list = [25, 50, 100, 150, 200]
        block_size_idx = 0
        block_size = block_size_list[block_size_idx]

        # default 22050 hz sample rate, match with audio_decoder 22050 hz
        audio_processor = AudioStreamProcessor(sr=self._voice_out_args.audio_sample_rate)
        for token_id in iter_tokens:
            if token_id == end_token_id:
                is_finalize = True
            if len(audio_tokens) >= block_size or (is_finalize and audio_tokens):
                if block_size_idx < len(block_size_list) - 1:
                    block_size_idx += 1
                    block_size = block_size_list[block_size_idx]
                tts_token = torch.tensor(audio_tokens, device=self._device).unsqueeze(0)

                if prev_mel is not None:
                    prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)

                # gen waveform and mel-spectrogram feat
                tts_speech, tts_mel = self._audio_decoder.token2wav(
                    tts_token,
                    uuid=this_uuid,
                    prompt_token=flow_prompt_speech_token.to(self._device),
                    prompt_feat=prompt_speech_feat.to(self._device),
                    finalize=is_finalize,
                )
                prev_mel = tts_mel

                # waveform tensor write to audio bytes
                audio_bytes = audio_processor.process(
                    tts_speech.clone().cpu().numpy()[0], last=is_finalize)

                tts_speechs.append(tts_speech.squeeze())
                tts_mels.append(tts_mel)
                if audio_bytes:
                    # print("audio_bytes====>", len(audio_bytes))
                    await self.queue_frame(AudioRawFrame(
                        audio=audio_bytes,
                        sample_rate=self._voice_out_args.audio_sample_rate,
                        num_channels=self._voice_out_args.audio_channels,
                    ))
                flow_prompt_speech_token = torch.cat(
                    (flow_prompt_speech_token, tts_token), dim=-1)
                audio_tokens = []

            if not is_finalize:
                complete_tokens.append(token_id)
                if token_id >= audio_offset:
                    audio_tokens.append(token_id - audio_offset)
                else:
                    text = self._glm_tokenizer.decode(
                        token_id, ignore_special_tokens=False)
                    if text:
                        await self.push_frame(TextFrame(text))
                    text_tokens.append(token_id)

        complete_text = self._glm_tokenizer.decode(
            complete_tokens, spaces_between_special_tokens=False)
        self._history_texts += complete_text


class GLMAudioVoiceProcessor(GLMVoiceBaseProcessor):
    """
    use Voice-Tokenizer (whisper) +  GLM4-Voice-9B(text/audio) + Text-Tokenizer (GLM-tokenizer), Voice-Decoder (CosyVoice)
    - A1-T2A2: (text/speech)-to-(tokens) (GLM4(text)/whisper(speech encoder)) -> GLM4(llm) -- text|speech tokens --> GLM4(text decoder)|CosyVoice(speech decoder(mel->waveform)))
    """
    pass


class GLMTextVoiceProcessor(GLMVoiceBaseProcessor):
    """
    use Text-Tokenizer (GLM-tokenizer) +  GLM4-Voice-9B(text/audio) + Text-Tokenizer (GLM-tokenizer), Voice-Decoder (CosyVoice)
    - T1-T2A2: (text/speech)-to-(tokens) (GLM4(text)/whisper(speech encoder)) -> GLM4(llm) -- text|speech tokens --> GLM4(text decoder)|CosyVoice(speech decoder(mel->waveform)))
    """

    async def run_text(self, frame: TextFrame) -> AsyncGenerator[Frame, None]:
        user_input = frame.text.strip()
        # history
        if "<|system|>" not in self._history_texts:
            self._history_texts += f"<|system|>\n{self._sys_prompt}"
        self._history_texts += f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n"

        self._session.ctx.state["prompt"] = self._history_texts
        iter_tokens = self._glm_model.generate(self._session)

        await self.tokens_decode_out(iter_tokens)

        yield None
