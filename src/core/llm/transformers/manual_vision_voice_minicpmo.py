import logging
import os
from threading import Thread
import time
import uuid

import numpy as np

try:
    import torch
    import librosa
    import soundfile as sf
    from PIL import Image
    from transformers import AutoModel, AutoTokenizer
    # from auto_gptq import AutoGPTQForCausalLM
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use omni MiniCPMo, you need to `pip install achatbot[llm_transformers_manual_vision_voice_minicpmo]`,"
    )
    raise Exception(f"Missing module: {e}")

from src.common.random import set_all_random_seed
from src.common.utils.helper import get_device, print_model_params
from src.common.chat_history import ChatHistory
from src.common.session import Session
from src.types.llm.transformers import TransformersLMArgs
from src.common.types import RECORDS_DIR
from .base import TransformersBaseLLM


class TransformersManualMiniCPMO(TransformersBaseLLM):
    # from: https://huggingface.co/openbmb/MiniCPM-o-2_6-int4/blob/main/modeling_minicpmo.py#L168
    RATE = 24000  # vocos config rate: 24000

    def __init__(self, **args) -> None:
        logging.debug(f"args:{args}")
        # session sys settings
        # language
        self.language = args.pop("language", "zh")
        # Interaction mode
        # "default": default system prompt and not refer to any task
        # "omni": input video and audio simultaneously
        # "audio_assistant": Default voice-only mode, the model will use the ref_audio's voice to reply user's question as a helpful assistant.
        # "audio_roleplay": Roleplay voice-only mode, the model will use the ref_audio's voice to reply, and also role-play the character based on the audio prompt.
        # "voice_cloning": TTS mode, the model will clone the voice of ref_audio.
        self.interaction_mode = args.pop("interaction_mode", "omni")
        # reference audio
        ref_audio_path = args.pop("ref_audio_path", None)
        self.ref_audio = None
        if ref_audio_path is not None:
            self.ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)

        # init vision/voice
        self.init_vision = args.pop("init_vision", True)
        self.init_audio = args.pop("init_audio", True)
        self.init_tts = args.pop("init_tts", True)

        # whether gen result audio (tts)
        self.generate_audio = args.pop("generate_audio", True)
        # whether save result audio
        self.save_output = args.pop("save_output", False)

        # whether use gptq ckpt
        self.use_gptq_ckpt = args.pop("use_gptq_ckpt", False)

        self.args = TransformersLMArgs(**args)
        if self.args.lm_torch_dtype != "auto":
            self.torch_dtype = getattr(torch, self.args.lm_torch_dtype)
        else:
            self.torch_dtype = "auto"

        self.args.lm_device = self.args.lm_device or get_device()
        logging.info("TransformersLMArgs: %s", self.args)

        # load omni model default, the default init_vision/init_audio/init_tts is True

        # if load vision-only model, please set init_audio=False and init_tts=False
        # if load audio-only model, please set init_vision=False
        if self.use_gptq_ckpt is True:
            self.init_gptq_llm()
        else:
            if self.args.lm_device_map:
                self._model = AutoModel.from_pretrained(
                    self.args.lm_model_name_or_path,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    #!NOTE: https://github.com/huggingface/transformers/issues/20896
                    # device_map for multi cpu/gpu with accelerate
                    # https://github.com/openai/whisper/discussions/1948
                    # flash_attention_2 is use in device map
                    device_map=self.args.lm_device_map,
                    attn_implementation=self.args.lm_attn_impl,
                    torch_dtype=self.torch_dtype,
                    init_vision=self.init_vision,
                    init_audio=self.init_audio,
                    init_tts=self.init_tts,
                ).eval()
            else:
                self._model = (
                    AutoModel.from_pretrained(
                        self.args.lm_model_name_or_path,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        attn_implementation=self.args.lm_attn_impl,
                        torch_dtype=self.torch_dtype,
                        init_vision=self.init_vision,
                        init_audio=self.init_audio,
                        init_tts=self.init_tts,
                    )
                    .to(self.args.lm_device)
                    .eval()
                )
        print_model_params(self._model, self.TAG)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.args.lm_model_name_or_path, trust_remote_code=True
        )

        # In addition to vision-only mode or open generate audio,
        # tts processor and vocos also needs to be initialized
        if self.init_audio is False and self.init_tts is False:
            self._model.init_tts()
        elif self.generate_audio is True:
            self._model.init_tts()

        self._sys_msg = None
        if self.interaction_mode is not None:
            self._sys_msg = self._model.get_sys_prompt(
                ref_audio=self.ref_audio, mode=self.interaction_mode, language=self.language
            )

        self._chat_history = ChatHistory(self.args.chat_history_size)
        if self._sys_msg:
            self._chat_history.init(self._sys_msg)

        self.warmup()
        self.reset_session()

    def init_gptq_llm(self):
        try:
            from auto_gptq import AutoGPTQForCausalLM
        except ModuleNotFoundError as e:
            logging.error(f"Exception: {e}")
            logging.error(
                """In order to use omni MiniCPMo, you need to do below steps:
                ```shell
                git clone https://github.com/OpenBMB/AutoGPTQ.git -b minicpmo
                cd AutoGPTQ && git checkout minicpmo && pip install -vvv --quiet --no-build-isolation -e .
                ```
                chose openbmb/MiniCPM-o-2_6-int4 ckpt from huggingface hub
                """
            )
            raise Exception(f"Missing module: {e}")

        self._model = AutoGPTQForCausalLM.from_quantized(
            self.args.lm_model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device="cuda:0",  # just for gpu acceleration, load in cuda:0
            attn_implementation=self.args.lm_attn_impl,
            disable_exllama=True,
            disable_exllamav2=True,
            init_vision=self.init_vision,
            init_audio=self.init_audio,
            init_tts=self.init_tts,
        ).eval()

    def reset_session(self):
        # a new conversation need reset session first,
        # it will reset the kv-cache
        self._model.reset_session()

    def set_system_prompt(self, **kwargs):
        # session sys settings
        # language
        self.language = kwargs.get("language", self.language)
        # interation mode
        # "default": default system prompt and not refer to any task
        # "omni": input video and audio simultaneously
        # "audio_assistant": Default voice-only mode, the model will use the ref_audio's voice to reply user's question as a helpful assistant.
        # "audio_roleplay": Roleplay voice-only mode, the model will use the ref_audio's voice to reply, and also role-play the character based on the audio prompt.
        # "voice_cloning": TTS mode, the model will clone the voice of ref_audio.
        self.interaction_mode = kwargs.get("interaction_mode", self.interaction_mode)
        # reference audio
        ref_audio_path = kwargs.get("ref_audio_path", None)
        if ref_audio_path is not None:
            self.ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)

        if self.interaction_mode is not None:
            self._sys_msg = self._model.get_sys_prompt(
                ref_audio=self.ref_audio, mode=self.interaction_mode, language=self.language
            )

    def warmup(self):
        if self.args.warmup_steps < 0:
            return
        logging.info(f"Warming up {self.__class__.__name__} device: {self._model.device}")
        if "cuda" in str(self._model.device):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()

        dummy_input_text = self.args.warnup_prompt
        content = [dummy_input_text]
        if self.init_vision is True:
            dummy_pil_image = Image.new("RGB", (100, 100), color="white")
            content = [dummy_pil_image, dummy_input_text]
        msgs = [
            {
                "role": "user",
                "content": content,
            }
        ]

        for i in range(self.args.warmup_steps):
            self._sys_msg and self._model.streaming_prefill(
                session_id="", msgs=[self._sys_msg], tokenizer=self._tokenizer
            )
            self._model.streaming_prefill(session_id="", msgs=msgs, tokenizer=self._tokenizer)
            streamer = self._model.streaming_generate(
                session_id="",
                tokenizer=self._tokenizer,
                max_new_tokens=self.args.lm_gen_max_new_tokens,
                min_new_tokens=self.args.lm_gen_min_new_tokens,
                do_sample=False if self.args.lm_gen_temperature == 0 else True,
                temperature=self.args.lm_gen_temperature,
                top_p=self.args.lm_gen_top_p,
                top_k=self.args.lm_gen_top_k,
                repetition_penalty=self.args.lm_gen_repetition_penalty,
                generate_audio=False,
            )
            for step in range(self.args.warmup_steps):
                for _ in streamer:
                    times = []
                    start_time = time.perf_counter()
                    for _ in streamer:
                        times.append(time.perf_counter() - start_time)
                        start_time = time.perf_counter()
                    logging.info(f"step {step} warnup TTFT time: {times[0]} s")
                    step += 1

        if "cuda" in str(self._model.device):
            end_event.record()
            torch.cuda.synchronize()
            logging.info(
                f"{self.__class__.__name__}:  warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s"
            )

    def get_prompt(self, session: Session) -> list:
        prompt = []
        if isinstance(session.ctx.state["prompt"], list):
            prompt = session.ctx.state["prompt"]
        return prompt

    @torch.inference_mode()
    def generate(self, session: Session, **kwargs):
        logging.debug(f"session state: {session.ctx.state} kwargs: {kwargs}")
        seed = kwargs.get("seed", self.args.lm_gen_seed)
        set_all_random_seed(seed)

        prompt = self.get_prompt(session)
        logging.debug(f"prompt: {prompt}")

        message = {"role": "user", "content": prompt}
        # history save in kv_cache: llm_past_key_values
        # self._chat_history.append(message)
        # msgs = self._chat_history.to_list()

        # 1. prefill system prompt and msgs and decode first token (prefill_decode(decode first token for TTFT(Time to First Token)))
        # TODO: if deploy on serve, change hf streaming_prefill method to support chat history @weedge
        self._sys_msg and self._model.streaming_prefill(
            session_id=session.ctx.client_id, msgs=[self._sys_msg], tokenizer=self._tokenizer
        )
        # https://huggingface.co/openbmb/MiniCPM-o-2_6/blob/main/modeling_minicpmo.py#L1049
        self._model.streaming_prefill(
            session_id=session.ctx.client_id, msgs=[message], tokenizer=self._tokenizer
        )

        # 2. generate(decoding) (decode_n_tokens for TPOT(Time per Output Token))
        # https://huggingface.co/openbmb/MiniCPM-o-2_6/blob/main/modeling_minicpmo.py#L1168
        lm_gen_temperature = kwargs.get("temperature", self.args.lm_gen_temperature)
        generate_audio = kwargs.get("generate_audio", self.generate_audio)
        streamer = self._model.streaming_generate(
            session_id=session.ctx.client_id,
            tokenizer=self._tokenizer,
            max_new_tokens=kwargs.get("max_new_tokens", self.args.lm_gen_max_new_tokens),
            min_new_tokens=kwargs.get("min_new_tokens", self.args.lm_gen_min_new_tokens),
            do_sample=False if lm_gen_temperature == 0 else True,
            temperature=lm_gen_temperature,
            top_p=kwargs.get("top_p", self.args.lm_gen_top_p),
            top_k=kwargs.get("top_k", self.args.lm_gen_top_k),
            repetition_penalty=kwargs.get(
                "repetition_penalty", self.args.lm_gen_repetition_penalty
            ),
            generate_audio=generate_audio,
        )

        audios = []
        text = ""

        if generate_audio:
            # https://huggingface.co/openbmb/MiniCPM-o-2_6/blob/main/modeling_minicpmo.py#L1496
            # _generate_mel_spec_audio_streaming -> streamer (OmniOutput)
            for r in streamer:  # OmniOutput
                sampling_rate = r.sampling_rate
                audios.append(r.audio_wav)
                if "<|tts_eos|>" in r.text:
                    r.text = r.text.replace("<|tts_eos|>", "")
                text += r.text
                time.sleep(0.05)
                yield r.__dict__  # OmniOutput.__dict__

            if self.save_output is True:
                res = np.concatenate(audios)
                session_dir = os.path.join(RECORDS_DIR, session.ctx.client_id)
                os.makedirs(session_dir, exist_ok=True)
                path = os.path.join(session_dir, f"{uuid.uuid4()}.wav")
                sf.write(path, res, samplerate=sampling_rate)
                logging.info(f"save to {path}")
        else:
            # https://huggingface.co/openbmb/MiniCPM-o-2_6/blob/main/modeling_minicpmo.py#L1230
            # llm_generate_chunk -> dict {"text":text}
            for r in streamer:
                # r["text"] = r["text"].split("<|tts_eos|>")[0]
                if "<|tts_eos|>" in r["text"]:
                    r["text"] = r["text"].replace("<|tts_eos|>", "")
                text += r["text"]
                time.sleep(0.05)
                yield r
        # self._chat_history.append({"role": "assistant", "content": [text]})
        logging.info(f"gen text: {text}")
        session.increment_chat_round()
        if self.args.chat_history_size and session.chat_round == self.args.chat_history_size:
            logging.info(f"chat round {session.chat_round} session {session.ctx.client_id} reset")
            self._model.reset_session()


class TransformersManualVisionMiniCPMO(TransformersManualMiniCPMO):
    """
        Vision only
    vision(images + text) -> AutoProcessor(MiniCPMVImageProcessor(images),MiniCPMOTokenizerFast(text)->MiniCPMOProcessor) -> tokens(text input_ids + images batch feature) -> SiglipVisionTransformer -> vllm embeddings (vision, vision_hidden_states)-> Qwen2ForCausalLM(Qwen2.5-7B,use Qwen2 LM) -> text + hidden stats(embeddings)
    - AutoProcessor: https://huggingface.co/openbmb/MiniCPM-o-2_6-int4/blob/main/preprocessor_config.json
        - ⭐️ MiniCPMOProcessor: https://huggingface.co/openbmb/MiniCPM-o-2_6-int4/blob/main/processing_minicpmo.py#L38
        - MiniCPMOTokenizerFast: https://huggingface.co/openbmb/MiniCPM-o-2_6-int4/blob/main/tokenization_minicpmo_fast.py
        - MiniCPMVImageProcessor: https://huggingface.co/openbmb/MiniCPM-o-2_6-int4/blob/main/image_processing_minicpmv.py#L121
    - ⭐️ SiglipVisionTransformer: https://huggingface.co/openbmb/MiniCPM-o-2_6-int4/blob/main/modeling_navit_siglip.py#L850
    - Qwen2ForCausalLM: https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/qwen2/modeling_qwen2.py
    """

    TAG = "llm_transformers_manual_vision_minicpmo"

    def __init__(self, **args) -> None:
        # vision-only vision understanding I1->T2
        args["init_vision"] = True  # vision
        args["init_audio"] = False  # no asr
        args["init_tts"] = False  # no tts
        args["generate_audio"] = False  # no gen audio

        super().__init__(**args)

    def get_prompt(self, session: Session) -> list:
        assert isinstance(session.ctx.state["prompt"], list)
        assert len(session.ctx.state["prompt"]) >= 2
        for item in session.ctx.state["prompt"][:-1]:  # user images prompt
            assert isinstance(item, Image.Image)
        assert isinstance(session.ctx.state["prompt"][-1], str)  # user str prompt

        prompt = session.ctx.state["prompt"]
        return prompt

    @torch.inference_mode()
    def generate(self, session: Session, **kwargs):
        for item in super().generate(session, **kwargs):
            yield item["text"]


class TransformersManualInstructSpeechMiniCPMO(TransformersManualMiniCPMO):
    r"""
    Instruction Speech:
    instruction text -> AutoProcessor(MiniCPMOTokenizerFast(text)->MiniCPMOProcessor) -> tokens(text input_ids) -> Qwen2ForCausalLM(Qwen2.5-7B,use Qwen2 LM) -> text + hidden stats(embeddings) -> ChatTTSProcessor(text_tokenizer:BertTokenizerFast) -> ConditionalChatTTS(ChatTTS-200M, use Llama2 LM) ->  audio vq codes -> _generate_mel_spec -> mel spectrograms -> vocos decode_mel_to_audio -> audio(waveform)
    - AutoProcessor: https://huggingface.co/openbmb/MiniCPM-o-2_6-int4/blob/main/preprocessor_config.json
        - ⭐️ MiniCPMOProcessor: https://huggingface.co/openbmb/MiniCPM-o-2_6-int4/blob/main/processing_minicpmo.py
        - MiniCPMOTokenizerFast: https://huggingface.co/openbmb/MiniCPM-o-2_6-int4/blob/main/tokenization_minicpmo_fast.py
    - Qwen2ForCausalLM: https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/qwen2/modeling_qwen2.py
    - ConditionalChatTTS(⭐️ nice code ⭐️): https://huggingface.co/openbmb/MiniCPM-o-2_6/blob/main/modeling_minicpmo.py#L2590
        - VQ-VAE(DVAE, vq use GroupedResidualFSQ): https://huggingface.co/openbmb/MiniCPM-o-2_6/blob/main/modeling_minicpmo.py#L2350
    - Vocos: https://huggingface.co/openbmb/MiniCPM-o-2_6/blob/main/modeling_minicpmo.py#L168
        - from https://github.com/gemelo-ai/vocos

    instruction: https://voxinstruct.github.io/VoxInstruct/
    Note: 适合离线一次生成语音，多次生成语音不一致
    colab笔记: https://github.com/weedge/doraemon-nb/blob/main/OpenBMB_MiniCPMo.ipynb
    """

    TAG = "llm_transformers_manual_instruct_speech_minicpmo"

    def __init__(self, **args) -> None:
        # tts T1->A2
        args["init_vision"] = False  # no vision
        args["init_audio"] = False  # no asr
        args["init_tts"] = True  # tts
        args["generate_audio"] = True  # gen audio

        # args["interaction_mode"] = None
        # instruct2speech | voice_cloning
        self.tts_task = args.pop("tts_task", "voice_cloning")
        if self.tts_task == "voice_cloning":
            assert os.path.exists(args.get("ref_audio_path"))
            args["init_audio"] = True  # voice_cloning use need use ref audio,
            args["interaction_mode"] = "voice_cloning"

        super().__init__(**args)

    def get_prompt(self, session: Session) -> list:
        assert isinstance(session.ctx.state["prompt"], list)
        if self.tts_task == "instruct2speech":
            assert len(session.ctx.state["prompt"]) == 1  # instruction text
        if self.tts_task == "voice_cloning":
            assert (
                len(session.ctx.state["prompt"]) == 2
            )  # instruction + tts text with cloning audio

        prompt = session.ctx.state["prompt"]
        return prompt

    @torch.inference_mode()
    def generate(self, session: Session, **kwargs):
        for item in super().generate(session, **kwargs):
            audio_wav = item.pop("audio_wav", None)
            yield audio_wav


class TransformersManualTextSpeechMiniCPMO(TransformersManualMiniCPMO):
    r"""
    text -> chat lm -> gen text -> speech lm -> audio
    """

    TAG = "llm_transformers_manual_text_speech_minicpmo"

    def __init__(self, **args) -> None:
        # tts T1->T2A2
        args["init_vision"] = False  # no vision
        args["init_audio"] = True  # voice_cloning use need use ref audio,
        args["init_tts"] = True  # tts
        args["generate_audio"] = True  # gen audio

        args["interaction_mode"] = "voice_cloning"
        # voice cloning chat
        self.tts_task = args.pop("tts_task", "voice_cloning_chat")
        assert os.path.exists(args.get("ref_audio_path"))

        super().__init__(**args)

    def get_prompt(self, session: Session) -> list:
        assert isinstance(session.ctx.state["prompt"], list)
        assert len(session.ctx.state["prompt"]) == 1
        assert isinstance(session.ctx.state["prompt"][0], str)

        prompt = session.ctx.state["prompt"]
        return prompt


class TransformersManualAudioMiniCPMO(TransformersManualMiniCPMO):
    r"""
    Audio Understanding:
    audio -> AutoProcessor(WhisperFeatureExtractor(audio)->MiniCPMOProcessor) -> tokens(audio_features) -> MiniCPMWhisperEncoder -> audio embeddings -> Qwen2ForCausalLM(Qwen2.5-7B,use Qwen2 LM) -> text + hidden stats(embeddings)
    - AutoProcessor: https://huggingface.co/openbmb/MiniCPM-o-2_6-int4/blob/main/preprocessor_config.json
        - ⭐️ MiniCPMOProcessor: https://huggingface.co/openbmb/MiniCPM-o-2_6-int4/blob/main/processing_minicpmo.py
        - WhisperFeatureExtractor: https://github.com/huggingface/transformers/blob/v4.42.2/src/transformers/models/whisper/feature_extraction_whisper.py#L36
    - ⭐️ MiniCPMWhisperEncoder: https://huggingface.co/openbmb/MiniCPM-o-2_6/blob/main/modeling_minicpmo.py#L1973
    - Qwen2ForCausalLM: https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/qwen2/modeling_qwen2.py
    """

    TAG = "llm_transformers_manual_audio_minicpmo"

    def __init__(self, **args) -> None:
        # audio understanding A1->T2
        args["init_vision"] = False  # no vision
        args["init_audio"] = True  # asr
        args["init_tts"] = False  # no tts
        args["generate_audio"] = False  # no gen audio

        audio_task = args.pop("audio_task", "asr")
        # ASR task
        self.task_prompt = (
            "Please listen to the audio snippet carefully and transcribe the content."
        )
        if args.get("language", "zh") == "zh":
            self.task_prompt = "请仔细听这段音频片段，并将其内容逐字记录。"
        # Speaker Analysis task
        if audio_task == "speaker_analysis":
            self.task_prompt = "Based on the speaker's content, speculate on their gender, condition, age range, and health status."
        # General Audio Caption
        if audio_task == "audio_caption":
            self.task_prompt = "Summarize the main content of the audio."
        # General Sound Scene Tagging
        if audio_task == "audio_tagging":
            self.task_prompt = (
                "Utilize one keyword to convey the audio's content or the associated scene."
            )

        self.task_prompt += "\n"

        super().__init__(**args)

    def get_prompt(self, session: Session):
        assert isinstance(session.ctx.state["prompt"], list)
        assert len(session.ctx.state["prompt"]) == 2
        assert isinstance(session.ctx.state["prompt"][0], str)  # task promt
        assert isinstance(session.ctx.state["prompt"][-1], np.ndarray)  # audio

        prompt = session.ctx.state["prompt"]
        prompt[0] = self.task_prompt
        return prompt

    @torch.inference_mode()
    def generate(self, session: Session, **kwargs):
        for item in super().generate(session, **kwargs):
            text = item.pop("text", "")
            if text == "":
                continue
            yield text


class TransformersManualVoiceMiniCPMO(TransformersManualMiniCPMO):
    r"""
    Voice: audio -> AutoProcessor(WhisperFeatureExtractor->MiniCPMOProcessor) -> tokens(audio_features) -> MiniCPMWhisperEncoder SiglipVisionTransformer -> audio embeddings -> Qwen2ForCausalLM(Qwen2.5-7B,use Qwen2 LM) -> text + hidden stats(embeddings) -> ChatTTSProcessor(text_tokenizer:BertTokenizerFast) -> ConditionalChatTTS(ChatTTS-200M, use Llama2 LM) ->  audio vq codes -> _generate_mel_spec -> mel spectrograms -> vocos decode_mel_to_audio -> audio(waveform)
    - AutoProcessor: https://huggingface.co/openbmb/MiniCPM-o-2_6-int4/blob/main/preprocessor_config.json
        - ⭐️ MiniCPMOProcessor: https://huggingface.co/openbmb/MiniCPM-o-2_6-int4/blob/main/processing_minicpmo.py
        - WhisperFeatureExtractor: https://github.com/huggingface/transformers/blob/v4.42.2/src/transformers/models/whisper/feature_extraction_whisper.py#L36
    - ⭐️ MiniCPMWhisperEncoder: https://huggingface.co/openbmb/MiniCPM-o-2_6/blob/main/modeling_minicpmo.py#L1973
    - Qwen2ForCausalLM: https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/qwen2/modeling_qwen2.py
    - ConditionalChatTTS(⭐️ nice code ⭐️): https://huggingface.co/openbmb/MiniCPM-o-2_6/blob/main/modeling_minicpmo.py#L2590
        - VQ-VAE(DVAE, vq use GroupedResidualFSQ): https://huggingface.co/openbmb/MiniCPM-o-2_6/blob/main/modeling_minicpmo.py#L2350
    - Vocos: https://huggingface.co/openbmb/MiniCPM-o-2_6/blob/main/modeling_minicpmo.py#L168
        - from https://github.com/gemelo-ai/vocos
    """

    TAG = "llm_transformers_manual_voice_minicpmo"

    def __init__(self, **args) -> None:
        # speech to speech A1 -> T2A2
        args["init_vision"] = False  # no vision
        args["init_audio"] = True  # asr
        args["init_tts"] = True  # tts
        args["generate_audio"] = True  # gen audio

        # mimick | audio_roleplay | audio_assistant
        self.voice_task = args.pop("voice_task", "mimick")
        interaction_mode = args.pop("interaction_mode", "omni")
        if interaction_mode in ["audio_roleplay", "audio_assistant"]:
            self.voice_task = interaction_mode

        super().__init__(**args)

    def get_prompt(self, session: Session):
        assert isinstance(session.ctx.state["prompt"], list)
        assert len(session.ctx.state["prompt"]) == 2
        assert isinstance(session.ctx.state["prompt"][0], str)  # prompt or instruction
        assert isinstance(session.ctx.state["prompt"][-1], np.ndarray)  # audio
        if self.voice_task != "mimick":
            return [session.ctx.state["prompt"][-1]]

        return session.ctx.state["prompt"]


class TransformersManualVisionVoiceMiniCPMO(TransformersManualMiniCPMO):
    r"""
    Voice: images + audio -> AutoProcessor(MiniCPMVImageProcessor(images) + WhisperFeatureExtractor(audio) -> MiniCPMOProcessor) -> tokens(image_batch_features + audio_features) -> SiglipVisionTransformer(images) + MiniCPMWhisperEncoder(audio) -> image + audio embeddings -> Qwen2ForCausalLM(Qwen2.5-7B,use Qwen2 LM) -> text + hidden stats(embeddings) -> ChatTTSProcessor(text_tokenizer:BertTokenizerFast) -> ConditionalChatTTS(ChatTTS-200M, use Llama2 LM) -> audio vq codes -> _generate_mel_spec -> mel spectrograms -> vocos decode_mel_to_audio -> audio(waveform)
    - AutoProcessor: https://huggingface.co/openbmb/MiniCPM-o-2_6-int4/blob/main/preprocessor_config.json
        - ⭐️ MiniCPMOProcessor: https://huggingface.co/openbmb/MiniCPM-o-2_6-int4/blob/main/processing_minicpmo.py
        - MiniCPMVImageProcessor: https://huggingface.co/openbmb/MiniCPM-o-2_6-int4/blob/main/image_processing_minicpmv.py#L121
        - WhisperFeatureExtractor: https://github.com/huggingface/transformers/blob/v4.42.2/src/transformers/models/whisper/feature_extraction_whisper.py#L36
    - ⭐️ SiglipVisionTransformer: https://huggingface.co/openbmb/MiniCPM-o-2_6-int4/blob/main/modeling_navit_siglip.py#L850
    - ⭐️ MiniCPMWhisperEncoder: https://huggingface.co/openbmb/MiniCPM-o-2_6/blob/main/modeling_minicpmo.py#L1973
    - Qwen2ForCausalLM: https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/qwen2/modeling_qwen2.py
    - ConditionalChatTTS(⭐️ nice code ⭐️): https://huggingface.co/openbmb/MiniCPM-o-2_6/blob/main/modeling_minicpmo.py#L2590
        - VQ-VAE(DVAE, vq use GroupedResidualFSQ): https://huggingface.co/openbmb/MiniCPM-o-2_6/blob/main/modeling_minicpmo.py#L2350
    - Vocos: https://huggingface.co/openbmb/MiniCPM-o-2_6/blob/main/modeling_minicpmo.py#L168
        - from https://github.com/gemelo-ai/vocos
    """

    # vision+audio to audio I1A1 -> T2A2
    TAG = "llm_transformers_manual_vision_voice_minicpmo"

    def get_prompt(self, session: Session) -> list:
        assert isinstance(session.ctx.state["prompt"], list)
        assert len(session.ctx.state["prompt"]) >= 2
        for item in session.ctx.state["prompt"][:-1]:  # user images prompt
            assert isinstance(item, Image.Image)
        assert isinstance(session.ctx.state["prompt"][-1], np.ndarray)

        prompt = session.ctx.state["prompt"]
        return prompt
