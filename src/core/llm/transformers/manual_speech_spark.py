import logging
from threading import Lock, Thread
from typing import Tuple

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    from deps.SparkTTS.sparktts.utils.token_parser import LEVELS_MAP, GENDER_MAP, TASK_TOKEN_MAP
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use TTS spark, you need to `pip install achatbot[llm_transformers_manual_speech_spark]`,"
    )
    raise Exception(f"Missing module: {e}")

from src.common.utils.helper import get_device
from src.common.session import Session
from src.types.llm.transformers import TransformersSpeechLMArgs
from .base import TransformersBaseLLM
from .streamer import TokenStreamer


class TransformersManualSpeechSpark(TransformersBaseLLM):
    """
    TTS: text + attribute + ref_text + audio(semantic tokens (vq indices) + global tokens (fsq indices)) ->  qwen2.5 -> semantic tokens (vq indices) + global tokens (fsq indices)
    """

    TAG = "llm_transformers_manual_speech_spark"
    DEFAULT_SYS_PROMPT = ""

    def __init__(self, **args):
        self.args = TransformersSpeechLMArgs(**args)
        self.args.lm_device = self.args.lm_device or get_device()
        logging.info("TransformersLMArgs: %s", self.args)
        self._model = AutoModelForCausalLM.from_pretrained(self.args.lm_model_name_or_path)
        self._model.eval().to(self.args.lm_device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.lm_model_name_or_path)

        self.warmup()

    def process_prompt_control(
        self,
        gender: str,
        pitch: str,
        speed: str,
        text: str,
    ):
        """
        Process input for voice creation.

        Args:
            gender (str): female | male.
            pitch (str): very_low | low | moderate | high | very_high
            speed (str): very_low | low | moderate | high | very_high
            text (str): The text input to be converted to speech.

        Return:
            str: Input prompt
        """
        assert gender in GENDER_MAP.keys()
        assert pitch in LEVELS_MAP.keys()
        assert speed in LEVELS_MAP.keys()

        gender_id = GENDER_MAP[gender]
        pitch_level_id = LEVELS_MAP[pitch]
        speed_level_id = LEVELS_MAP[speed]

        pitch_label_tokens = f"<|pitch_label_{pitch_level_id}|>"
        speed_label_tokens = f"<|speed_label_{speed_level_id}|>"
        gender_tokens = f"<|gender_{gender_id}|>"

        attribte_tokens = "".join([gender_tokens, pitch_label_tokens, speed_label_tokens])

        control_tts_inputs = [
            TASK_TOKEN_MAP["controllable_tts"],
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_style_label|>",
            attribte_tokens,
            "<|end_style_label|>",
        ]

        return "".join(control_tts_inputs)

    def process_prompt(
        self,
        text: str,
        global_token_ids: torch.Tensor,
        semantic_token_ids: torch.Tensor,
        ref_text: str = None,
    ) -> str:
        """
        Process input for voice cloning.

        Args:
            text (str): The text input to be converted to speech.
            global_token_ids, semantic_token_ids (tensor)
            ref_text (str, optional): Transcript of the prompt audio.

        Return:
            Tuple[str, torch.Tensor]: Input prompt; global tokens
        """

        global_tokens = "".join([f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()])

        # Prepare the input tokens for the model
        if ref_text is not None:
            semantic_tokens = "".join(
                [f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()]
            )
            inputs = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                ref_text,
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
                "<|start_semantic_token|>",
                semantic_tokens,
            ]
        else:
            inputs = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
            ]

        inputs = "".join(inputs)

        return inputs

    def warmup(self):
        if self.args.warmup_steps <= 0 or not self.args.warnup_prompt:
            logging.info("no warmup!")
            return

        prompt = self.process_prompt_control(
            "female", "moderate", "moderate", self.args.warnup_prompt
        )
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.args.lm_device)

        streamer = TokenStreamer(skip_prompt=True)
        warmup_gen_kwargs = dict(
            **model_inputs,
            streamer=streamer,
            min_new_tokens=self.args.lm_gen_min_new_tokens,
            max_new_tokens=self.args.lm_gen_max_new_tokens,
            top_k=self.args.lm_gen_top_k,
            top_p=self.args.lm_gen_top_p,
            do_sample=self.args.lm_gen_do_sample,
            temperature=self.args.lm_gen_temperature,
            repetition_penalty=self.args.lm_gen_repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        self._warmup(
            target=self._model.generate,
            kwargs=warmup_gen_kwargs,
            streamer=streamer,
        )

    # @torch.no_grad()
    @torch.inference_mode()
    def generate(self, session: Session, **kwargs):
        """
        TTS: text + attribute + ref_text + semantic tokens (vq indices) + global tokens (fsq indices) ->  qwen2.5 -> semantic tokens (vq indices) + global tokens (fsq indices)
        """
        text = session.ctx.state["prompt"]  # tts text

        gender = kwargs["gender"] if "gender" in kwargs else None
        pitch = kwargs["pitch"] if "pitch" in kwargs else "moderate"
        speed = kwargs["speed"] if "speed" in kwargs else "moderate"
        if gender is not None:
            prompt = self.process_prompt_control(gender, pitch, speed, text)
        else:
            assert "semantic_vq_indices" in session.ctx.state and isinstance(
                session.ctx.state["semantic_vq_indices"], torch.Tensor
            )
            semantic_vq_indices = session.ctx.state["semantic_vq_indices"]
            assert "global_fsq_indices" in session.ctx.state and isinstance(
                session.ctx.state["global_fsq_indices"], torch.Tensor
            )
            global_fsq_indices = session.ctx.state["global_fsq_indices"]

            ref_text = session.ctx.state["ref_text"] if "ref_text" in session.ctx.state else ""
            prompt = self.process_prompt(text, global_fsq_indices, semantic_vq_indices, ref_text)

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.args.lm_device)

        streamer = TokenStreamer(skip_prompt=True)
        generation_kwargs = dict(
            **model_inputs,
            streamer=streamer,
            max_length=8192,  # qwen2.5
            min_new_tokens=kwargs["min_new_tokens"]
            if "min_new_tokens" in kwargs
            else self.args.lm_gen_min_new_tokens,
            max_new_tokens=kwargs["max_new_tokens"]
            if "max_new_tokens" in kwargs
            else self.args.lm_gen_max_new_tokens,
            top_k=kwargs["top_k"] if "top_k" in kwargs else self.args.lm_gen_top_k,
            top_p=kwargs["top_p"] if "top_p" in kwargs else self.args.lm_gen_top_p,
            do_sample=kwargs["do_sample"] if "do_sample" in kwargs else self.args.lm_gen_do_sample,
            temperature=kwargs["temperature"]
            if "temperature" in kwargs
            else self.args.lm_gen_temperature,
            repetition_penalty=kwargs["repetition_penalty"]
            if "repetition_penalty" in kwargs
            else self.args.lm_gen_repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        logging.debug("generation_kwargs", generation_kwargs)
        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        for token_id in streamer:
            # print(token_id, end=",", flush=True)
            yield token_id
