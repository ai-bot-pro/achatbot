import logging
from threading import Lock, Thread

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use TTS llasa, you need to `pip install achatbot[llm_transformers_manual_speech_llasa]`,"
    )
    raise Exception(f"Missing module: {e}")

from src.common.utils.helper import get_device
from src.common.session import Session
from src.types.llm.transformers import TransformersSpeechLMArgs
from .base import TransformersBaseLLM
from .streamer import TokenStreamer


def ids_to_speech_tokens(speech_ids):
    speech_tokens_str = []
    for speech_id in speech_ids:
        speech_tokens_str.append(f"<|s_{speech_id}|>")
    return speech_tokens_str


def extract_speech_ids(speech_tokens_str):
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith("<|s_") and token_str.endswith("|>"):
            num_str = token_str[4:-2]

            num = int(num_str)
            speech_ids.append(num)
        else:
            logging.error(f"Unexpected token: {token_str}")
    return speech_ids


class TransformersManualSpeechLlasa(TransformersBaseLLM):
    """
    TTS: text + ref audio -> llama2 -> vq code tokens
    """

    TAG = "llm_transformers_manual_speech_llasa"
    DEFAULT_SYS_PROMPT = ""

    def __init__(self, **args):
        self.args = TransformersSpeechLMArgs(**args)
        self.args.lm_device = self.args.lm_device or get_device()
        logging.info("TransformersLMArgs: %s", self.args)
        self._model = AutoModelForCausalLM.from_pretrained(self.args.lm_model_name_or_path)
        self._model.eval().to(self.args.lm_device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.args.lm_model_name_or_path)

        # session ctx dict with lock, maybe need a session class
        self.session_lm_generat_lock = Lock()
        self.session_lm_generated_ids = {}  # session_id: ids(ptr)

        self.warmup()

    def warmup(self):
        if self.args.warmup_steps <= 0 or not self.args.warnup_prompt:
            logging.info("no warmup!")
            return

        formatted_text = (
            f"<|TEXT_UNDERSTANDING_START|>{self.args.warnup_prompt}<|TEXT_UNDERSTANDING_END|>"
        )

        # Tokenize the text
        chat = [
            {"role": "user", "content": "Convert the text to speech:" + formatted_text},
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"},
        ]

        input_ids = self._tokenizer.apply_chat_template(
            chat, tokenize=True, return_tensors="pt", continue_final_message=True
        )
        input_ids = input_ids.to("cuda")
        speech_end_id = self._tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")

        streamer = TokenStreamer(skip_prompt=True)
        warmup_gen_kwargs = dict(
            input_ids=input_ids,
            eos_token_id=speech_end_id,
            streamer=streamer,
            min_new_tokens=self.args.lm_gen_min_new_tokens,
            max_new_tokens=self.args.lm_gen_max_new_tokens,
            top_k=self.args.lm_gen_top_k,
            top_p=self.args.lm_gen_top_p,
            do_sample=self.args.lm_gen_do_sample,
            temperature=self.args.lm_gen_temperature,
            repetition_penalty=self.args.lm_gen_repetition_penalty,
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
        TTS: text + ref audio -> llama2 -> vq code tokens
        """
        prompt = session.ctx.state["prompt"]  # tts text
        speech_ids_prefix_str = ""
        if "vq_code_prompt" in session.ctx.state and isinstance(
            session.ctx.state["vq_code_prompt"], torch.Tensor
        ):
            vq_code_prompt = session.ctx.state[
                "vq_code_prompt"
            ]  # ref audio vq code tokens tensor shape (1, 1, T)
            vq_code_prompt = vq_code_prompt[0, 0, :]
            # Convert int 12345 to token <|s_12345|>
            speech_ids_prefix = ids_to_speech_tokens(vq_code_prompt)
            speech_ids_prefix_str = "".join(speech_ids_prefix)

        formatted_text = f"<|TEXT_UNDERSTANDING_START|>{prompt}<|TEXT_UNDERSTANDING_END|>"

        # Tokenize the text
        chat = [
            {"role": "user", "content": "Convert the text to speech:" + formatted_text},
            {
                "role": "assistant",
                "content": f"<|SPEECH_GENERATION_START|>{speech_ids_prefix_str}",
            },
        ]

        input_ids = self._tokenizer.apply_chat_template(
            chat, tokenize=True, return_tensors="pt", continue_final_message=True
        )
        input_ids = input_ids.to(self.args.lm_device)
        speech_end_id = self._tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
        streamer = TokenStreamer(skip_prompt=True)
        generation_kwargs = dict(
            input_ids=input_ids,
            eos_token_id=speech_end_id,
            streamer=streamer,
            max_length=2048,  # We trained our model with a max length of 2048
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
        )
        # print("generation_kwargs", generation_kwargs)
        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        session_id = session.ctx.client_id
        with self.session_lm_generat_lock:
            self.session_lm_generated_ids[session_id] = []

        for token_id in streamer:
            # print(token_id, end=",", flush=True)
            self.session_lm_generated_ids[session_id].append(token_id)

            if (
                len(self.session_lm_generated_ids[session_id])
                % self.args.lm_tokenizer_decode_batch_size
                == 0
            ):
                # print(generated_ids)
                speech_tokens = self._tokenizer.batch_decode(
                    torch.tensor(self.session_lm_generated_ids[session_id]).to(self.args.lm_device),
                    skip_special_tokens=True,
                )
                # Convert  token <|s_23456|> to int 23456
                speech_tokens = extract_speech_ids(speech_tokens)
                speech_vq_tokens = torch.tensor(speech_tokens).to(self.args.lm_device)
                yield speech_vq_tokens
                with self.session_lm_generat_lock:
                    self.session_lm_generated_ids[session_id] = []

        if len(self.session_lm_generated_ids[session_id]) > 0:  # last batch
            speech_tokens = self._tokenizer.batch_decode(
                torch.tensor(self.session_lm_generated_ids[session_id]).to(self.args.lm_device),
                skip_special_tokens=True,
            )
            # Convert  token <|s_23456|> to int 23456
            speech_tokens = extract_speech_ids(speech_tokens)
            speech_vq_tokens = torch.tensor(speech_tokens).to(self.args.lm_device)
            yield speech_vq_tokens

        with self.session_lm_generat_lock:
            self.session_lm_generated_ids.pop(session_id)
