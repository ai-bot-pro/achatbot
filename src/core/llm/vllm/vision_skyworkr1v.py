import logging
import re
import uuid
from time import perf_counter
import asyncio
import copy

from PIL import Image

try:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error("you need to `pip install achatbot[vllm]`")
    raise Exception(f"Missing module: {e}")

from src.common.random import set_all_random_seed
from src.common.interface import ILlm
from src.common.session import Session
from src.common.types import SessionCtx
from src.core.llm.vllm.base import VllmEngineBase


class VllmVisionSkyworkr1v(VllmEngineBase):
    """ """

    TAG = "llm_vllm_vision_skyworkr1v"

    async def warmup(self):
        if self.args.warmup_steps <= 0:
            return
        session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)

        dummy_pil_image = Image.new("RGB", (100, 100), color="white")
        session.ctx.state["prompt"] = [
            {"type": "image", "image": dummy_pil_image},
            {"type": "text", "text": self.args.warmup_prompt},
        ]
        for i in range(self.args.warmup_steps):
            logging.info(f"{i} warmup start")
            async for result_text in self.async_generate(
                session, thinking=self.gen_args.lm_gen_thinking
            ):
                pass

    async def async_generate(self, session: Session, **kwargs):
        """
        prompt = [
            {
                "type": "image",
                "image": PIL.Image,
            },
            {"type": "text", "text": "这张图片的内容是什么"},
        ]
        """
        enable_thinking = kwargs.get("thinking", self.gen_args.lm_gen_thinking)
        if enable_thinking is None:  # default thinking is True
            enable_thinking = True

        prompt = session.ctx.state["prompt"]
        assert isinstance(prompt, list) and len(prompt) > 0
        text = ""
        images = []
        for part in prompt:
            if part["type"] == "image":
                images.append(part["image"])
            if part["type"] == "text":
                text += part["text"]
        text = "<image>\n" * len(images) + text
        message = {"role": self.args.user_role, "content": text}

        if session.ctx.client_id not in self.session_chat_history:
            self.session_chat_history[session.ctx.client_id] = copy.deepcopy(self._chat_history)
        self.session_chat_history[session.ctx.client_id].append(message)
        chat_history = self.session_chat_history[session.ctx.client_id].to_list()
        logging.info(f"{session.ctx.client_id} chat_history:{chat_history}")

        prompt = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=True,
        )
        if enable_thinking is False:
            prompt = prompt.replace("<think>\n", "")
        logging.info(f"{prompt=}")

        # https://docs.vllm.ai/en/stable/api/inference_params.html#vllm.SamplingParams
        sampling_params = SamplingParams(
            n=1,
            seed=kwargs.get("seed") if kwargs.get("seed") else self.gen_args.lm_gen_seed,
            max_tokens=kwargs.get("max_new_tokens")
            if kwargs.get("max_new_tokens")
            else self.gen_args.lm_gen_max_new_tokens,
            temperature=kwargs.get("temperature")
            if kwargs.get("temperature")
            else self.gen_args.lm_gen_temperature,
            top_p=kwargs.get("top_p") if kwargs.get("top_p") else self.gen_args.lm_gen_top_p,
            top_k=kwargs.get("top_k") if kwargs.get("top_k") else self.gen_args.lm_gen_top_k,
            min_p=kwargs.get("min_p") if kwargs.get("min_p") else self.gen_args.lm_gen_min_p,
            # Penalizers,
            repetition_penalty=kwargs.get("repetition_penalty")
            if kwargs.get("repetition_penalty")
            else self.gen_args.lm_gen_repetition_penalty,
            min_tokens=kwargs.get("min_new_tokens")
            if kwargs.get("min_new_tokens")
            else self.gen_args.lm_gen_min_new_tokens,
            stop_token_ids=kwargs.get("stop_ids")
            if kwargs.get("stop_ids")
            else self.gen_args.lm_gen_stop_ids,
            stop=kwargs.get("stop_tokens")
            if kwargs.get("stop_tokens")
            else self.gen_args.lm_gen_stops,
        )
        # https://docs.vllm.ai/en/stable/api/vllm/v1/engine/async_llm.html?h=#vllm.v1.engine.async_llm.AsyncLLM.generate
        iterator = self.engine.generate(
            # https://docs.vllm.ai/en/stable/api/vllm/inputs/data.html#vllm.inputs.data.TextPrompt
            {
                "prompt": prompt,
                "multi_modal_data": {"image": images},
            },
            sampling_params=sampling_params,
            request_id=session.ctx.client_id,
        )

        prefix_len = 0
        generated_text = ""
        start = perf_counter()
        times = []
        is_output_think = self.gen_args.lm_gen_think_output
        is_thinking = False
        is_answer = True
        think_text = ""
        async for part in iterator:
            if not part.outputs:
                continue
            times.append(perf_counter() - start)
            text = part.outputs[0].text
            new_text = text[prefix_len:]
            prefix_len = len(text)
            if (
                ("<think>" in new_text or enable_thinking is True)
                and is_thinking is False
                and think_text == ""
            ):
                yield "思考中，请稍等。"
                is_thinking = True
            if "</think>" in new_text:
                is_thinking = False
                think_text += new_text
                logging.info(f"{think_text=}")
                new_text = new_text.replace("</think>", "")
                is_answer = True
            if is_thinking is True:
                think_text += new_text
                if is_output_think is True:
                    generated_text += new_text
                    yield new_text
                else:
                    yield None
                start = perf_counter()
                continue

            if self.tokenizer.eos_token in new_text:
                is_answer = False
                start = perf_counter()
                break

            if is_answer is True:
                generated_text += new_text
                yield new_text
            start = perf_counter()
        yield "."  # end the sentence for downstream process sentence, e.g.: tts
        self.session_chat_history[session.ctx.client_id].append(
            {"role": self.args.assistant_role, "content": generated_text}
        )
        if times:
            logging.info(f"{generated_text=} TTFT: {times[0]:.4f}s total time: {sum(times):.4f}s")
        else:
            logging.info(f"{generated_text=} total time: 0s")
