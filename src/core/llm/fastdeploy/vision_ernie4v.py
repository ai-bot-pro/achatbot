from abc import abstractmethod
import logging
import os
import io
from time import perf_counter
import requests

from PIL import Image

from src.core.llm.fastdeploy.base import FastdeployBase
from src.common.session import Session

try:
    from fastdeploy.engine.sampling_params import SamplingParams

except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "you need to see https://paddlepaddle.github.io/FastDeploy/get_started/installation/nvidia_gpu/"
    )
    raise Exception(f"Missing module: {e}")


class FastdeployVisionERNIE4v(FastdeployBase):
    TAG = "llm_fastdeploy_vision_ernie4v"

    def init(self):
        pass

    def warmup(self):
        pass

    def generate(self, session: Session, **kwargs):
        """
        prompt = [
            {
                "type": "image",
                "image": PIL.Image,
            },
            {"type": "text", "text": "这张图片的内容是什么"},
        ]
        """
        prompt = session.ctx.state["prompt"]
        assert len(prompt) > 0
        message = {"role": self.args.user_role, "content": prompt}
        if session.ctx.client_id not in self.session_chat_history:
            self.session_chat_history[session.ctx.client_id] = self._chat_history
        self.session_chat_history[session.ctx.client_id].append(message)
        chat_history = self.session_chat_history[session.ctx.client_id].to_list()
        logging.info(f"{session.ctx.client_id} chat_history:{chat_history}")
        prompt = self.tokenizer.apply_chat_template(chat_history, tokenize=False)
        images, videos = [], []
        for message in chat_history:
            content = message["content"]
            if not isinstance(content, list):
                continue
            for part in content:
                if part["type"] == "image_url":
                    img = part["image_url"]
                    if isinstance(img, Image.Image):
                        images.append(img)
                    if isinstance(img, str) and (
                        img.startswith("https://") or img.startswith("http://")
                    ):
                        image_bytes = requests.get(img).content
                        img = Image.open(io.BytesIO(image_bytes))
                        images.append(img)
                elif part["type"] == "video_url":
                    video = part["video_url"]
                    if isinstance(video, str) and (
                        video.startswith("https://") or video.startswith("http://")
                    ):
                        video_bytes = requests.get(video).content
                    else:
                        assert isinstance(video, bytes)
                        video_bytes = video
                    videos.append({"video": video_bytes, "max_frames": 30})

        prompts = {"prompt": prompt, "multimodal_data": {"image": images, "video": videos}}
        prompts["request_id"] = session.ctx.client_id
        prompts["max_tokens"] = self.engine.cfg.max_model_len
        logging.info(f"{prompts=}")
        sampling_params = SamplingParams(
            n=1,
            repetition_penalty=kwargs.get(
                "repetition_penalty", self.gen_args.lm_gen_repetition_penalty
            ),
            temperature=kwargs.get("temperature", self.gen_args.lm_gen_temperature),
            # top_k=kwargs.get("top_k", self.gen_args.lm_gen_top_k),
            top_p=kwargs.get("top_p", self.gen_args.lm_gen_top_p),
            max_tokens=kwargs.get("max_tokens", self.gen_args.lm_gen_max_tokens),
            reasoning_max_tokens=kwargs.get(
                "reasoning_max_tokens", self.gen_args.lm_gen_reasoning_max_tokens
            ),
            stop=kwargs.get("stop", self.gen_args.lm_gen_stops),
            stop_token_ids=kwargs.get("stop_token_ids", self.gen_args.lm_gen_stop_ids),
        )
        logging.info(f"{sampling_params=}")
        enable_thinking = kwargs.get("thinking", True)
        self.engine.add_requests(prompts, sampling_params, enable_thinking=enable_thinking)

        generated_text = ""
        start = perf_counter()
        times = []
        is_output_think = self.gen_args.lm_gen_think_output
        is_thinking = False
        is_answer = True
        think_text = ""

        for result in self.engine._get_generated_tokens(prompts["request_id"]):
            times.append(perf_counter() - start)
            if result.outputs and result.outputs.token_ids and len(result.outputs.token_ids) > 0:
                new_text = self.tokenizer.decode(result.outputs.token_ids)
                # print(new_text, flush=True, end="\n")
                if ("<think>" in new_text or enable_thinking is True) and think_text == "":
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
                    continue

                if "</s>" in new_text:
                    is_answer = False
                    break

                if is_answer is True:
                    generated_text += new_text
                    yield new_text
            start = perf_counter()
        yield "."  # end the sentence for downstream process sentence, e.g.: tts
        logging.info(f"{generated_text=} TTFT: {times[0]:.4f}s total time: {sum(times):.4f}s")
        self.session_chat_history[session.ctx.client_id].append(
            {"role": "assistant", "content": [{"type": "text", "text": generated_text}]}
        )


"""
MODEL=./models/baidu/ERNIE-4.5-VL-28B-A3B-Paddle python -m src.core.llm.fastdeploy.vision_ernie4v
"""
if __name__ == "__main__":
    import uuid
    import os
    import time
    import PIL

    from fastdeploy.engine.args_utils import EngineArgs
    from src.common.types import SessionCtx, TEST_DIR
    from src.types.llm.fastdeploy import FastDeployEngineArgs, LMGenerateArgs
    from src.common.logger import Logger

    Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)

    model = os.getenv("MODEL", "baidu/ERNIE-4.5-0.3B")
    generator = FastdeployVisionERNIE4v(
        **FastDeployEngineArgs(
            serv_args=EngineArgs(
                model=model,
                tensor_parallel_size=int(os.getenv("TP", 1)),
                quantization=os.getenv("QUANTIZATION", "wint4"),
                max_model_len=32768,
                enable_mm=True,
                limit_mm_per_prompt={"image": 100},
                reasoning_parser="ernie-45-vl",
            ).__dict__
        ).__dict__,
    )

    session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)
    image_file = os.path.join(TEST_DIR, "img_files", "03-Confusing-Pictures")
    session.ctx.state["prompt"] = [
        {"type": "image", "image": PIL.Image.open(image_file)},
        {"type": "text", "text": "这张图片的内容是什么"},
    ]
    first = True
    start_time = time.perf_counter()
    for text in generator.generate(session, thinking=True):
        print(text, flush=True, end="")
