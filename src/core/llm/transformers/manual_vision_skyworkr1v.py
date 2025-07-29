import logging
from threading import Thread
from time import perf_counter
import math
import re

from PIL import Image

try:
    import torch
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    from transformers.generation.streamers import TextIteratorStreamer

except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use Keye VLM, you need to `pip install achatbot[llm_transformers_manual_vision_skyworkr1v]`"
    )
    raise Exception(f"Missing module: {e}")


from src.common.utils.helper import get_device, print_model_params
from src.common.random import set_all_random_seed
from src.common.chat_history import ChatHistory
from src.common.session import Session
from src.types.speech.language import TO_LLM_LANGUAGE
from src.types.llm.transformers import TransformersLMArgs
from .base import TransformersBaseLLM


def split_model(model_path):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.model.rotary_emb"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0
    return device_map


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image: Image.Image, input_size=448, max_num=12):
    image = image.convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_prompt(conv_template, messages, thinking=True):
    assert messages[-1]["role"] == "user"

    for i, message in enumerate(messages):
        assert message.get("content")
        assert len(message.get("content")) > 0

        if message["role"] == "system":
            if message["content"][0]["text"]:
                conv_template.system_message = message["content"][0]["text"]
        elif message["role"] == "user":
            query = ""
            text = ""
            for item in message["content"]:
                if item["type"] == "text":
                    text = item["text"]
                if item["type"] == "image":
                    query += "<image>\n"
            query += text
            if i == len(messages) - 1:
                query = query.replace("<image>", "<IMAGE>")
            conv_template.append_message("user", query)
        elif message["role"] == "assistant":
            answer = ""
            for item in message["content"]:
                if item["type"] == "text":
                    answer += item["text"]
            conv_template.append_message("assistant", answer)
    conv_template.append_message("assistant", None)

    prompt = conv_template.get_prompt()
    if not prompt.endswith("\n<think>"):
        prompt += "\n<think>"
    if thinking is False:
        prompt = re.sub(r"\n<think>", "", prompt, count=1)

    return prompt


class TransformersManualVisionSkyworkR1V(TransformersBaseLLM):
    TAG = "llm_transformers_manual_vision_skyworkr1v"

    def __init__(self, **args) -> None:
        self.args = TransformersLMArgs(**args)
        gpu_prop = torch.cuda.get_device_properties("cuda")

        if self.args.lm_device_map is not None:
            if isinstance(self.args.lm_device_map, dict):
                customer_deivce_map = self.args.lm_device_map
                default_device_map = split_model(self.args.lm_model_name_or_path)
                self.args.lm_device_map = {**default_device_map, **customer_deivce_map}
            logging.info(f"TransformersLMArgs: {self.args}")
        if self.args.lm_device_map:
            self._model = AutoModel.from_pretrained(
                self.args.lm_model_name_or_path,
                torch_dtype=torch.bfloat16,
                #!NOTE: https://github.com/huggingface/transformers/issues/20896
                # device_map for multi cpu/gpu with accelerate
                device_map=self.args.lm_device_map,
                trust_remote_code=True,
                use_flash_attn=True if gpu_prop.major >= 8 else False,
                load_in_8bit=False,
                low_cpu_mem_usage=True,
            ).eval()
        else:
            self._model = (
                AutoModel.from_pretrained(
                    self.args.lm_model_name_or_path,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    use_flash_attn=True if gpu_prop.major >= 8 else False,
                    load_in_8bit=False,
                    low_cpu_mem_usage=True,
                )
                .eval()
                .to(self.args.lm_device)
            )

        logging.info(f"TransformersLMArgs: {self.args}")
        print_model_params(self._model, self.TAG)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.args.lm_model_name_or_path,
            use_fast=True,
            trust_remote_code=True,
        )
        self._model.img_context_token_id = self._tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.eos_token_id = self._tokenizer.convert_tokens_to_ids(
            self._model.conv_template.sep.strip()
        )

        self._chat_history = ChatHistory(self.args.chat_history_size)  # no history
        if self.args.init_chat_role and self.args.init_chat_prompt:
            self._chat_history.init(
                {
                    "role": self.args.init_chat_role,
                    "content": [{"type": "text", "text": self.args.init_chat_prompt}],
                }
            )
        self.session_chat_history = {}

        self.warmup()

    def warmup(self):
        if self.args.warmup_steps <= 0:
            return
        dummy_input_text = self.args.warmup_prompt
        dummy_pil_image = Image.new("RGB", (100, 100), color="white")
        dummy_msgs = [
            {
                "role": self.args.user_role,
                "content": [
                    {"type": "text", "text": dummy_input_text},
                    {"type": "image", "image": dummy_pil_image},
                ],
            }
        ]

        tpl = self._model.conv_template.copy()
        prompt = get_prompt(tpl, dummy_msgs)

        pixel_values = []
        for item in dummy_msgs[-1]["content"]:
            if item["type"] == "image":
                image_file = item["image"]
                pixel_values.append(load_image(image_file))
        num_patches_list = [img.size(0) for img in pixel_values]
        pixel_values = (
            torch.cat(pixel_values, dim=0).to(torch.bfloat16).to(self._model.device)
            if len(pixel_values) > 0
            else None
        )
        pixel_values is not None and logging.info(f"{pixel_values.shape=}")

        # replace <image> placeholder
        # NOTE: just process curr user image query
        for num_patches in num_patches_list:
            image_tokens = (
                "<img>" + "<IMG_CONTEXT>" * self._model.num_image_token * num_patches + "</img>"
            )
            prompt = prompt.replace("<IMAGE>", image_tokens, 1)

        inputs = self._tokenizer([prompt], padding=True, return_tensors="pt").to(self._model.device)
        for key, value in inputs.items():
            logging.info(f"{key}: {value.shape=}") if isinstance(
                value, torch.Tensor
            ) else logging.info(f"{key}: {value} {value.dtype=}")

        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)

        warmup_gen_kwargs = dict(
            **inputs,
            pixel_values=pixel_values,
            streamer=streamer,
            do_sample=True if self.args.lm_gen_temperature > 0 else False,
            temperature=self.args.lm_gen_temperature,
            top_k=self.args.lm_gen_top_k,
            top_p=self.args.lm_gen_top_p,
            repetition_penalty=self.args.lm_gen_repetition_penalty,
            min_new_tokens=1,
            max_new_tokens=128,
            eos_token_id=self.eos_token_id,
        )

        self._warmup(
            target=self._model.generate,
            kwargs=warmup_gen_kwargs,
            streamer=streamer,
        )

    @torch.inference_mode()
    def generate(self, session: Session, **kwargs):
        enable_thinking = kwargs.get("thinking", self.args.lm_gen_thinking)
        if enable_thinking is None:  # default thinking is True
            enable_thinking = True
        seed = kwargs.get("seed", self.args.lm_gen_seed)
        set_all_random_seed(seed)

        prompt = session.ctx.state["prompt"]
        assert len(prompt) > 0
        message = {"role": self.args.user_role, "content": prompt}
        self.add_chat_history(session, message)
        chat_history = self.get_session_chat_history(session.ctx.client_id)
        logging.info(f"{session.ctx.client_id} chat_history:{chat_history}")

        tpl = self._model.conv_template.copy()
        prompt = get_prompt(tpl, chat_history, thinking=enable_thinking)

        pixel_values = []
        for item in chat_history[-1]["content"]:
            if item["type"] == "image":
                image_file = item["image"]
                pixel_values.append(load_image(image_file))
        num_patches_list = [img.size(0) for img in pixel_values]
        pixel_values = (
            torch.cat(pixel_values, dim=0).to(torch.bfloat16).to(self._model.device)
            if len(pixel_values) > 0
            else None
        )
        pixel_values is not None and logging.info(f"{pixel_values.shape=}")
        logging.info(f"{prompt=}")

        # replace <image> placeholder
        # NOTE: just process current user image query
        for num_patches in num_patches_list:
            image_tokens = (
                "<img>" + "<IMG_CONTEXT>" * self._model.num_image_token * num_patches + "</img>"
            )
            prompt = prompt.replace("<IMAGE>", image_tokens, 1)

        inputs = self._tokenizer([prompt], padding=True, return_tensors="pt").to(self._model.device)
        for key, value in inputs.items():
            logging.info(f"{key}: {value.shape=}") if isinstance(
                value, torch.Tensor
            ) else logging.info(f"{key}: {value} {value.dtype=}")

        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            **inputs,
            pixel_values=pixel_values,
            streamer=streamer,
            do_sample=True
            if kwargs.get("temperature", self.args.lm_gen_temperature) > 0
            else False,
            temperature=kwargs.get("temperature", self.args.lm_gen_temperature),
            top_k=kwargs.get("top_k", self.args.lm_gen_top_k),
            top_p=kwargs.get("top_p", self.args.lm_gen_top_p),
            repetition_penalty=kwargs.get(
                "repetition_penalty", self.args.lm_gen_repetition_penalty
            ),
            min_new_tokens=kwargs.get("min_new_tokens", self.args.lm_gen_min_new_tokens),
            max_new_tokens=kwargs.get("max_new_tokens", self.args.lm_gen_max_new_tokens),
            eos_token_id=self.eos_token_id,
        )
        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        generated_text = ""
        start = perf_counter()
        times = []
        is_output_think = self.args.lm_gen_think_output
        is_thinking = False
        is_answer = True
        think_text = ""
        for new_text in streamer:
            times.append(perf_counter() - start)
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

            if "<|im_end|>" in new_text:
                is_answer = False
                start = perf_counter()
                break

            if is_answer is True:
                generated_text += new_text
                yield new_text
            start = perf_counter()
        yield "."  # end the sentence for downstream process sentence, e.g.: tts
        logging.info(f"{generated_text=} TTFT: {times[0]:.4f}s total time: {sum(times):.4f}s")
        torch.cuda.empty_cache()
        self.session_chat_history[session.ctx.client_id].append(
            {"role": "assistant", "content": [{"type": "text", "text": generated_text}]}
        )
