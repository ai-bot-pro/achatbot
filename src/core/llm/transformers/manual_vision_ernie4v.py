import logging
from threading import Thread
from time import perf_counter

from PIL import Image

try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    from transformers.generation.streamers import TextIteratorStreamer
    import torch

except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use ERNIE4.5 VLM, you need to `pip install achatbot[llm_transformers_manual_vision_ernie4v]`"
    )
    raise Exception(f"Missing module: {e}")


from src.common.utils.helper import print_model_params
from src.common.random import set_all_random_seed
from src.common.chat_history import ChatHistory
from src.common.session import Session
from src.types.llm.transformers import TransformersLMArgs
from .base import TransformersBaseLLM


def split_model(model_name, gpu):
    device_map = {}

    # splits layers into different GPUs (need use L4/L40S/A100-80GB for bfloat16)
    model_splits = {
        "baidu/ERNIE-4_5-VL-28B-A3B-PT_NVIDIA L4:4": {
            "model": [7, 7, 7, 7],  # 28 layer
            "vision_model": [8, 8, 8, 8],  # 32 layer
        },
        "baidu/ERNIE-4_5-VL-28B-A3B-PT_NVIDIA L40S:2": {
            "model": [11, 17],  # 28 layer
            "vision_model": [14, 18],  # 32 layer
        },
        "baidu/ERNIE-4_5-VL-28B-A3B-PT_NVIDIA A100 80GB PCIe": {
            "model": [28],  # 28 layer
            "vision_model": [32],  # 32 layer
        },
    }
    device_map["lm_head"] = 0

    num_layers_per_gpu = model_splits[model_name + "_" + gpu]["model"]
    num_layers = sum(num_layers_per_gpu)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    # exlude layer and last layer on cuda 0
    device_map["model.embed_tokens"] = 0
    device_map["model.norm"] = 0
    device_map["model.resampler_model"] = 0
    device_map[f"model.layers.{num_layers - 1}"] = 0

    num_layers_per_gpu = model_splits[model_name + "_" + gpu]["vision_model"]
    num_layers = sum(num_layers_per_gpu)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"vision_model.blocks.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model.patch_embed"] = 0
    device_map["vision_model.rotary_pos_emb"] = 0
    device_map["vision_model.ln"] = 0
    device_map[f"vision_model.blocks.{num_layers - 1}"] = 0

    return device_map


class TransformersManualVisionERNIE4v(TransformersBaseLLM):
    TAG = "llm_transformers_manual_vision_ernie4v"

    def __init__(self, **args) -> None:
        self.args = TransformersLMArgs(**args)
        gpu_prop = torch.cuda.get_device_properties("cuda")
        num_gpus = torch.cuda.device_count()
        gpu = gpu_prop.name + ":" + num_gpus

        if self.args.lm_device_map is not None:
            if isinstance(self.args.lm_device_map, dict):
                customer_deivce_map = self.args.lm_device_map
                default_device_map = split_model(
                    "/".join(self.args.lm_model_name_or_path.split("/")[-2:]), gpu
                )
                self.args.lm_device_map = {**default_device_map, **customer_deivce_map}
            logging.info(f"TransformersLMArgs: {self.args}")
        if self.args.lm_device_map:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.args.lm_model_name_or_path,
                torch_dtype=torch.bfloat16,
                #!NOTE: https://github.com/huggingface/transformers/issues/20896
                # device_map for multi cpu/gpu with accelerate
                device_map=self.args.lm_device_map,
                trust_remote_code=True,
            ).eval()
        else:
            self._model = (
                AutoModelForCausalLM.from_pretrained(
                    self.args.lm_model_name_or_path,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
                .eval()
                .to(self.args.lm_device)
            )

        logging.info(f"TransformersLMArgs: {self.args}")
        print_model_params(self._model, self.TAG)
        self._tokenizer = AutoProcessor.from_pretrained(
            self.args.lm_model_name_or_path,
            use_fast=True,
            trust_remote_code=True,
        )
        self._tokenizer.eval()
        self._model.add_image_preprocess(self._tokenizer)

        self._chat_history = ChatHistory(0)  # no history
        if self.args.init_chat_role and self.args.init_chat_prompt:
            self._chat_history.init(
                {
                    "role": self.args.init_chat_role,
                    "content": [{"type": "text", "text": self.args.init_chat_prompt}],
                }
            )

        self.warmup()

    def warmup(self):
        if self.args.warmup_steps <= 0:
            return
        dummy_input_text = self.args.warnup_prompt
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

        inputs = self._tokenizer.apply_chat_template(
            dummy_msgs,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(
            self._model.device,
            dtype=torch.bfloat16,
        )

        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)

        warmup_gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            do_sample=True if self.args.lm_gen_temperature > 0 else False,
            temperature=self.args.lm_gen_temperature,
            top_k=self.args.lm_gen_top_k,
            top_p=self.args.lm_gen_top_p,
            repetition_penalty=self.args.lm_gen_repetition_penalty,
            min_new_tokens=1,
            max_new_tokens=128,
        )

        self._warmup(
            target=self._model.generate,
            kwargs=warmup_gen_kwargs,
            streamer=streamer,
        )

    @torch.no_grad()
    def generate(self, session: Session, **kwargs):
        seed = kwargs.get("seed", self.args.lm_gen_seed)
        set_all_random_seed(seed)

        prompt = session.ctx.state["prompt"]
        assert len(prompt) > 0
        message = {"role": self.args.user_role, "content": prompt}
        self._chat_history.append(message)
        chat_history = self._chat_history.to_list()
        # logging.info(f"chat_history:{chat_history}")

        enable_thinking = kwargs.get("enable_thinking", self.args.lm_gen_thinking)
        text = self._tokenizer.apply_chat_template(
            chat_history,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking,
        ).to(
            self._model.device,
            dtype=torch.bfloat16,
        )
        image_inputs, video_inputs = self._tokenizer.process_vision_info(chat_history)
        inputs = self._tokenizer(
            [text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(self._model.device, dtype=torch.bfloat16)
        for key, value in inputs.items():
            logging.debug(f"{key}: {value.shape=}") if isinstance(
                value, torch.Tensor
            ) else logging.debug(f"{key}: {value}")

        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            **inputs,
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
            use_cache=True,
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
            if "<think>" in new_text or enable_thinking is True:
                yield "思考中，请稍等。"
                is_thinking = True
                think_text = ""
                think_text += new_text
                continue
            if "</think>" in new_text:
                is_thinking = False
                think_text += new_text
                logging.info(f"{think_text=}")
                think_text = ""
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
                continue

            if is_answer is True:
                generated_text += new_text
                yield new_text
            start = perf_counter()
        yield "."  # end the sentence for downstream process sentence, e.g.: tts
        logging.info(f"{generated_text=} TTFT: {times[0]:.4f}s total time: {sum(times):.4f}s")
        torch.cuda.empty_cache()
        self._chat_history.append(
            {"role": "assistant", "content": [{"type": "text", "text": generated_text}]}
        )
