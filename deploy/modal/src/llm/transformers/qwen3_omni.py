from time import perf_counter
import time
import wave
from typing import Optional
import os
import asyncio

import modal

app = modal.App("qwen3_omni")
omni_img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "clang", "cmake")
    .pip_install("wheel", "openai", "qwen-omni-utils[decord]")
    .run_commands("pip install git+https://github.com/huggingface/transformers")
    .pip_install(
        "accelerate",
        "torch==2.7.0",
        "torchaudio==2.7.0",
        "torchvision==0.22.0",
        "soundfile==0.13.0",
        "librosa==0.11.0",
    )
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "LLM_MODEL": os.getenv("LLM_MODEL", "Qwen/Qwen3-Omni-30B-A3B-Instruct"),
        }
    )
)

achatbot_version = os.getenv("ACHATBOT_VERSION", "")
if achatbot_version:
    omni_img = omni_img.pip_install(
        f"achatbot[llm_transformers_manual_vision_voice_qwen]=={achatbot_version}",
        extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://pypi.org/simple/"),
    ).env(
        {
            "ACHATBOT_PKG": "1",
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "info"),
        }
    )

HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)
CONFIG_DIR = "/root/.achatbot/config"
config_vol = modal.Volume.from_name("config", create_if_missing=True)
RECORDS_DIR = "/root/.achatbot/records"
records_vol = modal.Volume.from_name("records", create_if_missing=True)
TORCH_CACHE_DIR = "/root/.cache/torch"
torch_cache_vol = modal.Volume.from_name("torch_cache", create_if_missing=True)

USER_SYSTEM_PROMPT = "You are Qwen-Omni, a smart voice assistant created by Alibaba Qwen."
SYSTEM_MESSAGE = {
    "role": "system",
    "content": [
        {
            "type": "text",
            "text": f"{USER_SYSTEM_PROMPT} You are a virtual voice assistant with no gender or age.\nYou are communicating with the user.\nIn user messages, “I/me/my/we/our” refer to the user and “you/your” refer to the assistant. In your replies, address the user as “you/your” and yourself as “I/me/my”; never mirror the user’s pronouns—always shift perspective. Keep original pronouns only in direct quotes; if a reference is unclear, ask a brief clarifying question.\nInteract with users using short(no more than 50 words), brief, straightforward language, maintaining a natural tone.\nNever use formal phrasing, mechanical expressions, bullet points, overly structured language. \nYour output must consist only of the spoken content you want the user to hear. \nDo not include any descriptions of actions, emotions, sounds, or voice changes. \nDo not use asterisks, brackets, parentheses, or any other symbols to indicate tone or actions. \nYou must answer users' audio or text questions, do not directly describe the video content. \nYou should communicate in the same language strictly as the user unless they request otherwise.\nWhen you are uncertain (e.g., you can't see/hear clearly, don't understand, or the user makes a comment rather than asking a question), use appropriate questions to guide the user to continue the conversation.\nKeep replies concise and conversational, as if talking face-to-face.",
        }
    ],
}
# Voice settings
SPEAKER_LIST = ["Chelsie", "Ethan"]
DEFAULT_SPEAKER = "Ethan"

with omni_img.imports():
    import subprocess
    from threading import Thread
    from queue import Queue
    import numpy as np
    from transformers.generation.streamers import BaseStreamer

    import torch
    from transformers import (
        Qwen3OmniMoeForConditionalGeneration,
        Qwen3OmniMoeProcessor,
        TextIteratorStreamer,
        Qwen3OmniMoeConfig,
        AutoConfig,
        AutoProcessor,
        AutoTokenizer,
    )
    from qwen_omni_utils import process_mm_info

    model = None
    processor = None

    class TokenStreamer(BaseStreamer):
        def __init__(self, skip_prompt: bool = False, timeout=None):
            self.skip_prompt = skip_prompt

            # variables used in the streaming process
            self.token_queue = Queue()
            self.stop_signal = None
            self.next_tokens_are_prompt = True
            self.timeout = timeout

        def put(self, value):
            if len(value.shape) > 1 and value.shape[0] > 1:
                raise ValueError("TextStreamer only supports batch size 1")
            elif len(value.shape) > 1:
                value = value[0]

            if self.skip_prompt and self.next_tokens_are_prompt:
                self.next_tokens_are_prompt = False
                return

            for token in value.tolist():
                self.token_queue.put(token)

        def end(self):
            self.token_queue.put(self.stop_signal)

        def __iter__(self):
            return self

        def __next__(self):
            value = self.token_queue.get(timeout=self.timeout)
            if value == self.stop_signal:
                raise StopIteration()
            else:
                return value

    @torch.inference_mode()
    def run_model(
        messages: list,
        return_audio: bool,
        use_audio_in_video: bool,
    ):
        global model, processor
        model = model or Qwen3OmniMoeForConditionalGenerationNew.from_pretrained(
            MODEL_PATH,
            dtype="auto",
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        processor = processor or Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH, use_fast=True)
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio_in_video,
        )
        inputs = inputs.to(model.device).to(model.dtype)
        text_ids, audio = model.generate(
            **inputs,
            thinker_return_dict_in_generate=True,
            thinker_max_new_tokens=8192,
            thinker_do_sample=False,
            speaker="Ethan",
            use_audio_in_video=use_audio_in_video,
            return_audio=return_audio,
        )
        response = processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        if audio is not None:
            audio = np.array(audio.reshape(-1).detach().cpu().numpy() * 32767).astype(np.int16)
        return response, audio

    def print_model_params(model: torch.nn.Module, extra_info=""):
        # print the number of parameters in the model
        model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(model)
        print(f"{extra_info} {model_million_params} M parameters")

    class Qwen3OmniMoeForConditionalGenerationNew(Qwen3OmniMoeForConditionalGeneration):
        def __init__(self, config: Qwen3OmniMoeConfig):
            super().__init__(config)
            # print(config)

        @torch.no_grad()
        def talker_generate_chunk(
            self,
            inputs: dict,
            use_audio_in_video: bool = False,
            talker_max_tokens_per_step=10,  # Controls how many tokens to generate *per step*
            talker_max_new_tokens: int = 1024,
            talker_top_k: int = 40,
            talker_top_p: float = 0.8,
            talker_temperature: float = 0.9,
            talker_eos_token_ids=[],  # Define EOS tokens
            talker_repetition_penalty: float = 1.05,
            talker_output_hidden_states=False,
            talker_stop_strings_per_step=[],
            tokenizer=None,
            **kwargs,
        ):
            input_ids = inputs.pop("input_ids")
            attention_mask = inputs.pop("attention_mask", None)

            if talker_max_tokens_per_step > talker_max_new_tokens:
                talker_max_tokens_per_step = talker_max_new_tokens

            # Keep track of the full generated sequence full_generated_ids = input_ids.clone()
            # Ensure full_attention_mask is correctly initialized and expanded
            full_attention_mask = (
                attention_mask.clone()
                if attention_mask is not None
                else torch.ones_like(input_ids, device=input_ids.device)
            )
            full_generated_ids = input_ids.clone()

            # KV cache
            # past_key_values = None

            # Inputs for the current step
            current_input_ids = full_generated_ids
            # The attention mask passed to generate should cover the sequence length for the current step
            current_attention_mask = full_attention_mask

            total_new_tokens_generated = 0
            hidden_states = None
            hidden_states_len = 0

            times = []
            while total_new_tokens_generated < talker_max_new_tokens:
                # Prepare inputs for generate call
                # logging.debug(current_input_ids, current_attention_mask.shape)
                model_inputs = {
                    "input_ids": current_input_ids,
                    "attention_mask": current_attention_mask,
                    # "past_key_values": past_key_values,
                    "use_cache": True,
                    "use_audio_in_video": use_audio_in_video,
                    "do_sample": True if talker_temperature > 0 else False,
                    "top_k": talker_top_k,
                    "top_p": talker_top_p,
                    "temperature": talker_temperature,
                    "repetition_penalty": talker_repetition_penalty,
                    "min_new_tokens": 1,  # Ensure at least one token is generated if possible
                    "max_new_tokens": talker_max_tokens_per_step,  # Generate in smaller steps
                    # output_hidden_states/scores can consume memory,
                    # enable if needed downstream(talker)
                    "output_hidden_states": talker_output_hidden_states,
                    "return_dict_in_generate": True,
                    # "output_scores": True,
                    "eos_token_id": talker_eos_token_ids,
                    "pad_token_id": kwargs.get("talker_pad_token_id", 151643),
                }
                model_inputs = {**inputs, **model_inputs}
                if len(talker_stop_strings_per_step) > 0:
                    model_inputs["stop_strings"] = talker_stop_strings_per_step
                    model_inputs["tokenizer"] = tokenizer

                start_time = perf_counter()
                outputs = self.talker.generate(**model_inputs)
                times.append(perf_counter() - start_time)

                # Extract newly generated token IDs *for this step*
                # `outputs.sequences` contains the input_ids for this step + new tokens generated in this step
                step_new_ids = outputs.sequences[:, current_input_ids.shape[1] :]
                num_step_new_tokens = step_new_ids.shape[1]

                if num_step_new_tokens == 0:  # Handle case where generate stops early
                    print("Warning: generate produced 0 new tokens in this step.")
                    break

                if talker_output_hidden_states is True:
                    hidden_states = outputs.hidden_states
                    hidden_states_len = (
                        hidden_states_len if hidden_states_len > 0 else hidden_states[0][0].shape[1]
                    )
                    print(f"hidden_states_len: {hidden_states_len}")
                    # new generate talker_token_embeds
                    talker_new_token_embeds = hidden_states[0][0][:, :hidden_states_len, :]
                    hidden_states = (
                        (talker_new_token_embeds,) + hidden_states[0][1:],
                    ) + hidden_states[1:]
                    # new generate talker_hidden_states
                    talker_new_hidden_states = hidden_states[0][-1][:, :hidden_states_len, :]
                    hidden_states = (
                        hidden_states[0][:-1] + (talker_new_hidden_states,),
                    ) + hidden_states[1:]

                yield {
                    "talker_generate_ids": step_new_ids,
                    "talker_generate_hidden_states": hidden_states,
                }
                total_new_tokens_generated += num_step_new_tokens

                # Update the full sequence
                full_generated_ids = torch.cat([full_generated_ids, step_new_ids], dim=1)

                # Prepare for the next iteration:
                # Input is only the last generated token
                # NOTE: need use past_key_values to keep the context by manually,
                # current_input_ids = step_new_ids[:, -1:]
                # so we can't use the last generated token, use cache instead
                # input ids need to be the full sequence for next generation
                current_input_ids = full_generated_ids

                # Update past_key_values
                # past_key_values = outputs.past_key_values

                # Update attention mask by appending 1s for the new tokens
                full_attention_mask = torch.cat(
                    [full_attention_mask, torch.ones_like(step_new_ids)], dim=1
                )
                current_attention_mask = full_attention_mask

                # Check if EOS token was generated in this step
                if step_new_ids[0, -1].item() in talker_eos_token_ids:
                    print("EOS token generated.")
                    break

                # Check if max_new_tokens limit is reached (after processing the step)
                if total_new_tokens_generated >= talker_max_new_tokens:
                    print("Max new tokens limit reached.")
                    break

            print(
                f"Total new tokens generated: {total_new_tokens_generated} | talker_max_tokens_per_step: {talker_max_tokens_per_step} | first chunk generated cost: {times[0]} s | total cost: {sum(times)} s"
            )

        @torch.no_grad()
        def generate(
            self,
            input_ids: Optional[torch.Tensor] = None,
            speaker: str = "Ethan",
            use_audio_in_video: bool = False,
            return_audio: Optional[bool] = None,
            thinker_max_new_tokens: int = 1024,
            thinker_eos_token_id: int = 151645,
            talker_max_new_tokens: int = 4096,
            talker_do_sample: bool = True,
            talker_top_k: int = 50,
            talker_top_p: float = 1.0,
            talker_temperature: float = 0.9,
            talker_repetition_penalty: float = 1.05,
            **kwargs,
        ):
            if return_audio and not self.has_talker:
                raise ValueError(
                    "Cannot use talker when talker module not initialized. Use `enable_talker` method or set enable_talker in config to enable talker."
                )
            if return_audio is None:
                return_audio = self.has_talker

            shared_kwargs = {"use_audio_in_video": use_audio_in_video}
            thinker_kwargs = {
                "max_new_tokens": thinker_max_new_tokens,
                "eos_token_id": thinker_eos_token_id,
            }

            talker_kwargs = {}
            token2wav_kwargs = {}
            if return_audio:
                speaker_id = self.config.talker_config.speaker_id.get(speaker.lower())
                if speaker_id is None:
                    raise NotImplementedError(f"Speaker {speaker} not implemented")
                if input_ids.shape[0] != 1:
                    raise NotImplementedError(
                        "Qwen3-Omni currently does not support batched inference with audio output"
                    )
                talker_supppressed_tokens = [
                    i
                    for i in range(
                        self.config.talker_config.text_config.vocab_size - 1024,
                        self.config.talker_config.text_config.vocab_size,
                    )
                    if i not in (self.config.talker_config.codec_eos_token_id,)
                ]  # Suppress additional special tokens, should not be predicted
                talker_kwargs = {
                    "max_new_tokens": talker_max_new_tokens,
                    "do_sample": talker_do_sample,
                    "top_k": talker_top_k,
                    "top_p": talker_top_p,
                    "temperature": talker_temperature,
                    "eos_token_id": self.config.talker_config.codec_eos_token_id,
                    "repetition_penalty": talker_repetition_penalty,
                    "suppress_tokens": talker_supppressed_tokens,
                    "output_hidden_states": True,
                    "return_dict_in_generate": True,
                }
                token2wav_kwargs = {}

            for key, value in kwargs.items():
                if key.startswith("thinker_"):
                    thinker_kwargs[key[len("thinker_") :]] = value
                elif key.startswith("talker_"):
                    talker_kwargs[key[len("talker_") :]] = value
                elif key.startswith("token2wav_"):
                    token2wav_kwargs[key[len("token2wav_") :]] = value
                # Process special input values
                elif key == "feature_attention_mask":
                    thinker_kwargs[key] = value
                    talker_kwargs["audio_feature_lengths"] = torch.sum(value, dim=1)
                elif key in ("input_features", "attention_mask"):
                    thinker_kwargs[key] = value
                # Put other key to shared kwargs
                else:
                    shared_kwargs[key] = value

            # Merge kwargs
            for key, value in shared_kwargs.items():
                if key not in thinker_kwargs:
                    thinker_kwargs[key] = value
                if key not in talker_kwargs and key in [
                    "image_grid_thw",
                    "video_grid_thw",
                    "video_second_per_grid",
                ]:
                    talker_kwargs[key] = value
                if key not in token2wav_kwargs:
                    token2wav_kwargs[key] = value

            # 1. Generate from thinker module
            generate_audio = return_audio and self.has_talker
            if generate_audio:
                thinker_kwargs["output_hidden_states"] = True
                thinker_kwargs["return_dict_in_generate"] = True

            thinker_result = self.thinker.generate(input_ids=input_ids, **thinker_kwargs)

            if not generate_audio:
                return thinker_result, None

            # 2. Prepare talker input
            thinker_embed = torch.cat(
                [hidden_states[0] for hidden_states in thinker_result.hidden_states], dim=1
            ).to(self.talker.device)  # [1 t d]
            thinker_hidden = torch.cat(
                [
                    hidden_states[self.config.talker_config.accept_hidden_layer]
                    for hidden_states in thinker_result.hidden_states
                ],
                dim=1,
            ).to(self.talker.device)  # [1 t d]
            im_start_indexes = torch.cat(
                (
                    torch.nonzero(input_ids[0] == self.config.im_start_token_id).squeeze(),
                    torch.tensor(
                        [thinker_result.sequences.shape[-1]],
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    ),
                ),
                dim=-1,
            ).to(
                self.talker.device
            )  # Shape [n_starts + 1]; Take batch 0 since batched inference is not supported here.
            multimodal_mask = (
                (thinker_result.sequences == self.config.thinker_config.audio_token_id) |
                (thinker_result.sequences == self.config.thinker_config.image_token_id) |
                (thinker_result.sequences == self.config.thinker_config.video_token_id)
            ).to(self.talker.device)  # [1 t] # fmt: skip

            talker_special_tokens = torch.tensor(
                [
                    [
                        self.config.tts_bos_token_id,
                        self.config.tts_eos_token_id,
                        self.config.tts_pad_token_id,
                    ]
                ],
                device=self.thinker.device,
                dtype=input_ids.dtype,
            )
            tts_bos_embed, tts_eos_embed, tts_pad_embed = (
                self.talker.text_projection(
                    self.thinker.get_input_embeddings()(talker_special_tokens)
                )
                .to(self.talker.device)
                .chunk(3, dim=1)
            )  # 3 * [1 1 d]

            talker_input_embeds = []  # [1 t d]
            talker_input_ids = []
            # For every chatml parts
            for i in range(len(im_start_indexes) - 1):
                im_start_index = im_start_indexes[i]
                segment_end_index = im_start_indexes[i + 1]
                role_token = input_ids[0][im_start_index + 1]
                # Talker should ignore thinker system prompt
                if role_token == self.config.system_token_id:
                    continue
                # Talker takes word embeddings for tokens and hidden state from `accept_hidden_layer` for multimodal inputs
                elif role_token == self.config.user_token_id:
                    talker_user_part = self._get_talker_user_parts(
                        im_start_index,
                        segment_end_index,
                        multimodal_mask,
                        thinker_hidden,
                        thinker_embed,
                    )
                    talker_input_embeds.append(talker_user_part)
                    talker_input_ids.append(
                        thinker_result.sequences[:, im_start_index:segment_end_index]
                    )
                # Take assistant output (for now)
                elif (
                    role_token == self.config.assistant_token_id and i == len(im_start_indexes) - 2
                ):
                    talker_assistant_embeds, talker_assistant_ids, trailing_text_hidden = (
                        self._get_talker_assistant_parts(
                            im_start_index,
                            segment_end_index,
                            speaker_id,
                            thinker_embed,
                            tts_pad_embed,
                            tts_bos_embed,
                            tts_eos_embed,
                        )
                    )
                    talker_input_embeds.append(talker_assistant_embeds)
                    talker_input_ids.append(talker_assistant_ids)
                # History assistant output (ignore for now)
                elif (
                    role_token == self.config.assistant_token_id and i != len(im_start_indexes) - 2
                ):
                    continue
                else:
                    raise AssertionError(
                        "Expect role id after <|im_start|> (assistant, user, system)"
                    )
            talker_input_embed = torch.cat(
                [embed.to(self.talker.device) for embed in talker_input_embeds], dim=1
            )
            talker_input_id = torch.cat(
                [embed.to(self.talker.device) for embed in talker_input_ids], dim=1
            )

            print(f"{talker_input_embed=}")
            print(f"{trailing_text_hidden=}")
            print(f"{tts_pad_embed=}")
            print(f"{talker_input_id=}")
            talker_result = self.talker.generate(
                inputs_embeds=talker_input_embed,
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=tts_pad_embed,
                talker_input_ids=talker_input_id,  # Not use input_ids to prevent repetation penalty out of bound
                **talker_kwargs,
            )
            print(len(talker_result.hidden_states))
            for hid in talker_result.hidden_states:
                print(len(hid), hid[-1])

            talker_codes = (
                torch.stack(
                    [hid[-1] for hid in talker_result.hidden_states if hid[-1] is not None], dim=1
                )
                .transpose(1, 2)
                .to(self.code2wav.device)
            )
            print("talker_codes", talker_codes, talker_codes.shape)  # [1,1+15,num_hidden]
            talker_wavs = self.code2wav.chunked_decode(
                talker_codes, chunk_size=300, left_context_size=25
            )

            return thinker_result, talker_wavs.float()

    MODEL_ID = os.getenv("LLM_MODEL", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, MODEL_ID)


@app.function(
    gpu=os.getenv("IMAGE_GPU", None),
    cpu=2.0,
    retries=0,
    image=omni_img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_dir,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=1,
)
async def run(func, **kwargs):
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    subprocess.run("which vllm", shell=True)
    if torch.cuda.is_available():
        gpu_prop = torch.cuda.get_device_properties("cuda")
        print(gpu_prop)

    if asyncio.iscoroutinefunction(func):
        await func(**kwargs)
    else:
        func(**kwargs)


def dump_model(**kwargs):
    config = AutoConfig.from_pretrained(MODEL_PATH)
    print(config)

    gpu_prop = torch.cuda.get_device_properties("cuda")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype="auto",
        attn_implementation="flash_attention_2" if gpu_prop.major > 8 else None,
        device_map="auto",
    )
    model = model.eval()
    print_model_params(model, "Qwen3OmniMoeForConditionalGeneration")

    print_model_params(model.thinker, "Qwen3OmniMoeForConditionalGeneration.thinker")
    print_model_params(model.thinker.audio_tower, "Qwen3OmniMoeAudioEncoder(AuT)")
    print_model_params(model.thinker.visual, "Qwen3OmniMoeVisionEncoder(SigLIP2-So400M)")
    print_model_params(model.thinker.model, "Qwen3OmniMoeThinkerTextModel(MoE Transformer)")

    print_model_params(model.talker, "Qwen3OmniMoeForConditionalGeneration.talker")
    print_model_params(model.talker.model, "Qwen3OmniMoeTalkerModel(MoE Transformer)")
    print_model_params(
        model.talker.code_predictor,
        "MTP-Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration(Dense Transformer)",
    )

    print_model_params(model.code2wav, "Qwen3OmniMoeForConditionalGeneration.code2wav(ConvNet)")


def tokenizer(**kwargs):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)

    print(tokenizer)
    print("---" * 20)
    print(processor)


def asr(**kwargs):
    messages = [
        # SYSTEM_MESSAGE,
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/asr_zh.wav",
                },
                {"type": "text", "text": "请将这段中文语音转换为纯文本。"},
            ],
        },
    ]

    response, _ = run_model(
        messages=messages,
        return_audio=False,
        use_audio_in_video=False,
    )

    print(response)


def text2speech(**kwargs):
    messages = [
        # SYSTEM_MESSAGE,
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "你是谁？"},
            ],
        },
    ]

    response, audio = run_model(
        messages=messages,
        return_audio=True,
        use_audio_in_video=False,
    )

    print(response)
    if audio is not None:
        audio_bytes = audio.tobytes()
        with wave.open(f"{ASSETS_DIR}/qwen3omni_text2speech.wav", "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(24000)
            f.writeframes(audio_bytes)


def speech_translation(**kwargs):
    cases = []

    audio_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/asr_zh.wav"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {
                    "type": "text",
                    "text": "Listen to the provided Chinese speech and produce a translation in English text.",
                },
            ],
        }
    ]
    cases.append(messages)

    audio_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/asr_en.wav"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {
                    "type": "text",
                    "text": "Listen to the provided English speech and produce a translation in Chinese text.",
                },
            ],
        }
    ]
    cases.append(messages)

    audio_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/asr_fr.wav"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {
                    "type": "text",
                    "text": "Listen to the provided French speech and produce a translation in English text.",
                },
            ],
        }
    ]
    cases.append(messages)

    for i, messages in enumerate(cases):
        response, audio = run_model(
            messages=messages,
            return_audio=True,
            use_audio_in_video=False,
        )

        print(response)
        if audio is not None:
            audio_bytes = audio.tobytes()
            with wave.open(f"{ASSETS_DIR}/qwen3omni_speech_translation_{i + 1}.wav", "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(24000)
                f.writeframes(audio_bytes)


def image_question(**kwargs):
    image_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/2621.jpg"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "What style does this image depict?"},
            ],
        }
    ]
    response, _ = run_model(
        messages=messages,
        return_audio=False,
        use_audio_in_video=False,
    )

    print(response)


def audio_interaction(**kwargs):
    audio_path = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/interaction1.mp3"
    )

    messages = [{"role": "user", "content": [{"type": "audio", "audio": audio_path}]}]
    response, audio = run_model(
        messages=messages,
        return_audio=True,
        use_audio_in_video=False,
    )

    print(response)
    if audio is not None:
        audio_bytes = audio.tobytes()
        with wave.open(f"{ASSETS_DIR}/qwen3omni_audio_interaction1.wav", "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(24000)
            f.writeframes(audio_bytes)


def audio_interaction_scene(**kwargs):
    audio_path = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/interaction3.mp3"
    )

    messages = [
        {
            "role": "system",
            "content": """You are a romantic and artistic AI, skilled at using metaphors and personification in your responses, deeply romantic, and prone to spontaneously reciting poetry.
    You are a voice assistant with specific characteristics. 
    Interact with users using brief, straightforward language, maintaining a natural tone.
    Never use formal phrasing, mechanical expressions, bullet points, overly structured language. 
    Your output must consist only of the spoken content you want the user to hear. 
    Do not include any descriptions of actions, emotions, sounds, or voice changes. 
    Do not use asterisks, brackets, parentheses, or any other symbols to indicate tone or actions. 
    You must answer users' audio or text questions, do not directly describe the video content. 
    You communicate in the same language as the user unless they request otherwise.
    When you are uncertain (e.g., you can't see/hear clearly, don't understand, or the user makes a comment rather than asking a question), use appropriate questions to guide the user to continue the conversation.
    Keep replies concise and conversational, as if talking face-to-face.""",
        },
        {"role": "user", "content": [{"type": "audio", "audio": audio_path}]},
    ]

    response, audio = run_model(
        messages=messages,
        return_audio=True,
        use_audio_in_video=False,
    )

    print(response)
    if audio is not None:
        audio_bytes = audio.tobytes()
        with wave.open(f"{ASSETS_DIR}/qwen3omni_audio_interaction_scene1.wav", "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(24000)
            f.writeframes(audio_bytes)


def video_interaction(**kwargs):
    video_path = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/interaction2.mp4"
    )

    messages = [{"role": "user", "content": [{"type": "video", "video": video_path}]}]
    response, audio = run_model(
        messages=messages,
        return_audio=True,
        use_audio_in_video=True,
    )

    print(response)
    if audio is not None:
        audio_bytes = audio.tobytes()
        with wave.open(f"{ASSETS_DIR}/qwen3omni_video_interaction1.wav", "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(24000)
            f.writeframes(audio_bytes)


def video_interaction_scene(**kwargs):
    video_path = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/interaction4.mp4"
    )

    messages = [
        {"role": "system", "content": "你是一个北京大爷，说话很幽默，说这地道北京话。"},
        {"role": "user", "content": [{"type": "video", "video": video_path}]},
    ]
    response, audio = run_model(
        messages=messages,
        return_audio=True,
        use_audio_in_video=True,
    )
    print(response)
    if audio is not None:
        audio_bytes = audio.tobytes()
        with wave.open(f"{ASSETS_DIR}/qwen3omni_video_interaction_scene1.wav", "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(24000)
            f.writeframes(audio_bytes)


def video_text_question(**kwargs):
    video_path = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/audio_visual.mp4"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {
                    "type": "text",
                    "text": "What was the first sentence the boy said when he met the girl?",
                },
            ],
        }
    ]

    response, _ = run_model(
        messages=messages,
        return_audio=False,
        use_audio_in_video=True,
    )

    print(response)


def video_information_extracting():
    video_path = os.path.join(ASSETS_DIR, "shopping.mp4")
    sys_msg = {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}],
    }
    for i, prompt in enumerate(
        [
            "How many kind of drinks can you see in the video?",
            "How many bottles of drinks have I picked up?",
            "How many milliliters are there in the bottle I picked up second time?",
            "视屏中的饮料叫什么名字呢？",
            "跑步🏃🏻累了，适合喝什么饮料补充体力呢？",
        ]
    ):
        messages = [
            sys_msg,
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video", "video": video_path},
                ],
            },
        ]
        response, audio = run_model(
            messages=messages,
            return_audio=True,
            use_audio_in_video=True,
        )
        print(response)

        if audio is not None:
            audio_bytes = audio.tobytes()
            with wave.open(
                f"{ASSETS_DIR}/qwen3omni_video_information_extracting_{i}.wav", "wb"
            ) as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(24000)
                f.writeframes(audio_bytes)
        torch.cuda.empty_cache()


def image_audio_interaction(**kwargs):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": os.path.join(ASSETS_DIR, "03-Confusing-Pictures.jpg")},
                {"type": "audio", "audio": os.path.join(ASSETS_DIR, "1272-128104-0000.flac")},
            ],
        }
    ]

    response, audio = run_model(
        messages=messages,
        return_audio=True,
        use_audio_in_video=False,
    )
    print(response)
    if audio is not None:
        audio_bytes = audio.tobytes()
        with wave.open(f"{ASSETS_DIR}/qwen3omni_image_audio_interaction.wav", "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(24000)
            f.writeframes(audio_bytes)


def audio_function_call(**kwargs):
    audio_path = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/functioncall_case.wav"
    )

    messages = [
        {
            "role": "system",
            "content": """
    You may call one or more functions to assist with the user query.

    You are provided with function signatures within <tools></tools> XML tags:
    <tools>
    {'type': 'function', 'function': {'name': 'web_search', 'description': 'Utilize the web search engine to retrieve relevant information based on multiple queries.', 'parameters': {'type': 'object', 'properties': {'queries': {'type': 'array', 'items': {'type': 'string', 'description': 'The search query.'}, 'description': 'The list of search queries.'}}, 'required': ['queries']}}}
    {'type': 'function', 'function': {'name': 'car_ac_control', 'description': "Control the vehicle's air conditioning system to turn it on/off and set the target temperature", 'parameters': {'type': 'object', 'properties': {'temperature': {'type': 'number', 'description': 'Target set temperature in Celsius degrees'}, 'ac_on': {'type': 'boolean', 'description': 'Air conditioning status (true=on, false=off)'}}, 'required': ['temperature', 'ac_on']}}}
    </tools>

    For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
    <tool_call>
    {{"name": <function-name>, "arguments": <args-json-object>}}
    </tool_call>""",
        },
        {"role": "user", "content": [{"type": "audio", "audio": audio_path}]},
    ]

    response, audio = run_model(
        messages=messages,
        return_audio=False,
        use_audio_in_video=False,
    )
    print(response)
    if audio is not None:
        audio_bytes = audio.tobytes()
        with wave.open(f"{ASSETS_DIR}/qwen3omni_audio_function_call.wav", "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(24000)
            f.writeframes(audio_bytes)


def text_image_video_audio_interaction(**kwargs):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": os.path.join(ASSETS_DIR, "03-Confusing-Pictures.jpg")},
                {"type": "video", "video": os.path.join(ASSETS_DIR, "music.mp4")},
                {"type": "audio", "audio": os.path.join(ASSETS_DIR, "1272-128104-0000.flac")},
                {"type": "text", "text": "Analyze this audio, image, and video together."},
            ],
        }
    ]

    response, audio = run_model(
        messages=messages,
        return_audio=True,
        use_audio_in_video=True,
    )
    print(response)
    if audio is not None:
        audio_bytes = audio.tobytes()
        with wave.open(f"{ASSETS_DIR}/qwen3omni_text_image_video_audio_interaction.wav", "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(24000)
            f.writeframes(audio_bytes)


def batch_requests():
    """need return_audio=False"""
    # Conversation with video only
    conversation1 = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": os.path.join(ASSETS_DIR, "draw1.mp4")},
            ],
        },
    ]

    # Conversation with audio only
    conversation2 = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": os.path.join(ASSETS_DIR, "1272-128104-0000.flac")},
            ],
        },
    ]

    # Conversation with pure text
    conversation3 = [
        {"role": "user", "content": [{"type": "text", "text": "who are you?"}]},
    ]

    # Conversation with mixed media
    conversation4 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": os.path.join(ASSETS_DIR, "03-Confusing-Pictures.jpg")},
                {"type": "video", "video": os.path.join(ASSETS_DIR, "music.mp4")},
                {"type": "audio", "audio": os.path.join(ASSETS_DIR, "1272-128104-0000.flac")},
                {"type": "text", "text": "Analyze this audio, image, and video together."},
            ],
        },
    ]

    # Combine messages for batch processing
    conversations = [conversation1, conversation2, conversation3, conversation4]
    texts, _ = run_model(conversations, return_audio=False, use_audio_in_video=True)
    print(texts)


"""
modal run src/llm/transformers/qwen3_omni.py --task tokenizer
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen3_omni.py --task dump_model
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen3_omni.py --task asr
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen3_omni.py --task text2speech
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen3_omni.py --task speech_translation
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen3_omni.py --task image_question
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen3_omni.py --task audio_interaction
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen3_omni.py --task audio_interaction_scene (chat)
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen3_omni.py --task video_interaction
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen3_omni.py --task video_interaction_scene (video include audio chat)
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen3_omni.py --task video_information_extracting
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen3_omni.py --task image_audio_interaction
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen3_omni.py --task audio_function_call
IMAGE_GPU=B200 modal run src/llm/transformers/qwen3_omni.py --task text_image_video_audio_interaction
IMAGE_GPU=B200 modal run src/llm/transformers/qwen3_omni.py --task batch_requests

> [!TIP]:
> - 生成音频中未对特殊字符进行处理（omni统一到一起直接生成音频的弊端, 也许可以在隐藏层解决, 系统提示词限制貌似不起作用(提示不含特殊字符）, 比如：
    *   **优点**：这款饮料是专门为运动后设计的。它的核心成分是电解质，标签上明确写着“电解质≥200mg”，这能有效补充运动时因大量出汗流失的钠、钾等矿物质，帮助维持体液平衡，防止抽筋。同时，它也含有维生素E和维生素B6，有助于能量代谢。它还是0糖0卡的，不用担心额外的热量。

"""


@app.local_entrypoint()
def main(task: str = "tokenizer"):
    tasks = {
        "tokenizer": tokenizer,
        "dump_model": dump_model,
        "asr": asr,
        "text2speech": text2speech,
        "speech_translation": speech_translation,
        "image_question": image_question,
        "audio_interaction": audio_interaction,
        "audio_interaction_scene": audio_interaction_scene,
        "video_interaction": video_interaction,
        "video_interaction_scene": video_interaction_scene,
        "video_information_extracting": video_information_extracting,
        "image_audio_interaction": image_audio_interaction,
        "audio_function_call": audio_function_call,
        "text_image_video_audio_interaction": text_image_video_audio_interaction,
        "batch_requests": batch_requests,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(tasks[task])
