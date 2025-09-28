from time import perf_counter
import time
import wave
from typing import Optional, Generator, Callable
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
SYSTEM_PROMPT = f"{USER_SYSTEM_PROMPT} You are a virtual voice assistant with no gender or age.\nYou are communicating with the user.\nIn user messages, “I/me/my/we/our” refer to the user and “you/your” refer to the assistant. In your replies, address the user as “you/your” and yourself as “I/me/my”; never mirror the user’s pronouns—always shift perspective. Keep original pronouns only in direct quotes; if a reference is unclear, ask a brief clarifying question.\nInteract with users using short(no more than 50 words), brief, straightforward language, maintaining a natural tone.\nNever use formal phrasing, mechanical expressions, bullet points, overly structured language. \nYour output must consist only of the spoken content you want the user to hear. \nDo not include any descriptions of actions, emotions, sounds, or voice changes. \nDo not use asterisks, brackets, parentheses, or any other symbols to indicate tone or actions. \nYou must answer users' audio or text questions, do not directly describe the video content. \nYou should communicate in the same language strictly as the user unless they request otherwise.\nWhen you are uncertain (e.g., you can't see/hear clearly, don't understand, or the user makes a comment rather than asking a question), use appropriate questions to guide the user to continue the conversation.\nKeep replies concise and conversational, as if talking face-to-face."
SYSTEM_MESSAGE = {
    "role": "system",
    "content": [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
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
        Qwen3OmniMoeCode2Wav,
        AutoConfig,
        AutoProcessor,
        AutoTokenizer,
    )
    from qwen_omni_utils import process_mm_info

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

    @torch.inference_mode()
    def run_model_stream(
        messages: list,
        return_audio: bool,
        use_audio_in_video: bool,
        thinker_max_tokens_per_step: int = 10,
    ):
        """
        return yield {"text": str, "audio_wav": np.ndarray[T]}
        """
        global model, processor
        model = model or Qwen3OmniMoeForConditionalGenerationNew.from_pretrained(
            MODEL_PATH,
            dtype="auto",
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        processor = processor or Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH, use_fast=True)
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        print(text)
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
        # print(f"{inputs=}")
        gen_iter = model.generate_stream(
            inputs,
            use_audio_in_video=use_audio_in_video,
            thinker_max_tokens_per_step=thinker_max_tokens_per_step,
            thinker_max_new_tokens=8192,
            thinker_temperature=0.95,
            thinker_top_k=10,
            thinker_top_p=0.95,
            thinker_repetition_penalty=1.1,
            speaker="Ethan",
            return_audio=return_audio,
            tokenizer=processor.tokenizer,
            thinker_stop_strings_per_step=[",", ".", "，", "。"],
        )
        for item in gen_iter:
            thinker_ids = item.get("thinker_ids")
            print(f"{thinker_ids=}")
            assert thinker_ids is not None
            text = processor.decode(
                thinker_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            if "talker_wav" not in item:
                yield {"text": text}
            else:
                talker_wav = item.get("talker_wav")
                assert talker_wav is not None
                audio_wav = np.array(talker_wav.reshape(-1).detach().cpu().numpy() * 32767).astype(
                    np.int16
                )
                yield {"text": text, "audio_wav": audio_wav}

    def print_model_params(model: torch.nn.Module, extra_info=""):
        # print the number of parameters in the model
        model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(model)
        print(f"{extra_info} {model_million_params} M parameters")

    class Qwen3OmniMoeCode2WavNew(Qwen3OmniMoeCode2Wav):
        def chunked_decode(self, codes, chunk_size=300, left_context_size=25):
            wavs = []
            start_index = 0
            while start_index < codes.shape[-1]:
                end_index = min(start_index + chunk_size, codes.shape[-1])
                context_size = (
                    left_context_size if start_index - left_context_size > 0 else start_index
                )
                codes_chunk = codes[..., start_index - context_size : end_index]
                wav_chunk = self(codes_chunk)
                wavs.append(wav_chunk[..., context_size * self.total_upsample :])
                print(
                    f"{self.total_upsample=} {start_index=} {end_index=} {context_size=} {codes_chunk.shape=} {wav_chunk.shape=}"
                )
                start_index = end_index
            return torch.cat(wavs, dim=-1)

        def chunked_decode_stream(self, codes, chunk_size=300, left_context_size=25):
            start_index = 0
            while start_index < codes.shape[-1]:
                end_index = min(start_index + chunk_size, codes.shape[-1])
                context_size = (
                    left_context_size if start_index - left_context_size > 0 else start_index
                )
                codes_chunk = codes[..., start_index - context_size : end_index]
                wav_chunk = self(codes_chunk)
                yield wav_chunk[..., context_size * self.total_upsample :]
                start_index = end_index

    class Qwen3OmniMoeForConditionalGenerationNew(Qwen3OmniMoeForConditionalGeneration):
        def __init__(self, config: Qwen3OmniMoeConfig):
            super().__init__(config)
            # print(config)
            if hasattr(self, "code2wav"):
                self.code2wav = Qwen3OmniMoeCode2WavNew(config.code2wav_config)

        @torch.no_grad()
        def thinker_generate_chunk(
            self,
            inputs: dict,
            use_audio_in_video: bool = False,
            thinker_max_tokens_per_step: int = 10,  # Controls how many tokens to generate *per step*
            thinker_max_new_tokens: int = 1024,
            thinker_top_k: int = 20,
            thinker_top_p: float = 0.8,
            thinker_temperature: float = 0.9,
            thinker_eos_token_ids: list = [151643, 151645],  # Define EOS tokens
            thinker_repetition_penalty: float = 1.05,
            thinker_output_hidden_states: bool = False,
            thinker_stop_strings_per_step: list = [],
            tokenizer=None,
            **kwargs,
        ):
            """
            return yield {
                        "thinker_generate_ids": #[1,seq_len]
                        "thinker_generate_hidden_states": # tuple(tuple(tensor[1, 1, 2048])...)
                    }
            """
            input_ids = inputs.pop("input_ids")
            attention_mask = inputs.pop("attention_mask", None)

            if thinker_max_tokens_per_step > thinker_max_new_tokens:
                thinker_max_tokens_per_step = thinker_max_new_tokens

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
            while total_new_tokens_generated < thinker_max_new_tokens:
                # Prepare inputs for generate call
                # print(current_input_ids, current_attention_mask.shape)
                model_inputs = {
                    "input_ids": current_input_ids,
                    "attention_mask": current_attention_mask,
                    # "past_key_values": past_key_values,
                    "use_cache": True,
                    "use_audio_in_video": use_audio_in_video,
                    "do_sample": True if thinker_temperature > 0 else False,
                    "top_k": thinker_top_k,
                    "top_p": thinker_top_p,
                    "temperature": thinker_temperature,
                    "repetition_penalty": thinker_repetition_penalty,
                    "min_new_tokens": 1,  # Ensure at least one token is generated if possible
                    "max_new_tokens": thinker_max_tokens_per_step,  # Generate in smaller steps
                    # output_hidden_states/scores can consume memory,
                    # enable if needed downstream(talker)
                    "output_hidden_states": thinker_output_hidden_states,
                    "return_dict_in_generate": True,
                    # "output_scores": True,
                    "eos_token_id": thinker_eos_token_ids,
                    "pad_token_id": kwargs.get("thinker_pad_token_id", 151643),
                }
                model_inputs = {**inputs, **model_inputs}
                if len(thinker_stop_strings_per_step) > 0:
                    model_inputs["stop_strings"] = thinker_stop_strings_per_step
                    model_inputs["tokenizer"] = tokenizer

                start_time = perf_counter()
                # print(model_inputs)
                outputs = self.thinker.generate(**model_inputs)
                times.append(perf_counter() - start_time)

                # past_key_values = outputs.past_key_values
                # print(f"{past_key_values=}")

                # Extract newly generated token IDs *for this step*
                # `outputs.sequences` contains the input_ids for this step + new tokens generated in this step
                step_new_ids = outputs.sequences[:, current_input_ids.shape[1] :]
                num_step_new_tokens = step_new_ids.shape[1]
                total_new_tokens_generated += num_step_new_tokens

                if num_step_new_tokens == 0:  # Handle case where generate stops early
                    print("Warning: generate produced 0 new tokens in this step.")
                    break

                if thinker_output_hidden_states is True:
                    hidden_states = outputs.hidden_states
                    hidden_states_len = (
                        hidden_states_len if hidden_states_len > 0 else hidden_states[0][0].shape[1]
                    )
                    print(f"hidden_states_len: {hidden_states_len}")
                    # new generate thinker_token_embeds
                    thinker_new_token_embeds = hidden_states[0][0][:, :hidden_states_len, :]
                    hidden_states = (
                        (thinker_new_token_embeds,) + hidden_states[0][1:],
                    ) + hidden_states[1:]
                    # new generate thinker_hidden_states
                    thinker_new_hidden_states = hidden_states[0][-1][:, :hidden_states_len, :]
                    hidden_states = (
                        hidden_states[0][:-1] + (thinker_new_hidden_states,),
                    ) + hidden_states[1:]

                # Update the full sequence
                full_generated_ids = torch.cat([full_generated_ids, step_new_ids], dim=1)

                yield {
                    "thinker_generate_ids": step_new_ids,
                    "thinker_generate_hidden_states": hidden_states,
                    "full_generated_ids": full_generated_ids,
                }

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
                if step_new_ids[0, -1].item() in thinker_eos_token_ids:
                    print("EOS token generated.")
                    break

                # Check if max_new_tokens limit is reached (after processing the step)
                if total_new_tokens_generated >= thinker_max_new_tokens:
                    print("Max new tokens limit reached.")
                    break

            print(
                f"Total new tokens generated: {total_new_tokens_generated} | thinker_max_tokens_per_step: {thinker_max_tokens_per_step} | first chunk generated cost: {times[0]} s | total cost: {sum(times)} s"
            )

        @torch.no_grad()
        def talker_generate(
            self,
            hidden_states: tuple,
            sequences: torch.Tensor,
            input_ids: torch.Tensor,
            speaker_id: int,
            talker_kwargs: dict,
            token2wav_kwargs: dict,
        ):
            # 2. Prepare talker input
            thinker_embed = torch.cat([h[0] for h in hidden_states], dim=1).to(
                self.talker.device
            )  # [1 t d]
            thinker_hidden = torch.cat(
                [
                    h[self.config.talker_config.accept_hidden_layer] for h in hidden_states
                ],  # accept_hidden_layer:24
                dim=1,
            ).to(self.talker.device)  # [1 t d]
            im_start_indexes = torch.cat(
                (
                    torch.nonzero(input_ids[0] == self.config.im_start_token_id).squeeze(),
                    torch.tensor(
                        [sequences.shape[-1]],
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    ),
                ),
                dim=-1,
            ).to(
                self.talker.device
            )  # Shape [n_starts + 1]; Take batch 0 since batched inference is not supported here.
            multimodal_mask = (
                (sequences == self.config.thinker_config.audio_token_id) |
                (sequences == self.config.thinker_config.image_token_id) |
                (sequences == self.config.thinker_config.video_token_id)
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
                    print(
                        im_start_indexes,
                        "get_talker_user_parts",
                        im_start_index,
                        segment_end_index,
                        f"{multimodal_mask.shape=}",
                        f"{thinker_hidden.shape=}",
                        f"{thinker_embed.shape=}",
                    )
                    talker_user_part = self._get_talker_user_parts(
                        im_start_index,
                        segment_end_index,
                        multimodal_mask,
                        thinker_hidden,
                        thinker_embed,
                    )
                    talker_input_embeds.append(talker_user_part)
                    talker_input_ids.append(sequences[:, im_start_index:segment_end_index])
                # Take assistant output (for now)
                elif (
                    role_token == self.config.assistant_token_id and i == len(im_start_indexes) - 2
                ):
                    print(
                        im_start_indexes,
                        "get_talker_assistant_parts",
                        im_start_index,
                        segment_end_index,
                        speaker_id,
                        f"{thinker_embed.shape=}",
                        f"{tts_pad_embed.shape=}",
                        f"{tts_bos_embed.shape=}",
                        f"{tts_eos_embed.shape=}",
                    )
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

            print(f"{talker_input_embed=}", talker_input_embed.shape)
            print(f"{trailing_text_hidden=}", trailing_text_hidden.shape)
            print(f"{tts_pad_embed=}", tts_pad_embed.shape)
            print(f"{talker_input_id=}", talker_input_id.shape)
            talker_result = self.talker.generate(
                attention_mask=torch.ones(talker_input_id.shape, device=talker_input_id.device),
                pad_token_id=2150,
                inputs_embeds=talker_input_embed,
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=tts_pad_embed,
                talker_input_ids=talker_input_id,  # Not use input_ids to prevent repetation penalty out of bound
                **talker_kwargs,
            )
            # print(len(talker_result.hidden_states))
            # for hid in talker_result.hidden_states:
            #    print(len(hid), hid[-1])

            talker_codes = (
                torch.stack(
                    [hid[-1] for hid in talker_result.hidden_states if hid[-1] is not None], dim=1
                )
                .transpose(1, 2)
                .to(self.code2wav.device)
            )
            print("talker_codes", talker_codes.tolist(), talker_codes.shape)  # [1,1+15,num_hidden]
            torch.save(talker_codes, f"{ASSETS_DIR}/talker_codes.pt")

            chunk_size = token2wav_kwargs.get("chunk_size", 300)
            left_context_size = token2wav_kwargs.get("left_context_size", 25)
            talker_wavs = self.code2wav.chunked_decode(
                talker_codes, chunk_size=chunk_size, left_context_size=left_context_size
            )
            print("talker_wavs", talker_wavs, talker_wavs.shape)
            return talker_wavs.float()  # BFloat16 -> Float

        def _get_talker_assistant_parts(
            self,
            im_start_index,
            segment_end_index,
            speaker_id,
            thinker_embed,
            tts_pad_embed,
            tts_bos_embed,
            tts_eos_embed,
        ):
            assistant_hidden = self.talker.text_projection(
                thinker_embed[:, im_start_index:segment_end_index]
            ).to(self.talker.device)  # [1 t d]
            print(f"{assistant_hidden.shape=}")
            assistant_text_hidden = torch.cat(
                (
                    assistant_hidden[:, :3],
                    tts_pad_embed.expand(-1, 4, -1),
                    tts_bos_embed,
                    assistant_hidden[:, 3:4],  # First text
                ),
                dim=1,
            )
            codec_special_tokens = torch.tensor(
                [
                    [
                        self.config.talker_config.codec_nothink_id,
                        self.config.talker_config.codec_think_bos_id,
                        self.config.talker_config.codec_think_eos_id,
                        speaker_id,
                        self.config.talker_config.codec_pad_id,
                        self.config.talker_config.codec_bos_id,
                    ]
                ],
                device=self.talker.device,
                dtype=torch.long,
            )
            assistant_codec_hidden = torch.cat(
                (
                    torch.zeros(
                        (1, 3, self.config.talker_config.text_config.hidden_size),
                        device=self.talker.device,
                        dtype=self.talker.dtype,
                    ),
                    self.talker.get_input_embeddings()(codec_special_tokens).to(self.talker.device),
                ),
                dim=1,
            )
            trailing_text_hidden = torch.cat(
                (
                    assistant_hidden[:, 4:],
                    tts_eos_embed,
                ),
                dim=1,
            )

            print(assistant_text_hidden.shape, assistant_codec_hidden.shape)
            input_embeds = assistant_text_hidden + assistant_codec_hidden
            input_ids = torch.full(
                (1, assistant_text_hidden.shape[1]),
                fill_value=self.config.tts_pad_token_id,
                dtype=torch.long,
                device=assistant_text_hidden.device,
            )
            return input_embeds, input_ids, trailing_text_hidden

        @torch.no_grad()
        def generate_stream(
            self,
            inputs: dict,
            use_audio_in_video: bool = False,
            thinker_max_tokens_per_step=10,  # Controls how many tokens to generate *per step*
            thinker_max_new_tokens: int = 1024,
            thinker_top_k: int = 20,
            thinker_top_p: float = 0.8,
            thinker_temperature: float = 0.9,
            thinker_repetition_penalty: float = 1.05,
            thinker_eos_token_ids: list[int] = [151643, 151645],
            thinker_stop_strings_per_step: list[str] = [],
            tokenizer=None,
            return_audio: bool = True,
            speaker: str = "Chelsie",
            talker_top_k: int = 10,
            talker_top_p: float = 0.9,
            talker_temperature: float = 0.95,
            talker_repetition_penalty: float = 1.1,
            talker_max_new_tokens: int = 8192,
            **kwargs,
        ) -> Generator[dict, None, None]:
            """
            - return Generator[dict, None, None]
            {
                "thinker_ids": torch.Tensor, # (1,T)
                "talker_wav": torch.Tensor, # (1,1,T)
            }
            """
            input_ids = inputs.get("input_ids")
            thinker_chunk_stream = self.thinker_generate_chunk(
                inputs,
                use_audio_in_video=use_audio_in_video,
                thinker_max_tokens_per_step=thinker_max_tokens_per_step,
                thinker_max_new_tokens=thinker_max_new_tokens,
                thinker_top_k=thinker_top_k,
                thinker_top_p=thinker_top_p,
                thinker_temperature=thinker_temperature,
                thinker_eos_token_ids=thinker_eos_token_ids,
                thinker_repetition_penalty=thinker_repetition_penalty,
                thinker_output_hidden_states=return_audio,
                thinker_stop_strings_per_step=thinker_stop_strings_per_step,
                tokenizer=tokenizer,
                **kwargs,
            )
            if not return_audio:
                for thinker_chunk in thinker_chunk_stream:
                    yield {"thinker_ids": thinker_chunk["thinker_generate_ids"]}
            else:
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
                    "do_sample": talker_temperature > 0.0,
                    "top_k": talker_top_k,
                    "top_p": talker_top_p,
                    "temperature": talker_temperature,
                    "eos_token_id": self.config.talker_config.codec_eos_token_id,
                    "repetition_penalty": talker_repetition_penalty,
                    "suppress_tokens": talker_supppressed_tokens,
                    "output_hidden_states": True,
                    "return_dict_in_generate": True,
                }
                token2wav_kwargs = {"chunk_size": 300, "left_context_size": 25}
                for thinker_chunk in thinker_chunk_stream:
                    thinker_generate_hidden_states = thinker_chunk["thinker_generate_hidden_states"]
                    full_generated_ids = thinker_chunk["full_generated_ids"]
                    print(f"talker input sequences: {full_generated_ids}")
                    print(f"hidden_states tuples size {len(thinker_generate_hidden_states)}")
                    for i, item in enumerate(thinker_generate_hidden_states):
                        if isinstance(item, tuple):
                            print(
                                f"hidden_states tuple len: {len(item)}, item shape:{item[0].shape}",
                            )
                        elif isinstance(item, torch.Tensor):
                            print(f"{i=}", item.shape)
                        else:
                            print(f"{i=}", item)
                    talker_wav = self.talker_generate(
                        hidden_states=thinker_generate_hidden_states,
                        sequences=full_generated_ids,
                        input_ids=input_ids,
                        speaker_id=speaker_id,
                        talker_kwargs=talker_kwargs,
                        token2wav_kwargs=token2wav_kwargs,
                    )  # [1,1,T]
                    yield {
                        "thinker_ids": thinker_chunk["thinker_generate_ids"],
                        "talker_wav": talker_wav,
                    }

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
                # "chelsie": 2301, "ethan": 2302, "aiden": 2303
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

            print(f"talker input sequences: {thinker_result.sequences}")
            print(f"hidden_states tuples size {len(thinker_result.hidden_states)}")
            for i, item in enumerate(thinker_result.hidden_states):
                if isinstance(item, tuple):
                    print(
                        f"hidden_states tuple len: {len(item)}, item shape:{item[0].shape}",
                    )
                elif isinstance(item, torch.Tensor):
                    print(f"{i=}", item.shape)
                else:
                    print(f"{i=}", item)

            talker_wavs = self.talker_generate(
                hidden_states=thinker_result.hidden_states,
                sequences=thinker_result.sequences,
                input_ids=input_ids,
                speaker_id=speaker_id,
                talker_kwargs=talker_kwargs,
                token2wav_kwargs=token2wav_kwargs,
            )  # [1,1,T]
            return thinker_result, talker_wavs

    model: Qwen3OmniMoeForConditionalGenerationNew = None
    processor: Qwen3OmniMoeProcessor = None

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
    # processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)

    # print(tokenizer)
    print("---" * 20)
    # print(processor)

    text = tokenizer.decode(
        # [ 151644, 872, 198, 105043, 100165, 11319, 151645, 198, 151644, 77091, 198, 104198, 107076, 100338, 111477, 31935, 64559, 104800, 111411, 9370, 42140, 53772, 35243, 71304, 105483, 102064, 104949, 3837, 35946, 99882, 31935, 64559, 99320, 56007, 63488, 7751, 1773, 104139, 109944, 100364, 103929, 101037, 11319, 151645]
        # [104198, 107076, 100338, 111477,  31935,  64559, 104800, 111411,   9370, 42140,  53772,  35243,  71304, 105483, 102064, 104949,   3837,  35946, 99882,  31935,  64559,  99320,  56007,  63488,   7751,   1773, 104139, 109944, 100364, 103929, 101037,  11319, 151645]
    )
    print(text)


def code2wav(**kwargs):
    gpu_prop = torch.cuda.get_device_properties("cuda")
    model = Qwen3OmniMoeForConditionalGenerationNew.from_pretrained(
        MODEL_PATH,
        dtype="auto",
        attn_implementation="flash_attention_2" if gpu_prop.major > 8 else None,
        device_map="auto",
    )
    model = model.eval()

    talker_codes = torch.load(f"{ASSETS_DIR}/talker_codes.pt")
    print(talker_codes, talker_codes.shape)  # [1,1+15,num_hidden]
    chunk_size = kwargs.get("chunk_size", 300)
    left_context_size = kwargs.get("left_context_size", 25)
    # left_context_size = kwargs.get("left_context_size", 1)
    size = talker_codes.shape[-1]
    start = 0
    audio_bytes = b""
    while start < size:
        end = start + 10
        # end = start + 1
        talker_codes_chunk = talker_codes[..., start:end]
        start_time = perf_counter()
        talker_wavs = model.code2wav.chunked_decode(
            talker_codes_chunk, chunk_size=chunk_size, left_context_size=left_context_size
        )
        print("talker_wavs", talker_wavs, talker_wavs.shape)
        print(f"chunk {start}:{end} cost {perf_counter() - start_time} s")
        audio = talker_wavs.float()  # BFloat16 -> Float
        audio_chunk_bytes = (
            np.array(audio.reshape(-1).detach().cpu().numpy() * 32767).astype(np.int16).tobytes()
        )
        audio_bytes += audio_chunk_bytes
        # with wave.open(f"{ASSETS_DIR}/code2wav_{start}.wav", "wb") as f:
        #    f.setnchannels(1)
        #    f.setsampwidth(2)
        #    f.setframerate(24000)
        #    f.writeframes(audio_chunk_bytes)

        start = end
    with wave.open(f"{ASSETS_DIR}/code2wav.wav", "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(24000)
        f.writeframes(audio_bytes)


def code2wav_stream(**kwargs):
    gpu_prop = torch.cuda.get_device_properties("cuda")
    model = Qwen3OmniMoeForConditionalGenerationNew.from_pretrained(
        MODEL_PATH,
        dtype="auto",
        attn_implementation="flash_attention_2" if gpu_prop.major > 8 else None,
        device_map="auto",
    )
    model = model.eval()

    talker_codes = torch.load(f"{ASSETS_DIR}/talker_codes.pt")
    print(talker_codes, talker_codes.shape)  # [1,1+15,num_hidden]
    chunk_size = kwargs.get("chunk_size", 50)
    left_context_size = kwargs.get("left_context_size", 25)
    # left_context_size = kwargs.get("left_context_size", 1)
    audio_bytes = b""
    talker_wavs_iter = model.code2wav.chunked_decode_stream(
        talker_codes, chunk_size=chunk_size, left_context_size=left_context_size
    )
    start_time = perf_counter()
    for talker_wavs in talker_wavs_iter:
        print(f"{chunk_size=}  cost {perf_counter() - start_time} s")
        print("talker_wavs", talker_wavs, talker_wavs.shape)
        audio = talker_wavs.float()  # BFloat16 -> Float
        audio_chunk_bytes = (
            np.array(audio.reshape(-1).detach().cpu().numpy() * 32767).astype(np.int16).tobytes()
        )
        audio_bytes += audio_chunk_bytes
        # with wave.open(f"{ASSETS_DIR}/code2wav_{start}.wav", "wb") as f:
        #    f.setnchannels(1)
        #    f.setsampwidth(2)
        #    f.setframerate(24000)
        #    f.writeframes(audio_chunk_bytes)

    with wave.open(f"{ASSETS_DIR}/code2wav_stream.wav", "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(24000)
        f.writeframes(audio_bytes)


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


def text2text(**kwargs):
    messages = [
        # SYSTEM_MESSAGE,
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "你是谁？"},
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


def text2text_stream(**kwargs):
    messages = [
        # SYSTEM_MESSAGE,
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "你是谁？"},
            ],
        },
    ]

    gen_iter = run_model_stream(
        messages=messages,
        return_audio=False,
        use_audio_in_video=False,
        # thinker_max_tokens_per_step=1, issue
        thinker_max_tokens_per_step=10,
    )

    text = ""
    for item in gen_iter:
        print(item["text"])
        text += item["text"]
    print(text)


def text2speech_stream(**kwargs):
    """text2speech chunk stream"""
    messages = [
        # SYSTEM_MESSAGE,
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "你是谁？"},
            ],
        },
    ]

    gen_iter = run_model_stream(
        messages=messages,
        return_audio=True,
        use_audio_in_video=False,
        thinker_max_tokens_per_step=10,
    )
    text = ""
    audio = b""
    for i, item in enumerate(gen_iter):
        print(item.get("text"))
        text += item.get("text")
        if item.get("audio_wav") is not None:
            audio_bytes = item.get("audio_wav").tobytes()
            audio += audio_bytes
            with wave.open(f"{ASSETS_DIR}/qwen3omni_text2speech_stream_{i}.wav", "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(24000)
                f.writeframes(audio_bytes)
    print(text)
    with wave.open(f"{ASSETS_DIR}/qwen3omni_text2speech_stream.wav", "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(24000)
        f.writeframes(audio)


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


def audio_interaction_scene_stream(**kwargs):
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
    gen_iter = run_model_stream(
        messages=messages,
        return_audio=True,
        use_audio_in_video=False,
        thinker_max_tokens_per_step=10,
    )
    text = ""
    audio = b""
    for i, item in enumerate(gen_iter):
        print(item.get("text"))
        text += item.get("text")
        if item.get("audio_wav") is not None:
            audio_bytes = item.get("audio_wav").tobytes()
            audio += audio_bytes
            with wave.open(
                f"{ASSETS_DIR}/qwen3omni_audio_interaction_scene_stream_{i}.wav", "wb"
            ) as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(24000)
                f.writeframes(audio_bytes)

    print(text)
    with wave.open(f"{ASSETS_DIR}/qwen3omni_audio_interaction_scene_stream.wav", "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(24000)
        f.writeframes(audio)


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


def video_interaction_scene_stream(**kwargs):
    video_path = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/interaction4.mp4"
    )

    messages = [
        {"role": "system", "content": "你是一个北京大爷，说话很幽默，说这地道北京话。"},
        {"role": "user", "content": [{"type": "video", "video": video_path}]},
    ]
    gen_iter = run_model_stream(
        messages=messages,
        return_audio=True,
        use_audio_in_video=False,
        thinker_max_tokens_per_step=10,
    )
    text = ""
    audio = b""
    for i, item in enumerate(gen_iter):
        print(item.get("text"))
        text += item.get("text")
        if item.get("audio_wav") is not None:
            audio_bytes = item.get("audio_wav").tobytes()
            audio += audio_bytes
            with wave.open(
                f"{ASSETS_DIR}/qwen3omni_video_interaction_scene_stream_{i}.wav", "wb"
            ) as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(24000)
                f.writeframes(audio_bytes)
    print(text)
    with wave.open(f"{ASSETS_DIR}/qwen3omni_video_interaction_scene_stream.wav", "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(24000)
        f.writeframes(audio)


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


def image_text_interaction(**kwargs):
    messages = [
        SYSTEM_MESSAGE,
        {
            "role": "user",
            "content": [
                {"type": "image", "image": os.path.join(ASSETS_DIR, "03-Confusing-Pictures.jpg")},
                {"type": "text", "text": "请描述一下图片中的内容"},
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
        with wave.open(f"{ASSETS_DIR}/qwen3omni_image_text_interaction.wav", "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(24000)
            f.writeframes(audio_bytes)


def image_text_interaction_stream(**kwargs):
    messages = [
        SYSTEM_MESSAGE,
        # {"role": "system", "content": "you are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": os.path.join(ASSETS_DIR, "03-Confusing-Pictures.jpg")},
                {"type": "text", "text": "请描述一下图片中的内容"},
            ],
        },
    ]

    gen_iter = run_model_stream(
        messages=messages,
        return_audio=True,
        use_audio_in_video=False,
        thinker_max_tokens_per_step=15,
    )
    text = ""
    audio = b""
    for i, item in enumerate(gen_iter):
        print(item.get("text"))
        text += item.get("text")
        if item.get("audio_wav") is not None:
            audio_bytes = item.get("audio_wav").tobytes()
            audio += audio_bytes
            with wave.open(
                f"{ASSETS_DIR}/qwen3omni_image_text_interaction_stream_{i}.wav", "wb"
            ) as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(24000)
                f.writeframes(audio_bytes)

    print(text)
    with wave.open(f"{ASSETS_DIR}/qwen3omni_image_text_interaction_stream.wav", "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(24000)
        f.writeframes(audio)


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


def image_audio_interaction_stream(**kwargs):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": os.path.join(ASSETS_DIR, "03-Confusing-Pictures.jpg")},
                {"type": "audio", "audio": os.path.join(ASSETS_DIR, "1272-128104-0000.flac")},
            ],
        }
    ]

    gen_iter = run_model_stream(
        messages=messages,
        return_audio=True,
        use_audio_in_video=False,
        thinker_max_tokens_per_step=10,
    )
    text = ""
    audio = b""
    for i, item in enumerate(gen_iter):
        print(item.get("text"))
        text += item.get("text")
        if item.get("audio_wav") is not None:
            audio_bytes = item.get("audio_wav").tobytes()
            audio += audio_bytes
            with wave.open(
                f"{ASSETS_DIR}/qwen3omni_image_audio_interaction_stream_{i}.wav", "wb"
            ) as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(24000)
                f.writeframes(audio_bytes)

    print(text)
    with wave.open(f"{ASSETS_DIR}/qwen3omni_image_audio_interaction_stream.wav", "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(24000)
        f.writeframes(audio)


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


def achatbot_generate():
    import torchaudio
    import soundfile as sf
    from achatbot.core.llm.transformers.manual_vision_voice_qwen3 import (
        TransformersManualQwen3OmniLLM,
    )
    from achatbot.common.session import Session, SessionCtx
    from achatbot.core.llm import LLMEnvInit
    from achatbot.common.logger import Logger

    Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)

    session = Session(**SessionCtx("test_client_id", 16000, 2).__dict__)
    args = LLMEnvInit.get_qwen3omni_transformers_args()
    args["lm_model_name_or_path"] = "/root/.achatbot/models/Qwen/Qwen3-Omni-30B-A3B-Instruct"
    args["init_chat_prompt"] = SYSTEM_PROMPT
    args["verbose"] = True
    args["speaker"] = "Ethan"
    args["lm_attn_impl"] = "flash_attention_2"
    args["warmup_steps"] = 0
    args["warmup_prompt"] = "你叫什么名字？"
    args["thinker_args"]["lm_gen_temperature"] = 0.0
    args["code2wav_args"]["chunk_size"] = 50
    args["code2wav_args"]["left_context_size"] = 25
    llm = TransformersManualQwen3OmniLLM(**args)

    print("----start generate stream----")

    session.ctx.state["prompt"] = [
        {"type": "image", "image": os.path.join(ASSETS_DIR, "03-Confusing-Pictures.jpg")},
        # {"type": "audio", "audio": ""},
        # {"type": "video", "video": ""},
        {"type": "text", "text": "请描述一下图片中的内容"},
    ]
    kwargs = {
        "use_audio_in_video": False,
        "thinker_top_k": 10,
        "thinker_top_p": 0.9,
        "thinker_temperature": 0.95,
        "thinker_repetition_penalty": 1.1,
        "thinker_min_new_tokens": 1,
        "thinker_max_tokens_per_step": 15,
        "thinker_stop_strings_per_step": [",", ".", "，", "。"],
        "thinker_max_new_tokens": 150,
        "thinker_eos_token_ids": [
            151643,
            151645,
        ],
        "thinker_pad_token_id": 151643,
    }
    chunk_stream = llm.generate(session, **kwargs)
    gen_text = ""
    gen_all_text = ""
    audios = []
    times = []
    start_time = time.perf_counter()
    for i, chunk in enumerate(chunk_stream):
        times.append(time.perf_counter() - start_time)
        print(chunk)
        text = chunk["text"] if "text" in chunk else ""
        if gen_text != text:
            gen_text = text
            gen_all_text += gen_text
        if "audio_wav" in chunk:
            wav = chunk["audio_wav"]
            print(text, wav.shape)
            audios.append(wav.squeeze().cpu().numpy())
            # save_audio_path = os.path.join(ASSETS_DIR, f"achatbot_generate_stream-{i}-{text}.wav")
            # torchaudio.save(save_audio_path, wav, sample_rate=24000)
            # print(f"Audio saved to {save_audio_path}")
        else:
            print(text)
        start_time = time.perf_counter()

    print(f"gen all text: {gen_all_text}")
    if len(audios) > 0:
        save_audio_path = os.path.join(ASSETS_DIR, f"achatbot_generate_stream.wav")
        sf.write(save_audio_path, np.concatenate(audios), samplerate=24000)
        print(f"All Audio saved to {save_audio_path}")
        info = sf.info(save_audio_path, verbose=True)
        print(
            f"thinker->talker->code2wav chunk streaming first chunk time: {times[0]} s | wav duration: {info.duration} s | cost: {sum(times)} s | RTF: {sum(times) / info.duration}"
        )


"""
modal run src/llm/transformers/qwen3_omni.py --task tokenizer
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen3_omni.py --task dump_model
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen3_omni.py --task asr
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen3_omni.py --task text2speech
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen3_omni.py --task speech_translation
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen3_omni.py --task image_question
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen3_omni.py --task audio_interaction
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen3_omni.py --task audio_interaction_scene (speech chat)
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen3_omni.py --task video_interaction
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen3_omni.py --task video_interaction_scene (video include audio chat)
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen3_omni.py --task video_information_extracting
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen3_omni.py --task image_text_interaction
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen3_omni.py --task image_audio_interaction
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen3_omni.py --task audio_function_call
IMAGE_GPU=B200 modal run src/llm/transformers/qwen3_omni.py --task text_image_video_audio_interaction
IMAGE_GPU=B200 modal run src/llm/transformers/qwen3_omni.py --task batch_requests

# think chunk stream
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen3_omni.py --task text2text_stream
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen3_omni.py --task text2speech_stream
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen3_omni.py --task image_text_interaction_stream
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen3_omni.py --task image_audio_interaction_stream
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen3_omni.py --task audio_interaction_scene_stream (speech chat)
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen3_omni.py --task video_interaction_scene_stream (video include audio chat)

IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen3_omni.py --task code2wav
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen3_omni.py --task code2wav_stream

ACHATBOT_VERSION=0.0.26.post1 IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen3_omni.py --task achatbot_generate

> [!TIP]:
> - 生成音频中未对特殊字符进行处理（omni统一到一起直接生成音频的弊端, 也许可以在隐藏层解决, 系统提示词限制貌似不起作用(提示不含特殊字符）, 比如：
    *   **优点**：这款饮料是专门为运动后设计的。它的核心成分是电解质，标签上明确写着“电解质≥200mg”，这能有效补充运动时因大量出汗流失的钠、钾等矿物质，帮助维持体液平衡，防止抽筋。同时，它也含有维生素E和维生素B6，有助于能量代谢。它还是0糖0卡的，不用担心额外的热量。

"""


@app.local_entrypoint()
def main(task: str = "tokenizer"):
    tasks = {
        "tokenizer": tokenizer,
        "dump_model": dump_model,
        "code2wav": code2wav,
        "code2wav_stream": code2wav_stream,
        "asr": asr,
        "text2text": text2text,
        "text2speech": text2speech,
        "speech_translation": speech_translation,
        "image_question": image_question,
        "audio_interaction": audio_interaction,
        "audio_interaction_scene": audio_interaction_scene,
        "video_interaction": video_interaction,
        "video_interaction_scene": video_interaction_scene,
        "video_information_extracting": video_information_extracting,
        "image_text_interaction": image_text_interaction,
        "image_audio_interaction": image_audio_interaction,
        "audio_function_call": audio_function_call,
        "text_image_video_audio_interaction": text_image_video_audio_interaction,
        "batch_requests": batch_requests,
        "text2text_stream": text2text_stream,
        "text2speech_stream": text2speech_stream,
        "image_text_interaction_stream": image_text_interaction_stream,
        "image_audio_interaction_stream": image_audio_interaction_stream,
        "audio_interaction_scene_stream": audio_interaction_scene_stream,
        "video_interaction_scene_stream": video_interaction_scene_stream,
        "achatbot_generate": achatbot_generate,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(tasks[task])
