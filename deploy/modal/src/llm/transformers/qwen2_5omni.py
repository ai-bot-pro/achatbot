from time import perf_counter
from typing import Optional
import modal
import os
from transformers.generation.streamers import BaseStreamer

app = modal.App("qwen2_5_omni")
omni_img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "cmake")
    .pip_install("wheel", "openai", "qwen-omni-utils[decord]")
    .run_commands(
        f"pip install git+https://github.com/huggingface/transformers",
    )
    .pip_install(
        "accelerate",
        "torch",
        "torchvision",
        "torchaudio",
        "soundfile==0.13.0",
        "librosa==0.11.0",
    )
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)

HF_MODEL_DIR = "/root/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)

# NOTE: if want to generate speech, need use this system prompt to generate speech
SPEECH_SYS_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
# Voice settings
SPEAKER_LIST = ["Chelsie", "Ethan"]
DEFAULT_SPEAKER = "Ethan"

with omni_img.imports():
    import subprocess
    from threading import Thread
    from queue import Queue

    import torch, torchaudio
    from transformers import (
        Qwen2_5OmniForConditionalGeneration,
        Qwen2_5OmniProcessor,
        TextIteratorStreamer,
        AutoConfig,
    )
    from qwen_omni_utils import process_mm_info

    def print_model_params(model: torch.nn.Module, extra_info=""):
        # print the number of parameters in the model
        model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
        # print(model)
        print(f"{extra_info} {model_million_params} M parameters")

    class Qwen2_5OmniForConditionalGenerationNew(Qwen2_5OmniForConditionalGeneration):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            print_model_params(self.thinker, "qwen2.5omni_thinker")
            print_model_params(self.talker, "qwen2.5omni_talker")
            print_model_params(self.token2wav, "qwen2.5omni_token2wav")

        @torch.no_grad()
        # TODO: raushan, defaults should be saved in generation config
        def generate(
            self,
            input_ids: Optional[torch.Tensor] = None,
            speaker: str = "Chelsie",
            use_audio_in_video: bool = False,
            return_audio: Optional[bool] = None,
            thinker_max_new_tokens: int = 1024,
            talker_max_new_tokens: int = 4096,
            talker_do_sample: bool = True,
            talker_top_k: int = 40,
            talker_top_p: float = 0.8,
            talker_temperature: float = 0.9,
            talker_eos_token_id: list[int] = [8292, 8294],
            talker_repetition_penalty: float = 1.05,
            **kwargs,
        ):
            r"""
            Generate text response and audio from input.

            Args:
                input_ids (`Optional[torch.Tensor]`, *optional*):
                    Input ids, should obtain from processor.
                speaker (`str` , defaults to "Chelsie"):
                    Which speaker should be used in audio response.
                use_audio_in_video (`bool`, defaults to False):
                    Whether or not use audio track in video, should same as the parameter in `process_audio_info`.
                return_audio (`Optional[bool]`, *optional*):
                    Whether or not return response in audio format. When `return_audio=None`, this parameter is same as `config.enable_audio_output`.
                kwargs (*optional*):
                    - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model.
                    - With a *thinker_*, *talker_*, *token2wav_* prefix, they will be input for the `generate` method of the
                    thinker, talker and token2wav respectively. It has the priority over the keywords without a prefix.
            Returns:
                When `return_audio=False`:
                    - **Text** (`torch.Tensor`): Generated text token sequence.
                When `return_audio=True`:
                    - **Text** (`torch.Tensor`): Generated text token sequence.
                    - **Audio waveform** (`torch.Tensor`): Generated audio waveform.
            """
            if speaker not in self.speaker_map:
                raise ValueError(
                    f"{speaker} is not availible, availible speakers: {self.speaker_map.keys()}"
                )
            if return_audio and not self.has_talker:
                raise ValueError(
                    "Cannot use talker when talker module not initalized. Use `enable_talker` method or set enable_talker in config to enable talker."
                )
            if return_audio is None:
                return_audio = self.has_talker
            if input_ids.shape[0] != 1 and return_audio:
                raise NotImplementedError(
                    "Qwen2.5-Omni currently does not support batched inference with audio output"
                )

            shared_kwargs = {"use_audio_in_video": use_audio_in_video}
            thinker_kwargs = {
                "max_new_tokens": thinker_max_new_tokens,
            }
            talker_kwargs = {
                "max_new_tokens": talker_max_new_tokens,
                "do_sample": talker_do_sample,
                "top_k": talker_top_k,
                "top_p": talker_top_p,
                "temperature": talker_temperature,
                "eos_token_id": talker_eos_token_id,
                "repetition_penalty": talker_repetition_penalty,
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
                elif key == "input_features" or key == "attention_mask":
                    thinker_kwargs[key] = value
                # Put other key to shared kwargs
                else:
                    shared_kwargs[key] = value

            # Merge kwargs
            for key, value in shared_kwargs.items():
                if key not in thinker_kwargs:
                    thinker_kwargs[key] = value
                if key not in talker_kwargs:
                    talker_kwargs[key] = value
                if key not in token2wav_kwargs:
                    token2wav_kwargs[key] = value
            speaker_params = self.speaker_map[speaker]

            # 1. Generate from thinker module
            generate_audio = return_audio and self.has_talker
            if generate_audio:
                thinker_kwargs["output_hidden_states"] = True
                thinker_kwargs["return_dict_in_generate"] = True

            thinker_result = self.thinker.generate(input_ids=input_ids, **thinker_kwargs)

            if not generate_audio:
                return thinker_result

            # 2. Generate speech tokens from talker module
            thinker_generate_ids = thinker_result.sequences[:, input_ids.size(1) :].to(
                self.talker.device
            )
            thinker_token_embeds = [
                token_hidden_states[0].to(self.talker.device)
                for token_hidden_states in thinker_result.hidden_states
            ]
            thinker_hidden_states = [
                token_hidden_states[-1].to(self.talker.device)
                for token_hidden_states in thinker_result.hidden_states
            ]

            talker_text_bos_token = speaker_params["bos_token"]
            talker_input_text_ids = torch.cat(
                [
                    input_ids.to(self.talker.device),
                    torch.tensor(
                        [[talker_text_bos_token]], dtype=torch.long, device=self.talker.device
                    ),
                    thinker_generate_ids[:, :1],
                ],
                dim=-1,
            )

            talker_input_ids = torch.cat(
                [
                    torch.full_like(
                        input_ids,
                        fill_value=self.talker.codec_mask_token,
                        device=self.talker.device,
                    ),
                    torch.tensor(
                        [[self.talker.codec_pad_token]], dtype=torch.long, device=self.talker.device
                    ),
                    torch.tensor(
                        [[self.talker.codec_bos_token]], dtype=torch.long, device=self.talker.device
                    ),
                ],
                dim=1,
            )

            thinker_embed_tokens = self.thinker.get_input_embeddings()
            thinker_reply_part = torch.cat(thinker_hidden_states[1:], dim=1) + torch.cat(
                thinker_token_embeds[1:], dim=1
            )
            talker_inputs_embeds = thinker_hidden_states[0] + thinker_token_embeds[0]
            talker_text_bos_token = torch.tensor(
                [[talker_text_bos_token]], dtype=torch.long, device=self.thinker.device
            )
            talker_text_bos_embed = thinker_embed_tokens(talker_text_bos_token).to(
                self.talker.device
            )
            talker_inputs_embeds = torch.cat(
                [
                    talker_inputs_embeds,
                    talker_text_bos_embed,
                    thinker_reply_part[:, :1, :],
                ],
                dim=1,
            )

            eos_embedding = thinker_embed_tokens(
                torch.tensor(
                    [[self.talker.text_eos_token]], dtype=torch.long, device=self.thinker.device
                )
            ).to(self.talker.device)

            pad_embedding = thinker_embed_tokens(
                torch.tensor(
                    [[self.talker.text_pad_token]], dtype=torch.long, device=self.thinker.device
                )
            ).to(self.talker.device)

            thinker_reply_part = torch.cat(
                [
                    thinker_reply_part[:, 1:, :],
                    eos_embedding,
                    pad_embedding,
                ],
                dim=1,
            )

            talker_attention_mask = None
            if "attention_mask" in kwargs:
                talker_attention_mask = torch.cat(
                    [kwargs["attention_mask"], kwargs["attention_mask"].new_ones((1, 2))], dim=1
                ).to(self.talker.device)

            # stream
            skip_prompt = kwargs.get("skip_prompt", True)
            streamer = TokenStreamer(skip_prompt=skip_prompt)
            talker_kwargs = dict(
                input_ids=talker_input_ids,
                streamer=streamer,
                input_text_ids=talker_input_text_ids,
                thinker_reply_part=thinker_reply_part,
                inputs_embeds=talker_inputs_embeds,
                attention_mask=talker_attention_mask,
                suppress_tokens=[self.talker.codec_bos_token],
                **{
                    k: (v.to(self.talker.device) if torch.is_tensor(v) else v)
                    for k, v in talker_kwargs.items()
                },
            )
            # print(talker_kwargs.keys())
            thread = Thread(target=self.talker.generate, kwargs=talker_kwargs)
            thread.start()
            talker_generate_codes = []
            times = []
            start_time = perf_counter()
            for token_id in streamer:
                # print(token_id)
                times.append(perf_counter() - start_time)
                start_time = perf_counter()
                talker_generate_codes.append(token_id)
            print(
                f"generate first token cost time: {times[0]} s, {len(times)} tokens cost time: {sum(times)} s"
            )
            offset = 0
            if skip_prompt is False:
                offset = talker_input_ids.shape[1]
            # print(
            #    talker_input_ids.shape[1],
            #    # talker_generate_codes,
            #    talker_generate_codes[:offset],
            #    talker_generate_codes[offset:-1],
            # )
            talker_generate_codes = torch.tensor(
                [talker_generate_codes[offset:-1]],
                dtype=torch.long,
                device=self.talker.device,
            )

            # no stream
            # talker_result = self.talker.generate(
            #     input_ids=talker_input_ids,
            #     input_text_ids=talker_input_text_ids,
            #     thinker_reply_part=thinker_reply_part,
            #     inputs_embeds=talker_inputs_embeds,
            #     attention_mask=talker_attention_mask,
            #     suppress_tokens=[self.talker.codec_bos_token],
            #     **{
            #         k: (v.to(self.talker.device) if torch.is_tensor(v) else v)
            #         for k, v in talker_kwargs.items()
            #     },
            # )
            # print(talker_result.shape, talker_result)
            # talker_generate_codes = talker_result[:, talker_input_ids.shape[1] : -1]

            # print(f"talker_generate_codes:{talker_generate_codes.shape} {talker_generate_codes}")

            # 3. Generate wavs from code
            if self.token2wav.dtype != torch.float:
                self.token2wav.float()

            # print(self.token2wav.device, speaker_params, token2wav_kwargs)

            wav = self.token2wav(
                talker_generate_codes.to(self.token2wav.device),
                conditioning=speaker_params["cond"].to(self.token2wav.device).float(),
                reference_mel=speaker_params["ref_mel"].to(self.token2wav.device).float(),
                **token2wav_kwargs,
            )

            return thinker_result.sequences, wav.float()

    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    gpu_prop = torch.cuda.get_device_properties("cuda")
    print(gpu_prop)

    model_path = os.path.join(HF_MODEL_DIR, "Qwen/Qwen2.5-Omni-7B")
    config = AutoConfig.from_pretrained(model_path)
    model = Qwen2_5OmniForConditionalGenerationNew.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
        config=config,
    ).eval()

    # NOTE: when disable talker, generate must set return_audio=False
    # model.disable_talker()

    print_model_params(model, "Qwen2.5Omni")

    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

    subprocess.run("nvidia-smi", shell=True)

    def inference(
        messages,
        return_audio=False,
        use_audio_in_video=False,
        thinker_do_sample=False,
        speaker=DEFAULT_SPEAKER,
    ):
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # image_inputs, video_inputs = process_vision_info([messages])
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

        output = model.generate(
            **inputs,
            use_audio_in_video=use_audio_in_video,
            return_audio=return_audio,
            speaker=speaker,
            thinker_do_sample=thinker_do_sample,
        )
        print("\n====generate use memory=====\n")
        subprocess.run(
            """nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | awk -F',' '{print "GPU "$1": "$2"/"$3" MiB\\n"}'""",
            shell=True,
        )
        print("\n=========\n")
        # print(output)
        text_token_ids = output
        audio = None
        if return_audio and len(output) > 1:
            text_token_ids = output[0].detach()
            audio = output[1].unsqueeze(0).detach()

        text = processor.batch_decode(
            text_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        torch.cuda.empty_cache()

        return text, audio

    def thinker_inference_stream(
        messages,
        use_audio_in_video=False,
        speaker=DEFAULT_SPEAKER,
    ):
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # image_inputs, video_inputs = process_vision_info([messages])
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

        streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            use_audio_in_video=use_audio_in_video,
            return_audio=False,
            speaker=speaker,
            thinker_do_sample=True,
            # do_sample=True,
            top_k=10,
            top_p=0.9,
            temperature=0.95,
            repetition_penalty=1.1,
            min_new_tokens=0,
            max_new_tokens=2048,
        )
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        generated_text = ""
        times = []
        start_time = perf_counter()
        for new_text in streamer:
            times.append(perf_counter() - start_time)
            start_time = perf_counter()
            generated_text += new_text
            yield new_text
        print(
            f"generate [{generated_text}] first token cost time: {times[0]} s, {len(times)} tokens cost time: {sum(times)} s"
        )
        torch.cuda.empty_cache()

    @torch.no_grad()
    def thinker_talker_inference_stream(
        messages,
        use_audio_in_video=False,
        speaker=DEFAULT_SPEAKER,
        talker_eos_token_id: list[int] = [8292, 8294],
    ):
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # image_inputs, video_inputs = process_vision_info([messages])
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

        thinker_result = model.thinker.generate(
            **inputs,
            use_audio_in_video=use_audio_in_video,
            do_sample=True,
            top_k=10,
            top_p=0.9,
            temperature=0.95,
            repetition_penalty=1.1,
            min_new_tokens=0,
            max_new_tokens=2048,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        # print(
        #    thinker_result.sequences.shape,
        #    thinker_result.hidden_states[0][0].shape,
        #    thinker_result.hidden_states[0][-1].shape,
        #    len(thinker_result.hidden_states),
        # )

        # 2. Generate speech tokens from talker module
        input_ids = inputs["input_ids"]
        thinker_generate_ids = thinker_result.sequences[:, input_ids.size(1) :].to(
            model.talker.device
        )
        thinker_token_embeds = [
            token_hidden_states[0].to(model.talker.device)
            for token_hidden_states in thinker_result.hidden_states
        ]
        thinker_hidden_states = [
            token_hidden_states[-1].to(model.talker.device)
            for token_hidden_states in thinker_result.hidden_states
        ]
        gen_text = processor.batch_decode(
            thinker_result.sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        # print(gen_text)

        talker_text_bos_token = model.speaker_map[speaker]["bos_token"]
        talker_input_text_ids = torch.cat(
            [
                input_ids.to(model.talker.device),
                torch.tensor(
                    [[talker_text_bos_token]], dtype=torch.long, device=model.talker.device
                ),
                thinker_generate_ids[:, :1],
            ],
            dim=-1,
        )

        talker_input_ids = torch.cat(
            [
                torch.full_like(
                    input_ids, fill_value=model.talker.codec_mask_token, device=model.talker.device
                ),
                torch.tensor(
                    [[model.talker.codec_pad_token]], dtype=torch.long, device=model.talker.device
                ),
                torch.tensor(
                    [[model.talker.codec_bos_token]], dtype=torch.long, device=model.talker.device
                ),
            ],
            dim=1,
        )
        # print(f"talker_input_ids.shape:{talker_input_ids.shape}")

        thinker_embed_tokens = model.thinker.get_input_embeddings()
        thinker_reply_part = torch.cat(thinker_hidden_states[1:], dim=1) + torch.cat(
            thinker_token_embeds[1:], dim=1
        )
        talker_inputs_embeds = thinker_hidden_states[0] + thinker_token_embeds[0]
        talker_text_bos_token = torch.tensor(
            [[talker_text_bos_token]], dtype=torch.long, device=model.thinker.device
        )
        talker_text_bos_embed = thinker_embed_tokens(talker_text_bos_token).to(model.talker.device)
        talker_inputs_embeds = torch.cat(
            [
                talker_inputs_embeds,
                talker_text_bos_embed,
                thinker_reply_part[:, :1, :],
            ],
            dim=1,
        )

        eos_embedding = thinker_embed_tokens(
            torch.tensor(
                [[model.talker.text_eos_token]], dtype=torch.long, device=model.thinker.device
            )
        ).to(model.talker.device)

        pad_embedding = thinker_embed_tokens(
            torch.tensor(
                [[model.talker.text_pad_token]], dtype=torch.long, device=model.thinker.device
            )
        ).to(model.talker.device)

        thinker_reply_part = torch.cat(
            [
                thinker_reply_part[:, 1:, :],
                eos_embedding,
                pad_embedding,
            ],
            dim=1,
        )
        # print(f"thinker_reply_part.shape:{thinker_reply_part.shape}")

        talker_attention_mask = None
        if "attention_mask" in inputs:
            talker_attention_mask = torch.cat(
                [inputs["attention_mask"], inputs["attention_mask"].new_ones((1, 2))], dim=1
            ).to(model.talker.device)

        # talker_result = model.talker.generate(
        #    input_ids=talker_input_ids,
        #    input_text_ids=talker_input_text_ids,
        #    thinker_reply_part=thinker_reply_part,
        #    inputs_embeds=talker_inputs_embeds,
        #    attention_mask=talker_attention_mask,
        #    suppress_tokens=[model.talker.codec_bos_token],
        #    do_sample=True,
        #    top_k=10,
        #    top_p=0.9,
        #    temperature=0.95,
        #    repetition_penalty=1.1,
        #    min_new_tokens=0,
        #    max_new_tokens=8192,
        # )
        # talker_generate_codes = talker_result[:, talker_input_ids.shape[1] : -1]
        # print(talker_generate_codes)

        streamer = TokenStreamer(skip_prompt=True)
        talker_kwargs = dict(
            input_ids=talker_input_ids,
            streamer=streamer,
            input_text_ids=talker_input_text_ids,
            thinker_reply_part=thinker_reply_part,
            inputs_embeds=talker_inputs_embeds,
            attention_mask=talker_attention_mask,
            suppress_tokens=[model.talker.codec_bos_token],
            eos_token_id=talker_eos_token_id,
            do_sample=True,
            top_k=10,
            top_p=0.9,
            temperature=0.95,
            repetition_penalty=1.1,
            min_new_tokens=0,
            max_new_tokens=8192,
        )
        print(talker_kwargs.keys())
        thread = Thread(target=model.talker.generate, kwargs=talker_kwargs)
        thread.start()

        talker_generate_codes = []
        times = []
        start_time = perf_counter()
        for token_id in streamer:
            times.append(perf_counter() - start_time)
            start_time = perf_counter()
            talker_generate_codes.append(token_id)
        print(
            f"generate first token cost time: {times[0]} s, {len(times)} tokens cost time: {sum(times)} s"
        )
        talker_generate_codes = torch.tensor(
            [talker_generate_codes[:-1]],  # skip last token id talker_eos_token_id
            dtype=torch.long,
            device=model.talker.device,
        )
        # print(f"talker_generate_codes:{talker_generate_codes.shape} {talker_generate_codes}")

        # 3. Generate wavs from code
        if model.token2wav.dtype != torch.float:
            model.token2wav.float()

        # print(model.token2wav.device)
        model_token2wav_device = model.token2wav.device
        # model_token2wav_device = "cpu"
        # model.token2wav.to(model_token2wav_device)

        # print(model.token2wav.device, model.speaker_map[speaker])

        wav = (
            model.token2wav(
                talker_generate_codes.to(model_token2wav_device),
                conditioning=model.speaker_map[speaker]["cond"].to(model_token2wav_device).float(),
                reference_mel=model.speaker_map[speaker]["ref_mel"]
                .to(model_token2wav_device)
                .float(),
                num_steps=10,
                guidance_scale=0.5,
                sway_coefficient=-1.0,
            )
            .unsqueeze(0)
            .detach()
        )

        torch.cuda.empty_cache()
        return gen_text, wav.float()


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L40s"),
    cpu=2.0,
    retries=0,
    image=omni_img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_dir,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=100,
)
def run(func):
    func()


def voice_chatting():
    sys_msg = {
        "role": "system",
        "content": [{"type": "text", "text": SPEECH_SYS_PROMPT}],
    }

    for audio_path in ["guess_age_gender.wav", "translate_to_chinese.wav"]:
        audio_path = os.path.join(ASSETS_DIR, audio_path)
        audio_msg = [
            sys_msg,
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                ],
            },
        ]

        texts, audio = inference(audio_msg, return_audio=True, use_audio_in_video=True)
        print(texts[0], audio.shape)

        save_audio_path = os.path.join(ASSETS_DIR, f"generated_{os.path.basename(audio_path)}")
        torchaudio.save(save_audio_path, audio, sample_rate=24000)
        print(f"Audio saved to {save_audio_path}")


def multi_round_omni_chatting():
    conversations = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SPEECH_SYS_PROMPT}],
        },
    ]
    for video_path in ["draw1.mp4", "draw2.mp4", "draw3.mp4"]:
        video_path = os.path.join(ASSETS_DIR, video_path)
        conversations.append({"role": "user", "content": [{"type": "video", "video": video_path}]})
        texts, audio = inference(conversations, return_audio=True, use_audio_in_video=True)
        print(texts[0], audio.shape)
        save_audio_path = os.path.join(ASSETS_DIR, f"generated_{os.path.basename(video_path)}")
        torchaudio.save(save_audio_path, audio, sample_rate=24000)
        print(f"Audio saved to {save_audio_path}")


def omni_chatting_for_math():
    video_path = os.path.join(ASSETS_DIR, "math.mp4")
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SPEECH_SYS_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
            ],
        },
    ]

    response, audio = inference(messages, return_audio=True, use_audio_in_video=True)
    print(response[0], audio.shape)

    save_audio_path = os.path.join(ASSETS_DIR, f"generated_{os.path.basename(video_path)}")
    torchaudio.save(save_audio_path, audio, sample_rate=24000)
    print(f"Audio saved to {save_audio_path}")


def omni_chatting_for_music():
    video_path = os.path.join(ASSETS_DIR, "music.mp4")
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SPEECH_SYS_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
            ],
        },
    ]
    response, audio = inference(messages, return_audio=True, use_audio_in_video=True)
    print(response[0])

    save_audio_path = os.path.join(ASSETS_DIR, f"generated_{os.path.basename(video_path)}")
    torchaudio.save(save_audio_path, audio, sample_rate=24000)
    print(f"Audio saved to {save_audio_path}")


def screen_recording_interaction():
    video_path = os.path.join(ASSETS_DIR, "screen.mp4")
    for prompt in [
        "What the browser is used in this video?",
        "浏览器中的论文叫什么名字?",
        "这篇论文主要解决什么问题呢？",
    ]:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video", "video": video_path},
                ],
            },
        ]

        response, _ = inference(messages, return_audio=False, use_audio_in_video=False)
        print(response[0])


def universal_audio_understanding():
    for case in [
        {
            "audio_path": "1272-128104-0000.flac",
            "prompt": "Transcribe the English audio into text without any punctuation marks.",
            "sys_prompt": "You are a speech recognition model.",
        },
        {
            "audio_path": "BAC009S0764W0121.wav",
            "prompt": "请将这段中文语音转换为纯文本，去掉标点符号。",
            "sys_prompt": "You are a speech recognition model.",
        },
        {
            "audio_path": "10000611681338527501.wav",
            "prompt": "Transcribe the Russian audio into text without including any punctuation marks.",
            "sys_prompt": "You are a speech recognition model.",
        },
        {
            "audio_path": "7105431834829365765.wav",
            "prompt": "Transcribe the French audio into text without including any punctuation marks.",
            "sys_prompt": "You are a speech recognition model.",
        },
        {
            "audio_path": "1272-128104-0000.flac",
            "prompt": "Listen to the provided English speech and produce a translation in Chinese text.",
            "sys_prompt": "You are a speech translation model.",
        },
        {
            "audio_path": "cough.wav",
            "prompt": "Classify the given human vocal sound in English.",
            "sys_prompt": "You are a voice classification model.",
        },
    ]:
        audio_path = os.path.join(ASSETS_DIR, case["audio_path"])
        messages = [
            {"role": "system", "content": [{"type": "text", "text": case["sys_prompt"]}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": case["prompt"]},
                    {"type": "audio", "audio": audio_path},
                ],
            },
        ]
        texts, _ = inference(messages, use_audio_in_video=True, return_audio=False)
        print(texts[0])


def video_information_extracting():
    video_path = os.path.join(ASSETS_DIR, "shopping.mp4")
    sys_msg = {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}],
    }
    for prompt in [
        "How many kind of drinks can you see in the video?",
        "How many bottles of drinks have I picked up?",
        "How many milliliters are there in the bottle I picked up second time?",
        "视屏中的饮料叫什么名字呢？",
        "跑步🏃🏻累了，适合喝什么饮料补充体力呢？",
    ]:
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
        texts, _ = inference(messages, return_audio=False, use_audio_in_video=False)
        print(texts[0])


def batch_requests():
    """need return_audio=False"""
    # Conversation with video only
    conversation1 = [
        {"role": "system", "content": [{"type": "text", "text": SPEECH_SYS_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": os.path.join(ASSETS_DIR, "draw1.mp4")},
            ],
        },
    ]

    # Conversation with audio only
    conversation2 = [
        {"role": "system", "content": [{"type": "text", "text": SPEECH_SYS_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": os.path.join(ASSETS_DIR, "1272-128104-0000.flac")},
            ],
        },
    ]

    # Conversation with pure text
    conversation3 = [
        {"role": "system", "content": [{"type": "text", "text": SPEECH_SYS_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": "who are you?"}]},
    ]

    # Conversation with mixed media
    conversation4 = [
        {"role": "system", "content": [{"type": "text", "text": SPEECH_SYS_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": os.path.join(ASSETS_DIR, "03-Confusing-Pictures.jpg")},
                {"type": "video", "video": os.path.join(ASSETS_DIR, "music.mp4")},
                {"type": "audio", "audio": os.path.join(ASSETS_DIR, "1272-128104-0000.flac")},
                {
                    "type": "text",
                    "text": "What are the elements can you see and hear in these medias?",
                },
            ],
        },
    ]

    # Combine messages for batch processing
    conversations = [conversation1, conversation2, conversation3, conversation4]
    texts, _ = inference(conversations, return_audio=False, use_audio_in_video=True)
    print(texts)


def asr_stream():
    for case in [
        {
            "audio_path": "1272-128104-0000.flac",
            "prompt": "Listen to the provided English speech and produce a translation in Chinese text.",
            "sys_prompt": "You are a speech translation model.",
        },
        {
            "audio_path": "BAC009S0764W0121.wav",
            "prompt": "请将这段中文语音转换为纯文本，去掉标点符号。",
            "sys_prompt": "You are a speech recognition model.",
        },
    ]:
        audio_path = os.path.join(ASSETS_DIR, case["audio_path"])
        messages = [
            {"role": "system", "content": [{"type": "text", "text": case["sys_prompt"]}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": case["prompt"]},
                    {"type": "audio", "audio": audio_path},
                ],
            },
        ]
        text_streamer = thinker_inference_stream(messages, use_audio_in_video=True)
        for text in text_streamer:
            print(text)


def omni_chatting_stream():
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SPEECH_SYS_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": "who are you?"}]},
    ]
    # response, audio = inference(
    #    messages, return_audio=True, use_audio_in_video=False, thinker_do_sample=True
    # )
    response, audio = thinker_talker_inference_stream(messages, use_audio_in_video=False)
    print(response[0])

    save_audio_path = os.path.join(ASSETS_DIR, f"generated_omni_chatting_stream.wav")
    torchaudio.save(save_audio_path, audio, sample_rate=24000)
    print(f"Audio saved to {save_audio_path}")


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


"""
# NOTE: if want to generate speech, need use SPEECH_SYS_PROMPT to generate speech

# asr (audio understanding)
IMAGE_GPU=L4 modal run src/llm/transformers/qwen2_5omni.py --task universal_audio_understanding

# audio to text and speech
IMAGE_GPU=L40s modal run src/llm/transformers/qwen2_5omni.py --task voice_chatting

# vision(video no audio) to text
IMAGE_GPU=L40s modal run src/llm/transformers/qwen2_5omni.py --task video_information_extracting
IMAGE_GPU=L40s modal run src/llm/transformers/qwen2_5omni.py --task screen_recording_interaction

# vision(video with audio) to text and speech
IMAGE_GPU=L40s modal run src/llm/transformers/qwen2_5omni.py --task omni_chatting_for_math
IMAGE_GPU=L40s modal run src/llm/transformers/qwen2_5omni.py --task omni_chatting_for_music

# vision(video with audio) to text and speech with multi rounds chat, but need more GPU memory
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen2_5omni.py --task multi_round_omni_chatting

# batch requests
IMAGE_GPU=A100-80GB modal run src/llm/transformers/qwen2_5omni.py --task batch_requests

# stream
IMAGE_GPU=L4 modal run src/llm/transformers/qwen2_5omni.py --task asr_stream
IMAGE_GPU=L4 modal run src/llm/transformers/qwen2_5omni.py --task omni_chatting_stream
"""


@app.local_entrypoint()
def main(task: str = "universal_audio_understanding"):
    tasks = {
        "universal_audio_understanding": universal_audio_understanding,
        "voice_chatting": voice_chatting,
        "video_information_extracting": video_information_extracting,
        "screen_recording_interaction": screen_recording_interaction,
        "omni_chatting_for_math": omni_chatting_for_math,
        "omni_chatting_for_music": omni_chatting_for_music,
        "multi_round_omni_chatting": multi_round_omni_chatting,
        "batch_requests": batch_requests,
        "asr_stream": asr_stream,
        "omni_chatting_stream": omni_chatting_stream,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(tasks[task])
