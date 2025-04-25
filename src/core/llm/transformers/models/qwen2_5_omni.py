import logging
from threading import Thread
from time import perf_counter
from typing import Generator, Optional, Callable

import torch


try:
    from transformers import (
        Qwen2_5OmniForConditionalGeneration,
    )
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use Qwen2.5Omni, you need to `pip install git+https://github.com/huggingface/transformers`"
    )
    raise Exception(f"Missing module: {e}")


from src.common.utils.helper import print_model_params
from src.core.llm.transformers.streamer import TokenStreamer


class Qwen2_5OmniForConditionalGenerationStreaming(Qwen2_5OmniForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print_model_params(self.thinker, "qwen2.5omni_thinker")
        if self.has_talker:
            print_model_params(self.talker, "qwen2.5omni_talker")
            print_model_params(self.token2wav, "qwen2.5omni_token2wav")
            self.token2wav.float()

    @torch.no_grad()
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
        talker_eos_token_ids: list[int] = [8292, 8294],
        talker_repetition_penalty: float = 1.05,
        **kwargs,
    ):
        r"""
        Generate text response and audio from input.

        Args:
            input_ids (`Optional[torch.Tensor]`, *optional*):
                Input ids, should obtain from self._tokenizer.
            speaker (`str` , defaults to "Chelsie"):
                Which speaker should be used in audio response.
            use_audio_in_video (`bool`, defaults to False):
                Whether or not use audio track in video, should same as the parameter in `process_audio_info`.
            return_audio (`Optional[bool]`, *optional*):
                Whether or not return response in audio format. When `return_audio=None`, this parameter is same as `config.enable_audio_output`.
            kwargs (*optional*):
                - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-self.
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
            "eos_token_id": talker_eos_token_ids,
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
        embeds_to_talker = thinker_result.hidden_states[0][0].clone().to(self.talker.device)
        if thinker_kwargs.get("input_features", None) is not None:
            audio_ids_mask = input_ids == self.config.thinker_config.audio_token_index
            audio_mask = (
                audio_ids_mask.unsqueeze(-1).expand_as(embeds_to_talker).to(embeds_to_talker.device)
            )
            audio_mask_tensor = torch.zeros(
                [audio_ids_mask.sum(), embeds_to_talker.shape[-1]],
                dtype=embeds_to_talker.dtype,
                device=self.talker.device,
            )
            embeds_to_talker.masked_scatter_(audio_mask, audio_mask_tensor)
        if thinker_kwargs.get("pixel_values", None) is not None:
            image_ids_mask = input_ids == self.config.thinker_config.image_token_index
            image_mask = (
                image_ids_mask.unsqueeze(-1).expand_as(embeds_to_talker).to(embeds_to_talker.device)
            )
            image_mask_tensor = torch.zeros(
                [image_ids_mask.sum(), embeds_to_talker.shape[-1]],
                dtype=embeds_to_talker.dtype,
                device=self.talker.device,
            )
            embeds_to_talker.masked_scatter_(image_mask, image_mask_tensor)
        if thinker_kwargs.get("pixel_values_videos", None) is not None:
            video_ids_mask = input_ids == self.config.thinker_config.video_token_index
            video_mask = (
                video_ids_mask.unsqueeze(-1).expand_as(embeds_to_talker).to(embeds_to_talker.device)
            )
            video_mask_tensor = torch.zeros(
                [video_ids_mask.sum(), embeds_to_talker.shape[-1]],
                dtype=embeds_to_talker.dtype,
                device=self.talker.device,
            )
            embeds_to_talker.masked_scatter_(video_mask, video_mask_tensor)

        processed_thinker_hidden = (
            (embeds_to_talker,) + thinker_result.hidden_states[0][1:],
        ) + thinker_result.hidden_states[1:]

        thinker_generate_ids = thinker_result.sequences[:, input_ids.size(1) :].to(
            self.talker.device
        )
        thinker_token_embeds = [
            token_hidden_states[0].to(self.talker.device)
            for token_hidden_states in processed_thinker_hidden
        ]
        thinker_hidden_states = [
            token_hidden_states[-1].to(self.talker.device)
            for token_hidden_states in processed_thinker_hidden
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
                    input_ids, fill_value=self.talker.codec_mask_token, device=self.talker.device
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
        talker_text_bos_embed = thinker_embed_tokens(talker_text_bos_token).to(self.talker.device)
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
        # logging.debug(talker_kwargs.keys())
        thread = Thread(target=self.talker.generate, kwargs=talker_kwargs)
        thread.start()
        talker_generate_codes = []
        times = []
        start_time = perf_counter()
        for token_id in streamer:
            times.append(perf_counter() - start_time)
            start_time = perf_counter()
            talker_generate_codes.append(token_id)
        logging.info(
            f"generate first token cost time: {times[0]} s, {len(times)} tokens cost time: {sum(times)} s"
        )
        offset = 0
        if skip_prompt is False:
            offset = talker_input_ids.shape[1]
        # logging.debug(
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

        # logging.debug(f"talker_generate_codes:{talker_generate_codes.shape} {talker_generate_codes}")

        # 3. Generate wavs from code
        # logging.debug(self.token2wav.device, speaker_params, token2wav_kwargs)
        wav = self.token2wav(
            talker_generate_codes.to(self.token2wav.device),
            conditioning=speaker_params["cond"].to(self.token2wav.device).float(),
            reference_mel=speaker_params["ref_mel"].to(self.token2wav.device).float(),
            **token2wav_kwargs,
        )

        return thinker_result.sequences, wav.float()

    @torch.no_grad()
    def thinker_generate_chunk(
        self,
        inputs: dict,
        use_audio_in_video: bool = False,
        thinker_max_tokens_per_step=10,  # Controls how many tokens to generate *per step*
        thinker_max_new_tokens: int = 1024,
        thinker_top_k: int = 40,
        thinker_top_p: float = 0.8,
        thinker_temperature: float = 0.9,
        thinker_eos_token_ids=[151644, 151645],  # Define EOS tokens
        thinker_repetition_penalty: float = 1.05,
        thinker_output_hidden_states=False,
        thinker_stop_strings_per_step=[],
        tokenizer=None,
        **kwargs,
    ):
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
            # logging.debug(current_input_ids, current_attention_mask.shape)
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
            outputs = self.thinker.generate(**model_inputs)
            times.append(perf_counter() - start_time)

            # Extract newly generated token IDs *for this step*
            # `outputs.sequences` contains the input_ids for this step + new tokens generated in this step
            step_new_ids = outputs.sequences[:, current_input_ids.shape[1] :]
            num_step_new_tokens = step_new_ids.shape[1]

            if num_step_new_tokens == 0:  # Handle case where generate stops early
                logging.warning("Warning: generate produced 0 new tokens in this step.")
                break

            if thinker_output_hidden_states is True:
                hidden_states = outputs.hidden_states
                hidden_states_len = (
                    hidden_states_len if hidden_states_len > 0 else hidden_states[0][0].shape[1]
                )
                logging.debug(f"hidden_states_len: {hidden_states_len}")
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

            yield {
                "thinker_generate_ids": step_new_ids,
                "thinker_generate_hidden_states": hidden_states,
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
            if step_new_ids[0, -1].item() in thinker_eos_token_ids:
                logging.info("EOS token generated.")
                break

            # Check if max_new_tokens limit is reached (after processing the step)
            if total_new_tokens_generated >= thinker_max_new_tokens:
                logging.info("Max new tokens limit reached.")
                break

        logging.info(
            f"Total new tokens generated: {total_new_tokens_generated} | thinker_max_tokens_per_step: {thinker_max_tokens_per_step} | first chunk generated cost: {times[0]} s | total cost: {sum(times)} s"
        )

    @torch.no_grad()
    def talker_generate_chunk(
        self,
        inputs: dict,
        thinker_chunk_stream,
        speaker: str = "Chelsie",
        talker_eos_token_ids: list[int] = [8292, 8294],
        talker_top_k: int = 10,
        talker_top_p: float = 0.9,
        talker_temperature: float = 0.95,
        talker_repetition_penalty: float = 1.1,
        talker_min_new_tokens: int = 0,
        talker_max_new_tokens: int = 8192,
        talker_skip_thinker_token_ids: list[int] = [],  # skip tokens don't to talk
        code2wav_num_steps: int = 10,
        code2wav_guidance_scale: float = 0.5,
        code2wav_sway_coefficient: float = -1.0,
        code2wav_chunk_stream_func: Callable = None,
        mask_embedding: bool = True,
    ) -> Generator[dict, None, None]:
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask", None)

        for chunk in thinker_chunk_stream:
            thinker_generate_ids = chunk["thinker_generate_ids"].to(self.talker.device)
            # skip talk
            if (
                thinker_generate_ids.shape[1] == 1
                and thinker_generate_ids[0, -1].item() in talker_skip_thinker_token_ids
            ):
                logging.info(f"skip token {thinker_generate_ids} to talk")
                yield {"thinker_ids": thinker_generate_ids, "talker_wav": torch.empty([0])}
                continue
            thinker_generate_hidden_states = chunk["thinker_generate_hidden_states"]
            if thinker_generate_hidden_states is None or len(thinker_generate_hidden_states) < 2:
                if len(thinker_generate_hidden_states) < 2:
                    logging.warning(
                        f"thinker_generate_ids: {thinker_generate_ids} | len(thinker_generate_hidden_states): {len(thinker_generate_hidden_states)} < 2"
                    )
                yield {"thinker_ids": thinker_generate_ids, "talker_wav": torch.empty([0])}
                continue

            processed_thinker_hidden = thinker_generate_hidden_states
            if mask_embedding is True:
                logging.info("mask embedding")
                embeds_to_talker = (
                    thinker_generate_hidden_states[0][0].clone().to(self.talker.device)
                )
                if inputs.get("input_features", None) is not None:
                    audio_ids_mask = input_ids == self.config.thinker_config.audio_token_index
                    audio_mask = (
                        audio_ids_mask.unsqueeze(-1)
                        .expand_as(embeds_to_talker)
                        .to(embeds_to_talker.device)
                    )
                    audio_mask_tensor = torch.zeros(
                        [audio_ids_mask.sum(), embeds_to_talker.shape[-1]],
                        dtype=embeds_to_talker.dtype,
                        device=self.talker.device,
                    )
                    embeds_to_talker.masked_scatter_(audio_mask, audio_mask_tensor)
                if inputs.get("pixel_values", None) is not None:
                    image_ids_mask = input_ids == self.config.thinker_config.image_token_index
                    image_mask = (
                        image_ids_mask.unsqueeze(-1)
                        .expand_as(embeds_to_talker)
                        .to(embeds_to_talker.device)
                    )
                    image_mask_tensor = torch.zeros(
                        [image_ids_mask.sum(), embeds_to_talker.shape[-1]],
                        dtype=embeds_to_talker.dtype,
                        device=self.talker.device,
                    )
                    embeds_to_talker.masked_scatter_(image_mask, image_mask_tensor)
                if inputs.get("pixel_values_videos", None) is not None:
                    video_ids_mask = input_ids == self.config.thinker_config.video_token_index
                    video_mask = (
                        video_ids_mask.unsqueeze(-1)
                        .expand_as(embeds_to_talker)
                        .to(embeds_to_talker.device)
                    )
                    video_mask_tensor = torch.zeros(
                        [video_ids_mask.sum(), embeds_to_talker.shape[-1]],
                        dtype=embeds_to_talker.dtype,
                        device=self.talker.device,
                    )
                    embeds_to_talker.masked_scatter_(video_mask, video_mask_tensor)

                processed_thinker_hidden = (
                    (embeds_to_talker,) + thinker_generate_hidden_states[0][1:],
                ) + thinker_generate_hidden_states[1:]

            thinker_token_embeds = [
                token_hidden_states[0].to(self.talker.device)
                for token_hidden_states in processed_thinker_hidden
            ]
            thinker_hidden_states = [
                token_hidden_states[-1].to(self.talker.device)
                for token_hidden_states in processed_thinker_hidden
            ]
            logging.debug(
                f"len(thinker_generate_hidden_states):{len(thinker_generate_hidden_states)}"
            )
            for i in range(len(thinker_generate_hidden_states)):
                logging.debug(
                    f"thinker_generate_hidden_states[{i}]:{thinker_generate_hidden_states[i][0].shape}, {thinker_generate_hidden_states[i][-1].shape}"
                )

            talker_text_bos_token = self.speaker_map[speaker]["bos_token"]
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
            logging.debug(f"talker_input_text_ids.shape:{talker_input_text_ids.shape}")

            talker_input_ids = torch.cat(
                [
                    torch.full_like(
                        input_ids,
                        fill_value=self.talker.codec_mask_token,
                        device=self.talker.device,
                    ),
                    torch.tensor(
                        [[self.talker.codec_pad_token]],
                        dtype=torch.long,
                        device=self.talker.device,
                    ),
                    torch.tensor(
                        [[self.talker.codec_bos_token]],
                        dtype=torch.long,
                        device=self.talker.device,
                    ),
                ],
                dim=1,
            )
            logging.debug(f"talker_input_ids.shape:{talker_input_ids.shape}")

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
            logging.debug(
                f"talker_inputs_embeds.shape {talker_inputs_embeds.shape} talker_text_bos_embed.shape {talker_text_bos_embed.shape} thinker_reply_part.shape {thinker_reply_part.shape}"
            )
            talker_inputs_embeds = torch.cat(
                [
                    talker_inputs_embeds,
                    talker_text_bos_embed,
                    thinker_reply_part[:, :1, :],
                ],
                dim=1,
            )
            logging.debug(
                f"talker_inputs_embeds.shape {talker_inputs_embeds.shape} talker_text_bos_embed.shape {talker_text_bos_embed.shape}"
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
            logging.debug(f"thinker_reply_part.shape:{thinker_reply_part.shape}")

            talker_attention_mask = None
            if attention_mask is not None:
                talker_attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((1, 2))], dim=1
                ).to(self.talker.device)

            streamer = TokenStreamer(skip_prompt=True)
            talker_kwargs = dict(
                input_ids=talker_input_ids,
                streamer=streamer,
                input_text_ids=talker_input_text_ids,
                thinker_reply_part=thinker_reply_part,
                inputs_embeds=talker_inputs_embeds,
                attention_mask=talker_attention_mask,
                suppress_tokens=[self.talker.codec_bos_token],
                eos_token_id=talker_eos_token_ids,
                pad_token_id=8292,
                do_sample=True if talker_temperature > 0.0 else False,
                top_k=talker_top_k,
                top_p=talker_top_p,
                temperature=talker_temperature,
                repetition_penalty=talker_repetition_penalty,
                min_new_tokens=talker_min_new_tokens,
                max_new_tokens=talker_max_new_tokens,
            )
            # logging.debug(talker_kwargs.keys())
            thread = Thread(target=self.talker.generate, kwargs=talker_kwargs)
            thread.start()

            code2wav_chunk_stream_func = code2wav_chunk_stream_func or self.code2wav_chunk_stream
            # Generate wavs from code
            code2wav_stream = code2wav_chunk_stream_func(
                talker_streamer=streamer,
                speaker=speaker,
                talker_eos_token_ids=talker_eos_token_ids,
                code2wav_num_steps=code2wav_num_steps,
                code2wav_guidance_scale=code2wav_guidance_scale,
                code2wav_sway_coefficient=code2wav_sway_coefficient,
            )

            for wav in code2wav_stream:
                yield {"thinker_ids": thinker_generate_ids, "talker_wav": wav}

    @torch.no_grad()
    def code2wav_chunk_stream(
        self,
        talker_streamer: TokenStreamer,
        speaker: str = "Chelsie",
        talker_eos_token_ids: list[int] = [8292, 8294],
        code2wav_num_steps: int = 10,
        code2wav_guidance_scale: float = 0.5,
        code2wav_sway_coefficient: float = -1.0,
    ) -> Generator[torch.Tensor, None, None]:
        """
        fixed chunk stream
        """
        if self.token2wav.dtype != torch.float:
            self.token2wav.float()

        code2wav_times = []
        talker_generate_codes = []
        times = []
        start_time = perf_counter()
        pre_offset = 0
        for token_id in talker_streamer:
            times.append(perf_counter() - start_time)
            start_time = perf_counter()
            if token_id in talker_eos_token_ids:
                break
            talker_generate_codes.append(token_id)
            chunk_code_length = len(talker_generate_codes) * 2 - 24
            if chunk_code_length > 0 and chunk_code_length % 48 == 0:
                codes_tensor = torch.tensor(
                    [talker_generate_codes[pre_offset:]],
                    dtype=torch.long,
                    device=self.talker.device,
                )
                pre_offset = len(talker_generate_codes)
                wav = self.token2wav(
                    codes_tensor.to(self.token2wav.device),
                    conditioning=self.speaker_map[speaker]["cond"]
                    .to(self.token2wav.device)
                    .float(),
                    reference_mel=self.speaker_map[speaker]["ref_mel"]
                    .to(self.token2wav.device)
                    .float(),
                    num_steps=10,
                    guidance_scale=0.5,
                    sway_coefficient=-1.0,
                ).detach()
                code2wav_times.append(perf_counter() - start_time)
                yield wav  # (T,)
                start_time = perf_counter()

        logging.info(
            f"talker generate first token cost time: {times[0]} s, {len(times)} tokens cost time: {sum(times)} s"
        )

        if len(talker_generate_codes) > pre_offset:
            codes_tensor = torch.tensor(
                [talker_generate_codes[pre_offset:]],
                dtype=torch.long,
                device=self.talker.device,
            )
            wav = self.token2wav(
                codes_tensor.to(self.token2wav.device),
                conditioning=self.speaker_map[speaker]["cond"].to(self.token2wav.device).float(),
                reference_mel=self.speaker_map[speaker]["ref_mel"]
                .to(self.token2wav.device)
                .float(),
                num_steps=code2wav_num_steps,
                guidance_scale=code2wav_guidance_scale,
                sway_coefficient=code2wav_sway_coefficient,
            ).detach()
            code2wav_times.append(perf_counter() - start_time)
            yield wav  # (T,)

        logging.info(
            f"code2wav streaming first chunk time: {code2wav_times[0]} s | cost: {sum(code2wav_times)} s"
        )

    @torch.no_grad()
    def generate_stream(
        self,
        inputs: dict,
        use_audio_in_video: bool = False,
        thinker_max_tokens_per_step=10,  # Controls how many tokens to generate *per step*
        thinker_max_new_tokens: int = 1024,
        thinker_top_k: int = 40,
        thinker_top_p: float = 0.8,
        thinker_temperature: float = 0.9,
        thinker_repetition_penalty: float = 1.05,
        thinker_eos_token_ids=[151644, 151645],
        thinker_stop_strings_per_step=[],
        tokenizer=None,
        return_audio=True,
        speaker="Chelsie",
        talker_top_k: int = 10,
        talker_top_p: float = 0.9,
        talker_temperature: float = 0.95,
        talker_repetition_penalty: float = 1.1,
        talker_min_new_tokens: int = 0,
        talker_max_new_tokens: int = 8192,
        talker_eos_token_ids: list[int] = [8292, 8294],
        talker_skip_thinker_token_ids: list[int] = [],
        code2wav_num_steps: int = 10,
        code2wav_guidance_scale: float = 0.5,
        code2wav_sway_coefficient: float = -1.0,
        code2wav_chunk_stream_func: Callable = None,
        mask_embedding: bool = True,
        **kwargs,
    ) -> Generator[dict, None, None]:
        """
        - return Generator[dict, None, None]
        {
            "thinker_ids": torch.Tensor, # (1,T)
            "talker_wav": torch.Tensor, # (T,)
        }
        """
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
            talker_streamer = self.talker_generate_chunk(
                inputs,
                thinker_chunk_stream=thinker_chunk_stream,
                speaker=speaker,
                talker_eos_token_ids=talker_eos_token_ids,
                talker_top_k=talker_top_k,
                talker_top_p=talker_top_p,
                talker_temperature=talker_temperature,
                talker_repetition_penalty=talker_repetition_penalty,
                talker_min_new_tokens=talker_min_new_tokens,
                talker_max_new_tokens=talker_max_new_tokens,
                talker_skip_thinker_token_ids=talker_skip_thinker_token_ids,
                code2wav_num_steps=code2wav_num_steps,
                code2wav_guidance_scale=code2wav_guidance_scale,
                code2wav_sway_coefficient=code2wav_sway_coefficient,
                code2wav_chunk_stream_func=code2wav_chunk_stream_func,
                mask_embedding=mask_embedding,
            )

            for talker_chunk in talker_streamer:
                yield talker_chunk

    @torch.no_grad()
    def thinker_all_talker_stream(
        self,
        inputs: dict,
        use_audio_in_video: bool = False,
        thinker_max_new_tokens: int = 1024,
        thinker_top_k: int = 40,
        thinker_top_p: float = 0.8,
        thinker_temperature: float = 0.9,
        thinker_repetition_penalty: float = 1.05,
        thinker_eos_token_ids=[151644, 151645],
        thinker_stop_strings_per_step=[],
        tokenizer=None,
        return_audio=True,
        speaker="Chelsie",
        talker_top_k: int = 10,
        talker_top_p: float = 0.9,
        talker_temperature: float = 0.95,
        talker_repetition_penalty: float = 1.1,
        talker_min_new_tokens: int = 0,
        talker_max_new_tokens: int = 8192,
        talker_eos_token_ids: list[int] = [8292, 8294],
        talker_skip_thinker_token_ids: list[int] = [],
        code2wav_num_steps: int = 10,
        code2wav_guidance_scale: float = 0.5,
        code2wav_sway_coefficient: float = -1.0,
        code2wav_chunk_stream_func: Callable = None,
        mask_embedding: bool = True,
        **kwargs,
    ) -> Generator[dict, None, None]:
        """
        - return Generator[dict, None, None]
        {
            "thinker_ids": torch.Tensor, # (1,T)
            "talker_wav": torch.Tensor, # (T,)
        }
        """

        def to_generator(item):
            yield item

        thinker_result = self.thinker.generate(
            **inputs,
            use_audio_in_video=use_audio_in_video,
            do_sample=True if thinker_temperature > 0 else False,
            top_k=thinker_top_k,
            top_p=thinker_top_p,
            temperature=thinker_temperature,
            repetition_penalty=thinker_repetition_penalty,
            min_new_tokens=1,
            max_new_tokens=thinker_max_new_tokens,
            eos_token_id=thinker_eos_token_ids,
            output_hidden_states=return_audio,
            return_dict_in_generate=True,
        )
        input_ids = inputs["input_ids"]
        thinker_generate_ids = thinker_result.sequences[:, input_ids.size(1) :]
        if not return_audio:
            yield {"thinker_ids": thinker_generate_ids}
        else:
            talker_streamer = self.talker_generate_chunk(
                inputs,
                thinker_chunk_stream=to_generator(
                    {
                        "thinker_generate_ids": thinker_generate_ids,
                        "thinker_generate_hidden_states": thinker_result.hidden_states,
                    }
                ),
                speaker=speaker,
                talker_eos_token_ids=talker_eos_token_ids,
                talker_top_k=talker_top_k,
                talker_top_p=talker_top_p,
                talker_temperature=talker_temperature,
                talker_repetition_penalty=talker_repetition_penalty,
                talker_min_new_tokens=talker_min_new_tokens,
                talker_max_new_tokens=talker_max_new_tokens,
                talker_skip_thinker_token_ids=talker_skip_thinker_token_ids,
                code2wav_num_steps=code2wav_num_steps,
                code2wav_guidance_scale=code2wav_guidance_scale,
                code2wav_sway_coefficient=code2wav_sway_coefficient,
                code2wav_chunk_stream_func=code2wav_chunk_stream_func,
                mask_embedding=mask_embedding,
            )

            for talker_chunk in talker_streamer:
                yield talker_chunk
