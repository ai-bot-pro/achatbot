import logging
from time import perf_counter
from typing import Generator, Optional

import torch

try:
    from transformers import (
        Qwen3OmniMoeForConditionalGeneration,
        Qwen3OmniMoeConfig,
        Qwen3OmniMoeCode2Wav,
    )
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use Qwen3Omni, you need to `pip install git+https://github.com/huggingface/transformers`"
    )
    raise Exception(f"Missing module: {e}")


class Qwen3OmniMoeCode2WavStreaming(Qwen3OmniMoeCode2Wav):
    def chunked_decode_stream(self, codes, chunk_size=50, left_context_size=25):
        start_index = 0
        while start_index < codes.shape[-1]:
            end_index = min(start_index + chunk_size, codes.shape[-1])
            context_size = left_context_size if start_index - left_context_size > 0 else start_index
            codes_chunk = codes[..., start_index - context_size : end_index]
            wav_chunk = self(codes_chunk)
            yield wav_chunk[..., context_size * self.total_upsample :]
            start_index = end_index


class Qwen3OmniMoeForConditionalGenerationStreaming(Qwen3OmniMoeForConditionalGeneration):
    def __init__(self, config: Qwen3OmniMoeConfig):
        super().__init__(config)
        if hasattr(self, "code2wav"):
            self.code2wav = Qwen3OmniMoeCode2WavStreaming(config.code2wav_config)

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
        {
            "thinker_generate_ids": # [1,seq_len],
            "thinker_generate_hidden_states":  # tuple(tuple(tensor[1, 1, 2048])...)
            "full_generated_ids": # tuple(tuple(tensor[1, 1, 2048])...)
        }
        """
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask", None)

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
            # logging.info(current_input_ids, current_attention_mask.shape)
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
            model_inputs["input_ids"] = current_input_ids
            model_inputs["attention_mask"] = current_attention_mask
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
            total_new_tokens_generated += num_step_new_tokens

            if num_step_new_tokens == 0:  # Handle case where generate stops early
                logging.warning("Warning: generate produced 0 new tokens in this step.")
                break

            if thinker_output_hidden_states is True:
                hidden_states = outputs.hidden_states
                hidden_states_len = (
                    hidden_states_len if hidden_states_len > 0 else hidden_states[0][0].shape[1]
                )
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
    def talker_generate(
        self,
        hidden_states: tuple,
        sequences: torch.Tensor,
        input_ids: torch.Tensor,
        speaker_id: int,
        talker_kwargs: dict,
        token2wav_kwargs: dict,
    ) -> Generator[torch.Tensor, None, None]:
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
            self.talker.text_projection(self.thinker.get_input_embeddings()(talker_special_tokens))
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
                talker_input_ids.append(sequences[:, im_start_index:segment_end_index])
            # Take assistant output (for now)
            elif role_token == self.config.assistant_token_id and i == len(im_start_indexes) - 2:
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
            elif role_token == self.config.assistant_token_id and i != len(im_start_indexes) - 2:
                continue
            else:
                raise AssertionError("Expect role id after <|im_start|> (assistant, user, system)")
        talker_input_embed = torch.cat(
            [embed.to(self.talker.device) for embed in talker_input_embeds], dim=1
        )
        talker_input_id = torch.cat(
            [embed.to(self.talker.device) for embed in talker_input_ids], dim=1
        )

        talker_result = self.talker.generate(
            attention_mask=torch.ones(talker_input_id.shape, device=talker_input_id.device),
            pad_token_id=2150,
            inputs_embeds=talker_input_embed,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            talker_input_ids=talker_input_id,  # Not use input_ids to prevent repetation penalty out of bound
            **talker_kwargs,
        )

        talker_codes = (
            torch.stack(
                [hid[-1] for hid in talker_result.hidden_states if hid[-1] is not None], dim=1
            )
            .transpose(1, 2)
            .to(self.code2wav.device)
        )

        chunk_size = token2wav_kwargs.get("chunk_size", 50)
        left_context_size = token2wav_kwargs.get("left_context_size", 25)
        return self.code2wav.chunked_decode_stream(
            talker_codes, chunk_size=chunk_size, left_context_size=left_context_size
        )

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
        token2wav_kwargs: dict = {"chunk_size": 50, "left_context_size": 25},
        skip_token_ids: Optional[list] = [],
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
            for thinker_chunk in thinker_chunk_stream:
                yield {
                    "thinker_ids": thinker_chunk["thinker_generate_ids"],
                }
                if (
                    thinker_chunk["thinker_generate_ids"].shape[-1] == 1
                    and thinker_chunk["thinker_generate_ids"][0, -1].item() in skip_token_ids
                ):
                    # Skip if only one token generated, which is likely to be a stop token
                    continue
                thinker_generate_hidden_states = thinker_chunk["thinker_generate_hidden_states"]
                full_generated_ids = thinker_chunk["full_generated_ids"]
                wav_iter = self.talker_generate(
                    hidden_states=thinker_generate_hidden_states,
                    sequences=full_generated_ids,
                    input_ids=input_ids,
                    speaker_id=speaker_id,
                    talker_kwargs=talker_kwargs,
                    token2wav_kwargs=token2wav_kwargs,
                )  # [1,1,T]
                for talker_wav in wav_iter:
                    yield {
                        "talker_wav": talker_wav.float(),  # BFloat16 -> Float32
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

        # logging.info(f"talker input sequences: {thinker_result.sequences}")
        # logging.info(f"hidden_states tuples size {len(thinker_result.hidden_states)}")
        # for i, item in enumerate(thinker_result.hidden_states):
        #    if isinstance(item, tuple):
        #        logging.info(
        #            f"hidden_states tuple len: {len(item)}, item shape:{item[0].shape}",
        #        )
        #    elif isinstance(item, torch.Tensor):
        #        logging.info(f"{i=}", item.shape)
        #    else:
        #        logging.info(f"{i=}", item)

        talker_wavs = self.talker_generate(
            hidden_states=thinker_result.hidden_states,
            sequences=thinker_result.sequences,
            input_ids=input_ids,
            speaker_id=speaker_id,
            talker_kwargs=talker_kwargs,
            token2wav_kwargs=token2wav_kwargs,
        )  # [1,1,T]
        return thinker_result, talker_wavs
