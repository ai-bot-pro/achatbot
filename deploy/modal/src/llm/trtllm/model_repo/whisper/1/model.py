# -*- coding: utf-8 -*-

import json
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.dlpack import to_dlpack
import triton_python_backend_utils as pb_utils
from tensorrt_llm.bindings import GptJsonConfig
from tensorrt_llm.runtime import ModelRunnerCpp

from .tokenizer import get_tokenizer
from .fbank import FeatureExtractor


def read_config(component, engine_dir):
    config_path = engine_dir / component / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    model_config = OrderedDict()
    model_config.update(config["pretrained_config"])
    model_config.update(config["build_config"])
    return model_config


class TritonPythonModel:
    def initialize(self, args):
        self.device = torch.device("cuda")

        # parameters
        model_config = json.loads(args["model_config"])
        parameters = model_config["parameters"]
        for key, value in parameters.items():
            parameters[key] = value["string_value"]
        pb_utils.Logger.log_info(f"parameters: {parameters}")

        # encoder config
        engine_dir = Path(parameters["engine_dir"])
        encoder_config_dict = read_config("encoder", engine_dir)

        # feature extractor
        mel_filters_dir = parameters["mel_filters_dir"]
        self.feature_extractor = FeatureExtractor(
            n_mels=int(parameters["n_mels"]),
            mel_filters_dir=mel_filters_dir,
        )

        # decoder(gpt-2) config
        decoder_config_path = engine_dir / "decoder" / "config.json"
        decoder_json_config = GptJsonConfig.parse_file(decoder_config_path)
        # pb_utils.Logger.log_info(f"Using decoder config: {decoder_json_config}")
        # https://github.com/NVIDIA/TensorRT-LLM/blob/a7c50cc426e1865afb0be0545a6035f7af420870/cpp/include/tensorrt_llm/runtime/modelConfig.h#L342
        assert (
            decoder_json_config.model_config.supports_inflight_batching
        ), f"{decoder_json_config.model_config.supports_inflight_batching}, expected True"

        self.decoder_model_config = decoder_json_config.model_config

        # encoder-decoder model engine runner
        runner_kwargs = dict(
            engine_dir=engine_dir,
            # https://github.com/NVIDIA/TensorRT-LLM/blob/v0.18.0/tensorrt_llm/runtime/model_runner_cpp.py#L199
            # seq2seq encoder-decoder transformer model support
            is_enc_dec=True,
            # https://github.com/NVIDIA/TensorRT-LLM/blob/v0.19.0rc0/cpp/tensorrt_llm/batch_manager/trtGptModelInflightBatching.cpp#L266
            # must set crossKvCacheFraction
            cross_kv_cache_fraction=float(parameters.get("cross_kv_cache_fraction", 0.5)),
            kv_cache_free_gpu_memory_fraction=float(
                parameters.get("kv_cache_free_gpu_mem_fraction", 0.5)
            ),
            debug_mode=False,
            ## default use engine config set in cpp runtime
            # max_batch_size=self.decoder_model_config.max_batch_size,
            # max_input_len=self.decoder_model_config.max_input_len,
            max_beam_width=self.decoder_model_config.max_beam_width,
        )
        pb_utils.Logger.log_info(f"runner_kwargs: {runner_kwargs}")
        # https://github.com/NVIDIA/TensorRT-LLM/blob/v0.18.0/tensorrt_llm/runtime/model_runner_cpp.py#L87
        self.model_runner_cpp = ModelRunnerCpp.from_dir(**runner_kwargs)

        # tokenizer
        self.tokenizer = get_tokenizer(
            num_languages=encoder_config_dict["num_languages"],
            tokenizer_dir=parameters["tokenizer_dir"],
        )
        self.blank = self.tokenizer.encode(
            " ",
            allowed_special=self.tokenizer.special_tokens_set,
        )[0]
        # https://huggingface.co/openai/whisper-large-v3/blob/main/added_tokens.json#L1521
        # <|endoftext|>
        self.eot_id = 50257
        self.zero_pad = parameters["zero_pad"] == "true"

        # output
        output0_config = pb_utils.get_output_config_by_name(model_config, "TRANSCRIPTS")
        self.out0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def process_batch(self, wav_batch, wav_len, prompt_id):
        # Convert numpy arrays to torch tensors
        wav_batch = torch.from_numpy(wav_batch).to(self.device)
        wav_len = wav_len.astype(np.int32)

        # Replicate prompt_id for batch size
        batch_size = wav_batch.shape[0]
        prompt_ids = np.tile(prompt_id, (batch_size, 1)).astype(np.int32)

        # Batch processing for each sample in the batch
        padding = 0 if self.zero_pad else self.decoder_model_config.max_input_len
        batch_mel_list = []
        for i in range(batch_size):
            wav_i = wav_batch[i : i + 1, : int(wav_len[i].item())]
            mel = self.feature_extractor.compute_feature(
                wav_i[0].to(self.device), padding_target_len=padding
            ).transpose(1, 2)
            batch_mel_list.append(mel.squeeze(0))

        # Move prompt IDs to GPU
        decoder_input_ids = torch.tensor(prompt_ids, dtype=torch.int32, device=self.device)

        # Calculate mel lengths
        mel_input_lengths = torch.tensor(
            [mel.shape[0] for mel in batch_mel_list], dtype=torch.int32, device=self.device
        )

        pb_utils.Logger.log_info(f"decoder_input_ids: {decoder_input_ids.shape}")
        pb_utils.Logger.log_info(f"mel_input_lengths: {mel_input_lengths.shape}")
        pb_utils.Logger.log_info(f"batch_mel_list: {len(batch_mel_list)}")
        for i, mel in enumerate(batch_mel_list):
            pb_utils.Logger.log_info(f"batch_mel_list[{i}]: {mel.shape}")

        # Run batch inference
        outputs = self.model_runner_cpp.generate(
            batch_input_ids=decoder_input_ids,
            encoder_input_features=batch_mel_list,
            encoder_output_lengths=mel_input_lengths // 2,
            max_new_tokens=96,
            end_id=self.eot_id,
            pad_id=self.eot_id,
            num_beams=1,
            output_sequence_lengths=True,
            return_dict=True,
        )
        torch.cuda.synchronize()
        # pb_utils.Logger.log_info(f"outputs: {outputs}")

        # Process outputs
        output_ids = outputs["output_ids"].cpu().numpy()

        return output_ids

    def execute(self, requests):
        responses = []

        for request in requests:
            # Get batch inputs
            text_prefix = pb_utils.get_input_tensor_by_name(request, "TEXT_PREFIX").as_numpy()
            wav_batch = pb_utils.get_input_tensor_by_name(request, "WAV").as_numpy()
            wav_len = pb_utils.get_input_tensor_by_name(request, "WAV_LEN").as_numpy()

            # Use the same text_prefix for all items in the request
            prefix = text_prefix[0][0].decode("utf-8")
            if prefix == "":
                prefix = "<|startoftranscript|><|ko|><|transcribe|><|notimestamps|>"
            prompt_id = self.tokenizer.encode(
                prefix, allowed_special=self.tokenizer.special_tokens_set
            )

            # Process the entire batch
            output_ids = self.process_batch(wav_batch, wav_len, prompt_id)

            # Decode outputs for each item in batch
            transcripts = []

            # Handle case where output_ids is 3-dimensional
            # ([batch_size, beam_size, seq_len])
            if len(output_ids.shape) == 3:
                output_ids = output_ids[:, 0, :]  # Remove beam_size dimension

            for output_id in output_ids:
                token_list = output_id.tolist()
                s = self.tokenizer.decode(token_list)
                s = re.sub(r"<\|.*?\|>", "", s)
                transcripts.append(s)

            # Create response tensor
            out0 = pb_utils.Tensor("TRANSCRIPTS", np.array(transcripts).astype(self.out0_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out0])
            responses.append(inference_response)

        return responses

    def finalize(self):
        print("Cleaning up...")
