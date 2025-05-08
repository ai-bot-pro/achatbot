# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# reference: https://github.com/NVIDIA/TensorRT-LLM/blob/v0.20.0rc0/examples/models/core/whisper/run.py

import json
from collections import OrderedDict
from pathlib import Path
import math

import numpy as np
import torch
import tensorrt_llm
import tensorrt_llm.logger as logger
from tensorrt_llm._utils import str_dtype_to_torch, str_dtype_to_trt, trt_dtype_to_torch
from tensorrt_llm.runtime import ModelConfig, SamplingConfig
from tensorrt_llm.runtime.session import Session, TensorInfo


class WhisperEncoding:
    def __init__(self, engine_dir):
        self.session = self.get_session(engine_dir)
        config_path = engine_dir / "encoder" / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        self.n_mels = config["pretrained_config"]["n_mels"]
        self.dtype = config["pretrained_config"]["dtype"]
        self.num_languages = config["pretrained_config"]["num_languages"]

    def get_session(self, engine_dir):
        serialize_path = engine_dir / "encoder" / "rank0.engine"
        with open(serialize_path, "rb") as f:
            session = Session.from_serialized_engine(f.read())
        return session

    def get_audio_features(self, mel, mel_input_lengths, encoder_downsampling_factor=2):
        if isinstance(mel, list):
            longest_mel = max([f.shape[-1] for f in mel])
            mel = [
                torch.nn.functional.pad(f, (0, longest_mel - f.shape[-1]), mode="constant")
                for f in mel
            ]
            mel = torch.cat(mel, dim=0).type(str_dtype_to_torch("float16")).contiguous()
        bsz, seq_len = mel.shape[0], mel.shape[2]
        position_ids = (
            torch.arange(
                math.ceil(seq_len / encoder_downsampling_factor),
                dtype=torch.int32,
                device=mel.device,
            )
            .expand(bsz, -1)
            .contiguous()
        )

        inputs = OrderedDict()
        inputs["input_features"] = mel
        inputs["input_lengths"] = mel_input_lengths
        inputs["position_ids"] = position_ids

        output_list = [
            TensorInfo("input_features", str_dtype_to_trt(self.dtype), mel.shape),
            TensorInfo("input_lengths", str_dtype_to_trt("int32"), mel_input_lengths.shape),
            TensorInfo("position_ids", str_dtype_to_trt("int32"), inputs["position_ids"].shape),
        ]

        output_info = self.session.infer_shapes(output_list)

        logger.debug(f"output info {output_info}")
        outputs = {
            t.name: torch.empty(tuple(t.shape), dtype=trt_dtype_to_torch(t.dtype), device="cuda")
            for t in output_info
        }
        stream = torch.cuda.current_stream()
        ok = self.session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)
        assert ok, "Engine execution failed"
        stream.synchronize()
        encoder_output = outputs["encoder_output"]
        encoder_output_lengths = mel_input_lengths // encoder_downsampling_factor
        return encoder_output, encoder_output_lengths


class WhisperDecoding:
    def __init__(self, engine_dir, runtime_mapping, debug_mode=False):
        self.decoder_config = self.get_config(engine_dir)
        self.decoder_generation_session = self.get_session(engine_dir, runtime_mapping, debug_mode)

    def get_config(self, engine_dir):
        config_path = engine_dir / "decoder" / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        decoder_config = OrderedDict()
        decoder_config.update(config["pretrained_config"])
        decoder_config.update(config["build_config"])
        return decoder_config

    def get_session(self, engine_dir, runtime_mapping, debug_mode=False):
        serialize_path = engine_dir / "decoder" / "rank0.engine"
        with open(serialize_path, "rb") as f:
            decoder_engine_buffer = f.read()

        decoder_model_config = ModelConfig(
            max_batch_size=self.decoder_config["max_batch_size"],
            max_beam_width=self.decoder_config["max_beam_width"],
            num_heads=self.decoder_config["num_attention_heads"],
            num_kv_heads=self.decoder_config["num_attention_heads"],
            hidden_size=self.decoder_config["hidden_size"],
            vocab_size=self.decoder_config["vocab_size"],
            cross_attention=True,
            num_layers=self.decoder_config["num_hidden_layers"],
            gpt_attention_plugin=self.decoder_config["plugin_config"]["gpt_attention_plugin"],
            remove_input_padding=self.decoder_config["plugin_config"]["remove_input_padding"],
            has_position_embedding=self.decoder_config["has_position_embedding"],
            has_token_type_embedding=False,
        )
        decoder_generation_session = tensorrt_llm.runtime.GenerationSession(
            decoder_model_config, decoder_engine_buffer, runtime_mapping, debug_mode=debug_mode
        )

        return decoder_generation_session

    def generate(
        self,
        decoder_input_ids,
        encoder_outputs,
        encoder_max_input_length,
        eot_id,
        max_new_tokens=40,
        num_beams=1,
        repetition_penalty=1.0,
    ):
        encoder_input_lengths = torch.tensor(
            [encoder_outputs.shape[1] for x in range(encoder_outputs.shape[0])],
            dtype=torch.int32,
            device="cuda",
        )
        decoder_input_lengths = torch.tensor(
            [decoder_input_ids.shape[-1] for _ in range(decoder_input_ids.shape[0])],
            dtype=torch.int32,
            device="cuda",
        )
        decoder_max_input_length = torch.max(decoder_input_lengths).item()

        cross_attention_mask = (
            torch.ones(
                [
                    decoder_input_ids.shape[0],
                    decoder_max_input_length + max_new_tokens,
                    encoder_max_input_length,
                ]
            )
            .int()
            .cuda()
        )

        # Add debug logging
        logger.debug(f"Repetition penalty in generate: {repetition_penalty}")
        logger.debug(f"Repetition penalty type: {type(repetition_penalty)}")

        # Ensure repetition_penalty is a float
        try:
            if isinstance(repetition_penalty, (list, np.ndarray)):
                repetition_penalty = float(repetition_penalty[0])
            else:
                repetition_penalty = float(repetition_penalty)
        except (ValueError, TypeError) as e:
            logger.error(f"Error converting repetition_penalty to float: {e}")
            repetition_penalty = 1.0  # Default value

        logger.debug(f"Repetition penalty after conversion: {repetition_penalty}")

        # generation config
        sampling_config = SamplingConfig(
            end_id=eot_id, pad_id=eot_id, num_beams=num_beams, repetition_penalty=repetition_penalty
        )

        # Add debug logging
        logger.debug(f"Sampling config: {sampling_config}")

        self.decoder_generation_session.setup(
            decoder_input_lengths.size(0),
            decoder_max_input_length,
            max_new_tokens,
            beam_width=num_beams,
            encoder_max_input_length=encoder_outputs.shape[1],
        )

        torch.cuda.synchronize()

        decoder_input_ids = decoder_input_ids.type(torch.int32).cuda()
        output_ids = self.decoder_generation_session.decode(
            decoder_input_ids,
            decoder_input_lengths,
            sampling_config,
            encoder_output=encoder_outputs,
            encoder_input_lengths=encoder_input_lengths,
            cross_attention_mask=cross_attention_mask,
        )
        torch.cuda.synchronize()

        # get the list of int from output_ids tensor
        output_ids = output_ids.cpu().numpy().tolist()
        return output_ids


class WhisperTRTLLM(object):
    def __init__(self, engine_dir):
        world_size = 1
        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
        torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)
        engine_dir = Path(engine_dir)

        self.encoder = WhisperEncoding(engine_dir)
        self.decoder = WhisperDecoding(engine_dir, runtime_mapping, debug_mode=False)

    def process_batch(
        self,
        mel,
        mel_input_lengths,
        decoder_input_ids,
        eot_id=50257,
        max_new_tokens=96,
        num_beams=1,
        repetition_penalty=1.0,
    ):
        # Add debug logging
        logger.debug(f"Repetition penalty in process_batch: {repetition_penalty}")
        logger.debug(f"Repetition penalty type in process_batch: {type(repetition_penalty)}")

        # Ensure repetition_penalty is a float
        try:
            if isinstance(repetition_penalty, (list, np.ndarray)):
                repetition_penalty = float(repetition_penalty[0])
            else:
                repetition_penalty = float(repetition_penalty)
        except (ValueError, TypeError) as e:
            logger.error(f"Error converting repetition_penalty to float in process_batch: {e}")
            repetition_penalty = 1.0  # Default value

        logger.debug(f"Repetition penalty after conversion in process_batch: {repetition_penalty}")

        # encoder_output = self.encoder.get_audio_features(mel)
        encoder_output, encoder_output_lengths = self.encoder.get_audio_features(
            mel, mel_input_lengths
        )
        logger.info(f"encoder_output: {encoder_output.shape}")
        logger.info(f"encoder_output_lengths: {encoder_output_lengths}")
        encoder_max_input_length = torch.max(encoder_output_lengths).item()
        output_ids = self.decoder.generate(
            decoder_input_ids,
            encoder_output,
            encoder_max_input_length,
            eot_id,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
        )
        return output_ids
