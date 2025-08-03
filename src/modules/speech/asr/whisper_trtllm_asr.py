import logging
import asyncio
import base64
import os
import re
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import AsyncGenerator
from abc import ABC, abstractmethod
import threading

import numpy as np
import tiktoken
import torch

from torch.utils.data import DataLoader

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_torch, str_dtype_to_trt, trt_dtype_to_torch
from tensorrt_llm.bindings import GptJsonConfig, KVCacheType
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelConfig, SamplingConfig
from tensorrt_llm.runtime.session import Session as TrtLLMSession, TensorInfo

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp


from src.types.speech.language import WHISPER_LANGUAGES
from src.types.speech.asr.trtllm_whisper import WhisperTensorRTLLMASRArgs
from src.common.utils.audio_utils import bytes2NpArrayWith16
from src.common.session import Session
from src.modules.speech.asr.base import ASRBase
from src.modules.speech.help.whisper_utils import (
    log_mel_spectrogram,
    mel_filters,
)


class WhisperTensorrtLLMAsr(ASRBase):
    """
    - https://github.com/NVIDIA/TensorRT-LLM/blob/v0.20.0/examples/models/core/whisper/README.md
    - https://github.com/NVIDIA/TensorRT-LLM/blob/v0.20.0/examples/models/core/whisper/run.py

    NOTE:
    - TrtLLM unsupport timestamp: https://github.com/NVIDIA/TensorRT-LLM/issues/647
    - openai whisper timing DTW: https://github.com/openai/whisper/blob/main/whisper/timing.py#L141 (use Triton dtw_kernel with cuda)

    - if need timestamps, use torchaudio with Wav2Vec2 to align, e.g.: WhisperX use torchaudio with Wav2Vec2 to align
        - torchaudio with Wav2Vec2: https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html
        - whisperx with Wav2Vec2: https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py
    """

    TAG = "whisper_trtllm_asr"

    def __init__(self, **args) -> None:
        self.asr_audio = None
        self.lock = threading.Lock()
        self.args = WhisperTensorRTLLMASRArgs(**args)
        tensorrt_llm.logger.set_level(self.args.log_level)
        if self.args.use_py_session:
            self.model = WhisperTRTLLMPythonSession(
                self.args.engine_dir,
                self.args.assets_dir,
                debug_mode=self.args.debug,
            )
        else:
            self.model = WhisperTRTLLMModelRunnerCpp(
                self.args.engine_dir,
                self.args.assets_dir,
                max_input_len=self.args.max_input_len,
                max_output_len=self.args.max_output_len,
                max_batch_size=self.args.max_batch_size,
                num_beams=self.args.num_beams,
                kv_cache_free_gpu_memory_fraction=self.args.kv_cache_free_gpu_memory_fraction,
                cross_kv_cache_fraction=self.args.cross_kv_cache_fraction,
                debug_mode=self.args.debug,
            )

        logging.info(f"WhisperTensorrtLLMAsr initialized {self.model} with args: {self.args}")

    def set_audio_data(self, audio_data):
        self.lock.acquire()
        if isinstance(audio_data, (bytes, bytearray)):
            self.asr_audio = bytes2NpArrayWith16(audio_data)
        if (
            isinstance(audio_data, str)
            and audio_data.endswith(".wav")
            and os.path.exists(audio_data)
        ):
            with open(audio_data, "rb") as f:
                self.asr_audio = bytes2NpArrayWith16(f.read())
        self.lock.release()
        return

    async def transcribe_stream(self, session: Session) -> AsyncGenerator[str, None]:
        mel, duration = log_mel_spectrogram(
            self.asr_audio.copy(), filters=self.model.filters, device="cuda", return_duration=True
        )
        # now just generate full text, not stream, @todo
        prediction = await asyncio.to_thread(
            self.model.predict,
            mel,
            text_prefix=f"<|startoftranscript|><|{self.args.language}|><|transcribe|><|notimestamps|>",
            padding_strategy=self.args.padding_strategy,
        )
        yield prediction

    async def transcribe(self, session: Session) -> dict:
        mel, duration = log_mel_spectrogram(
            self.asr_audio.copy(), filters=self.model.filters, device="cuda", return_duration=True
        )
        prediction = await asyncio.to_thread(
            self.model.predict,
            mel,
            text_prefix=f"<|startoftranscript|><|{self.args.language}|><|transcribe|><|notimestamps|>",
            padding_strategy=self.args.padding_strategy,
        )

        res = {
            "language": self.args.language,
            "language_probability": None,
            "text": prediction.strip(),
            "words": [],
        }
        return res


def get_tokenizer(name: str = "multilingual", num_languages: int = 99, tokenizer_dir: str = None):
    if tokenizer_dir is None:
        vocab_path = os.path.join(os.path.dirname(__file__), f"assets/{name}.tiktoken")
    else:
        vocab_path = os.path.join(tokenizer_dir, f"{name}.tiktoken")
    ranks = {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in open(vocab_path) if line)
    }
    n_vocab = len(ranks)
    special_tokens = {}

    specials = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in list(WHISPER_LANGUAGES.keys())[:num_languages]],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
    ]

    for token in specials:
        special_tokens[token] = n_vocab
        n_vocab += 1

    return tiktoken.Encoding(
        name=os.path.basename(vocab_path),
        explicit_n_vocab=n_vocab,
        pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        mergeable_ranks=ranks,
        special_tokens=special_tokens,
    )


def remove_tensor_padding(input_tensor, input_tensor_lengths=None, pad_value=None):
    if pad_value:
        assert input_tensor_lengths is None, (
            "input_tensor_lengths should be None when pad_value is provided"
        )
        # Text tensor case: batch, seq_len
        assert torch.all(input_tensor[:, 0] != pad_value), (
            "First token in each sequence should not be pad_value"
        )
        assert input_tensor_lengths is None

        # Create a mask for all non-pad tokens
        mask = input_tensor != pad_value

        # Apply the mask to input_tensor to remove pad tokens
        output_tensor = input_tensor[mask].view(1, -1)

    else:
        # Audio tensor case: batch, seq_len, feature_len
        # position_ids case: batch, seq_len
        assert input_tensor_lengths is not None, (
            "input_tensor_lengths must be provided for 3D input_tensor"
        )

        # Initialize a list to collect valid sequences
        valid_sequences = []

        for i in range(input_tensor.shape[0]):
            valid_length = input_tensor_lengths[i]
            valid_sequences.append(input_tensor[i, :valid_length])

        # Concatenate all valid sequences along the batch dimension
        output_tensor = torch.cat(valid_sequences, dim=0)
    return output_tensor


def read_config(component, engine_dir):
    config_path = Path(engine_dir) / component / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    model_config = OrderedDict()
    model_config.update(config["pretrained_config"])
    model_config.update(config["build_config"])
    return model_config


class WhisperEncoding:
    def __init__(self, engine_dir):
        self.session = self.get_session(engine_dir)
        config = read_config("encoder", engine_dir)
        self.n_mels = config["n_mels"]
        self.dtype = config["dtype"]
        self.num_languages = config["num_languages"]
        self.encoder_config = config

    def get_session(self, engine_dir):
        serialize_path = Path(engine_dir) / "encoder" / "rank0.engine"
        with open(serialize_path, "rb") as f:
            session = TrtLLMSession.from_serialized_engine(f.read())
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
        if self.encoder_config["plugin_config"]["remove_input_padding"]:
            # mel B,D,T -> B,T,D -> BxT, D
            mel = mel.transpose(1, 2)
            mel = remove_tensor_padding(mel, mel_input_lengths)
            position_ids = remove_tensor_padding(
                position_ids, mel_input_lengths // encoder_downsampling_factor
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

        output_info = (self.session).infer_shapes(output_list)

        logging.info(f"output info {output_info}")
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
        self.decoder_config = read_config("decoder", engine_dir)
        self.decoder_generation_session = self.get_session(engine_dir, runtime_mapping, debug_mode)

    def get_session(self, engine_dir, runtime_mapping, debug_mode=False):
        serialize_path = Path(engine_dir) / "decoder" / "rank0.engine"
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
            kv_cache_type=KVCacheType.PAGED
            if self.decoder_config["plugin_config"]["paged_kv_cache"] is True
            else KVCacheType.CONTINUOUS,
            has_position_embedding=self.decoder_config["has_position_embedding"],
            dtype=self.decoder_config["dtype"],
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
        encoder_input_lengths,
        eot_id,
        max_new_tokens=40,
        num_beams=1,
    ):
        batch_size = decoder_input_ids.shape[0]
        decoder_input_lengths = torch.tensor(
            [decoder_input_ids.shape[-1] for _ in range(decoder_input_ids.shape[0])],
            dtype=torch.int32,
            device="cuda",
        )
        decoder_max_input_length = torch.max(decoder_input_lengths).item()

        cross_attention_mask = (
            torch.ones(
                [batch_size, decoder_max_input_length + max_new_tokens, encoder_max_input_length]
            )
            .int()
            .cuda()
        )
        # generation config
        # https://nvidia.github.io/TensorRT-LLM/python-api/tensorrt_llm.runtime.html#tensorrt_llm.runtime.SamplingConfig
        sampling_config = SamplingConfig(end_id=eot_id, pad_id=eot_id, num_beams=num_beams)
        self.decoder_generation_session.setup(
            decoder_input_lengths.size(0),
            decoder_max_input_length,
            max_new_tokens,
            beam_width=num_beams,
            encoder_max_input_length=encoder_max_input_length,
        )

        torch.cuda.synchronize()

        decoder_input_ids = decoder_input_ids.type(torch.int32).cuda()
        if self.decoder_config["plugin_config"]["remove_input_padding"]:
            # 50256 is the index of <pad> for all whisper models' decoder
            WHISPER_PAD_TOKEN_ID = 50256
            decoder_input_ids = remove_tensor_padding(
                decoder_input_ids, pad_value=WHISPER_PAD_TOKEN_ID
            )
            if encoder_outputs.dim() == 3:
                encoder_output_lens = torch.full(
                    (encoder_outputs.shape[0],),
                    encoder_outputs.shape[1],
                    dtype=torch.int32,
                    device="cuda",
                )

                encoder_outputs = remove_tensor_padding(encoder_outputs, encoder_output_lens)
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


class WhisperTRTLLM(ABC):
    def __init__(
        self,
        engine_dir: str,  # model engine directory
        assets_dir: str,  # tokinizer_dir/multilingual.tiktoken and mel_filters_dir/mel_filters.npz
        **kwargs,
    ):
        world_size = 1
        runtime_rank = tensorrt_llm.mpi_rank()
        self.runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
        torch.cuda.set_device(runtime_rank % self.runtime_mapping.gpus_per_node)
        engine_dir = Path(engine_dir)
        encoder_config = read_config("encoder", engine_dir)
        decoder_config = read_config("decoder", engine_dir)
        self.n_mels = encoder_config["n_mels"]
        self.num_languages = encoder_config["num_languages"]
        is_multilingual = decoder_config["vocab_size"] >= 51865
        if is_multilingual:
            tokenizer_name = "multilingual"
            assert (Path(assets_dir) / "multilingual.tiktoken").exists(), (
                "multilingual.tiktoken file is not existed in assets_dir"
            )
        else:
            tokenizer_name = "gpt2"
            assert (Path(assets_dir) / "gpt2.tiktoken").exists(), (
                "gpt2.tiktoken file is not existed in assets_dir"
            )
        self.tokenizer = get_tokenizer(
            name=tokenizer_name, num_languages=self.num_languages, tokenizer_dir=assets_dir
        )
        self.eot_id = self.tokenizer.encode(
            "<|endoftext|>", allowed_special=self.tokenizer.special_tokens_set
        )[0]
        self.filters = mel_filters("cuda", self.n_mels, assets_dir)

    def process_batch(
        self,
        mel,
        mel_input_lengths,
        text_prefix="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        num_beams=1,
        max_new_tokens=96,
    ):
        # 1. prepare decoder input ids (tokenizer encode prompt)
        prompt_id = self.tokenizer.encode(
            text_prefix, allowed_special=self.tokenizer.special_tokens_set
        )
        prompt_id = torch.tensor(prompt_id)
        batch_size = len(mel)
        decoder_input_ids = prompt_id.repeat(batch_size, 1)

        # 2. generate text from mel spectrograms and task prompt token ids
        output_ids = self.generate(
            mel,
            mel_input_lengths,
            decoder_input_ids,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )

        # 3. decode output_ids to text
        texts = []
        for i in range(len(output_ids)):
            text = self.tokenizer.decode(output_ids[i][0]).strip()
            texts.append(text)

        return texts

    @abstractmethod
    def generate(
        self,
        mel: torch.Tensor,
        mel_input_lengths: int,
        decoder_input_ids: torch.Tensor,
        **kwargs,
    ) -> list:
        """
        Generate text from mel spectrograms and task prompt token ids using the Whisper model.
        ---
        Returns:
        - output_ids: list of list of int, each list is the generated token ids for each batch
        """
        raise NotImplementedError("must be implemented in the child class")

    def predict(
        self,
        mel: torch.Tensor,
        text_prefix="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        dtype="float16",
        batch_size=1,
        num_beams=1,
        padding_strategy="max",
        max_new_tokens=96,
    ):
        mel = mel.type(str_dtype_to_torch(dtype))
        mel = mel.unsqueeze(0)
        # repeat the mel spectrogram to match the batch size
        mel = mel.repeat(batch_size, 1, 1)
        if padding_strategy == "longest":
            pass
        else:
            mel = torch.nn.functional.pad(mel, (0, 3000 - mel.shape[2]))
        features_input_lengths = torch.full(
            (mel.shape[0],), mel.shape[2], dtype=torch.int32, device=mel.device
        )

        predictions = self.process_batch(
            mel,
            features_input_lengths,
            text_prefix=text_prefix,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
        prediction = predictions[0]

        # remove all special tokens in the prediction
        prediction = re.sub(r"<\|.*?\|>", "", prediction)
        return prediction.strip()

    def decode_wav_file(
        self,
        input_file_path,
        text_prefix="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        dtype="float16",
        batch_size=1,
        num_beams=1,
        padding_strategy="longest",
        max_new_tokens=96,
    ):
        mel, total_duration = log_mel_spectrogram(
            input_file_path,
            self.filters,
            device="cuda",
            return_duration=True,
        )
        prediction = self.predict(
            mel, text_prefix, dtype, batch_size, num_beams, padding_strategy, max_new_tokens
        )

        logging.info(f"prediction: {prediction}")
        results = [(0, [""], prediction.split())]

        return results, total_duration

    def decode_dataset(
        self,
        dataset,
        text_prefix="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        dtype="float16",
        batch_size=1,
        num_beams=1,
        sample_rate=16000,
        compute_cer=False,
        padding_strategy="longest",
    ):
        def collate_wrapper(batch):
            speeches, durations, labels, ids = [], [], [], []
            for item in batch:
                speech = item["audio"]["array"]
                duration = speech.shape[-1]
                speech = speech.astype(np.float32)
                speech = torch.from_numpy(speech)
                speeches.append(speech)
                durations.append(duration)
                labels.append(item["text"])
                if "id" in item:
                    ids.append(item["id"])
                else:
                    ids.append(item["segment_id"])
            return speeches, durations, labels, ids

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_wrapper,
        )
        results = []
        total_duration = 0
        for batch in data_loader:
            waveforms, durations, texts, ids = batch
            total_duration += sum(durations) / sample_rate

            for wave in waveforms:
                assert wave.is_pinned()

            if padding_strategy == "longest":
                longest_duration = max(durations)
            elif padding_strategy == "nopad":
                longest_duration = 0
            else:
                longest_duration = int(16000 * 30)

            features = [
                log_mel_spectrogram(
                    wave,
                    self.filters,
                    padding=longest_duration - wave.shape[-1],
                    device="cuda",
                )
                .type(str_dtype_to_torch(dtype))
                .unsqueeze(0)
                for wave in waveforms
            ]

            # pad to the even number of features, for remove_padding option, conv layer padding corner case
            for i, feature in enumerate(features):
                if feature.shape[2] % 2:
                    features[i] = torch.nn.functional.pad(feature, (0, 1))

            features_input_lengths = torch.tensor(
                [f.shape[2] for f in features], dtype=torch.int32, device="cuda"
            )

            predictions = self.process_batch(
                features, features_input_lengths, text_prefix, num_beams
            )
            for wav_id, label, prediction in zip(ids, texts, predictions):
                # remove all special tokens in the prediction
                prediction = re.sub(r"<\|.*?\|>", "", prediction)
                label = label.split()
                prediction = prediction.split()
                if compute_cer:
                    label = list("".join(label))
                    prediction = list("".join(prediction))
                logging.info(f"wav_id: {wav_id}, label: {label}, prediction: {prediction}")
                results.append((wav_id, label, prediction))
        return results, total_duration


class WhisperTRTLLMModelRunnerCpp(WhisperTRTLLM):
    def __init__(
        self,
        engine_dir: str,  # model engine directory
        assets_dir: str,  # tokinizer_dir/multilingual.tiktoken and mel_filters_dir/mel_filters.npz
        max_input_len=3000,
        max_output_len=96,
        max_batch_size=1,
        num_beams=1,
        kv_cache_free_gpu_memory_fraction=0.9,
        cross_kv_cache_fraction=0.5,
        debug_mode=False,
    ):
        super().__init__(engine_dir, assets_dir)
        json_config = GptJsonConfig.parse_file(Path(engine_dir) / "decoder" / "config.json")
        assert json_config.model_config.supports_inflight_batching
        runner_kwargs = dict(
            engine_dir=engine_dir,
            is_enc_dec=True,
            max_batch_size=max_batch_size,
            max_input_len=max_input_len,
            max_output_len=max_output_len,
            max_beam_width=num_beams,
            kv_cache_free_gpu_memory_fraction=kv_cache_free_gpu_memory_fraction,
            cross_kv_cache_fraction=cross_kv_cache_fraction,
            debug_mode=debug_mode,
        )
        # https://nvidia.github.io/TensorRT-LLM/python-api/tensorrt_llm.runtime.html#tensorrt_llm.runtime.ModelRunner.from_dir
        self.model_runner_cpp = ModelRunnerCpp.from_dir(**runner_kwargs)

    @torch.no_grad()
    def generate(
        self,
        mel: torch.Tensor | list,
        mel_input_lengths: int,
        decoder_input_ids: torch.Tensor,
        encoder_downsampling_factor: int = 2,
        **kwargs,
    ) -> list:
        if isinstance(mel, list):
            mel = [m.transpose(1, 2).type(str_dtype_to_torch("float16")).squeeze(0) for m in mel]
        else:
            mel = mel.transpose(1, 2).type(str_dtype_to_torch("float16"))

        encoder_output_lengths = mel_input_lengths // encoder_downsampling_factor
        max_new_tokens = kwargs.get("max_new_tokens", 96)

        # https://nvidia.github.io/TensorRT-LLM/python-api/tensorrt_llm.runtime.html#tensorrt_llm.runtime.ModelRunner.generate
        outputs = self.model_runner_cpp.generate(
            batch_input_ids=decoder_input_ids,
            encoder_input_features=mel,
            encoder_output_lengths=encoder_output_lengths,
            max_new_tokens=max_new_tokens,
            end_id=self.eot_id,
            pad_id=self.eot_id,
            num_beams=kwargs.get("num_beams", 1),
            output_sequence_lengths=True,
            return_dict=True,
        )
        torch.cuda.synchronize()
        output_ids = outputs["output_ids"].cpu().numpy().tolist()

        return output_ids


class WhisperTRTLLMPythonSession(WhisperTRTLLM):
    def __init__(
        self,
        engine_dir: str,  # model engine directory
        assets_dir: str,  # tokinizer_dir/multilingual.tiktoken and mel_filters_dir/mel_filters.npz
        debug_mode=False,
    ):
        super().__init__(
            engine_dir,
            assets_dir,
        )
        self.encoder = WhisperEncoding(engine_dir)
        self.decoder = WhisperDecoding(engine_dir, self.runtime_mapping, debug_mode=debug_mode)

    @torch.no_grad()
    def generate(
        self,
        mel: torch.Tensor | list,
        mel_input_lengths: int,
        decoder_input_ids: torch.Tensor,
        **kwargs,
    ) -> list:
        # encoder get audio features
        encoder_output, encoder_output_lengths = self.encoder.get_audio_features(
            mel, mel_input_lengths
        )
        encoder_max_input_length = torch.max(encoder_output_lengths).item()

        # decoder generate text
        output_ids = self.decoder.generate(
            decoder_input_ids,
            encoder_output,
            encoder_max_input_length,
            encoder_output_lengths,
            self.eot_id,
            max_new_tokens=kwargs.get("max_new_tokens", 96),
            num_beams=kwargs.get("num_beams", 1),
        )

        return output_ids
