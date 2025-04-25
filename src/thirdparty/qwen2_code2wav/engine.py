import glob
import logging
import os
from typing import List, Union, Tuple

import numpy as np
import torch

from . import Code2WavEngineConfig, Code2WavGenerationConfig

from ..model_loader.weight_utils import safetensors_weights_iterator
from .modeling_fast import Qwen2Code2wav


class Code2WavEngine:
    def __init__(self, **kwargs) -> None:
        self.args = Code2WavEngineConfig(**kwargs)
        model_path = self.args.model_path
        enable_torch_compile = self.args.enable_torch_compile
        enable_torch_compile_first_chunk = self.args.enable_torch_compile_first_chunk
        odeint_method = self.args.odeint_method
        odeint_method_relaxed = self.args.odeint_method_relaxed
        batched_chunk = self.args.batched_chunk
        frequency: str = self.args.frequency
        device: Union[int, str] = self.args.device
        code2wav_dynamic_batch: bool = self.args.code2wav_dynamic_batch  # todo batch chunk

        if isinstance(device, int):
            device = f"cuda:{device}"
        self.device = torch.device(device)

        logging.info(
            f"Code2WavEngine starting up on device {self.device}, with model {model_path}, method: {odeint_method}, relaxed: {odeint_method_relaxed}"
        )

        # load spk_dict ["Ethan", "Chelsie"]
        if os.path.exists(os.path.join(model_path, "spk_dict.pt")):
            self.code2wav_conds, self.code2wav_ref_mels = self.load_spk_dict(model_path)
        assert len(self.code2wav_conds) > 0 and len(self.code2wav_ref_mels) > 0, "No speakers found"
        if "default" not in self.code2wav_conds:
            self.code2wav_conds["default"] = self.code2wav_conds[
                list(self.code2wav_conds.keys())[0]
            ]
        if "default" not in self.code2wav_ref_mels:
            self.code2wav_ref_mels["default"] = self.code2wav_ref_mels[
                list(self.code2wav_ref_mels.keys())[0]
            ]

        self.frequency = frequency
        self.code2wav_steps: int = 10
        self.code2wav_bs_mel: int = 24 if frequency == "50hz" else 32
        self.factor: int = 2 if frequency == "50hz" else 4

        dit_model, bigvgan_model = self.load_code2wav(model_path)
        self.code2wav = Qwen2Code2wav(
            dit_ckpt=dit_model,
            bigvgan_ckpt=bigvgan_model,
            steps=self.code2wav_steps,
            bs_mel=self.code2wav_bs_mel,
            odeint_method=odeint_method,
            odeint_method_relaxed=odeint_method_relaxed,
            batched_chunk=batched_chunk,
            frequency=frequency,
            device=self.device,
            with_weight_norm=False,
        )
        self.torch_compile_model(enable_torch_compile, enable_torch_compile_first_chunk)

        self.code2wav_y_all = torch.randn(
            1, 32768, 80, device=self.device, dtype=list(self.code2wav_ref_mels.values())[0].dtype
        )

    def get_voice(self, voice_type: str = "default"):
        if voice_type not in self.code2wav_conds:
            logging.warning(f"voice type {voice_type} not found, using default")
            voice_type = "default"
        code2wav_cond = self.code2wav_conds[voice_type]
        code2wav_ref_mel = self.code2wav_ref_mels[voice_type]
        return code2wav_cond, code2wav_ref_mel

    def load_spk_dict(self, model_path):
        code2wav_conds, code2wav_ref_mels = {}, {}

        if not os.path.exists(os.path.join(model_path, "spk_dict.pt")):
            return code2wav_conds, code2wav_ref_mels

        for key, value in torch.load(os.path.join(model_path, "spk_dict.pt")).items():
            code2wav_conds[key] = value["cond"].to(self.device)
            code2wav_ref_mels[key] = value["ref_mel"].to(self.device)
        return code2wav_conds, code2wav_ref_mels

    def load_code2wav(self, model_path):
        dit_model, bigvgan_model = {}, {}
        safetensors = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        legacy_weights = False
        for key, value in safetensors_weights_iterator(safetensors, use_tqdm_on_load=True):
            legacy_weights = legacy_weights or "input_embed.spk_encoder.fc.conv.weight" in key
            if legacy_weights:
                break
        for key, value in safetensors_weights_iterator(safetensors, use_tqdm_on_load=True):
            if key.startswith("token2wav.code2wav_bigvgan_model."):
                if "generator" not in bigvgan_model:
                    bigvgan_model["generator"] = {}
                bigvgan_model["generator"][key.replace("token2wav.code2wav_bigvgan_model.", "")] = (
                    value
                )
            if key.startswith("token2wav.code2wav_dit_model."):
                key = key.replace("token2wav.code2wav_dit_model.", "transformer.")
                if key.startswith("transformer.input_embed.spk_encoder"):
                    if legacy_weights:
                        dit_model[key] = value
                    else:
                        dit_model[
                            key.replace(".bias", ".conv.bias").replace(".weight", ".conv.weight")
                        ] = value
                elif ".ff.ff.0.weight" in key or ".ff.ff.0.bias" in key:
                    dit_model[
                        key.replace(".ff.ff.0.weight", ".ff.ff.0.0.weight").replace(
                            ".ff.ff.0.bias", ".ff.ff.0.0.bias"
                        )
                    ] = value
                elif ".ff.ff.3.weight" in key or ".ff.ff.3.bias" in key:
                    dit_model[
                        key.replace(".ff.ff.3.weight", ".ff.ff.2.weight").replace(
                            ".ff.ff.3.bias", ".ff.ff.2.bias"
                        )
                    ] = value
                else:
                    dit_model[key] = value
        return dit_model, bigvgan_model

    def torch_compile_model(
        self,
        enable_torch_compile,
        enable_torch_compile_first_chunk,
    ):
        if not enable_torch_compile:
            return

        # compile the bigvgan
        self.code2wav.code2wav_bigvgan_model.vocoder.forward = torch.compile(
            self.code2wav.code2wav_bigvgan_model.vocoder.forward,
        )
        # compile the dit
        if hasattr(self.code2wav, "enable_torch_compile"):
            self.code2wav.enable_torch_compile(enable_torch_compile_first_chunk)

        logging.info("Code2Wav model torch compiled")

    @torch.inference_mode()
    def step_generate_waveform(
        self,
        code: List[int],
        prev_generated: Union[torch.Tensor, List[torch.Tensor]],
        progress: int,
        finished: bool = False,
        y_all: torch.Tensor = None,
        voice_type: str = "default",
        gen_args: Code2WavGenerationConfig = Code2WavGenerationConfig(),
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]:
        """
        Generate waveform from code list step by step.
        """
        cond, ref_mel = self.get_voice(voice_type)
        chunk_code_length = len(code) * self.factor - self.code2wav.future_cache_size
        if (
            chunk_code_length > 0 and chunk_code_length % self.code2wav.chunk_size == 0
        ) or finished:
            code = torch.tensor(code, dtype=torch.long, device=self.device).reshape(1, -1)
            if progress == 0 and finished:
                process_chunk = self.code2wav.process_little_chunk
            else:
                process_chunk = self.code2wav.process_chunk

            return process_chunk(
                cond,
                ref_mel,
                codec_all=code,
                y_all=self.code2wav_y_all if y_all is None else y_all,
                i=progress,
                steps=gen_args.num_steps,
                prev_generated=prev_generated,
                finished=finished,
                cfg_strength=gen_args.guidance_scale,
                sway_sampling_coef=gen_args.sway_coefficient,
            )
        else:
            return prev_generated, None
