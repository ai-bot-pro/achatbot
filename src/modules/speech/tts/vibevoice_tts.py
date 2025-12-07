import logging
import os
import sys
import time
import copy
import threading
import asyncio
import traceback
from pathlib import Path
from queue import Empty, Queue
from typing import AsyncGenerator, Any, Callable, Dict, Iterator, Optional, Tuple, cast

import numpy as np
import torch
import torchaudio

try:
    cur_dir = os.path.dirname(__file__)
    if bool(os.getenv("ACHATBOT_PKG", "")):
        sys.path.insert(1, os.path.join(cur_dir, "../../../VibeVoice"))
    else:
        sys.path.insert(1, os.path.join(cur_dir, "../../../../deps/VibeVoice"))
    from deps.VibeVoice.vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference,
    )
    from deps.VibeVoice.vibevoice.processor.vibevoice_streaming_processor import (
        VibeVoiceStreamingProcessor,
    )
    from deps.VibeVoice.vibevoice.modular.streamer import AudioStreamer
except ModuleNotFoundError as e:
    logging.error(
        "In order to use zonos-tts use transformer, you need to `pip install achatbot[tts_vibevoice]`."
    )
    raise Exception(f"Missing module: {e}")

from src.common.utils.audio_utils import AUDIO_EXTENSIONS
from src.common.random import set_all_random_seed
from src.common.types import PYAUDIO_PAFLOAT32, PYAUDIO_PAINT16
from src.common.utils.helper import file_md5_hash, get_device, print_model_params
from src.common.interface import ITts
from src.common.session import Session
from src.types.speech.tts.vibevoice import VibeVoiceTTSArgs
from .base import BaseTTS

SAMPLE_RATE = 24000


class VibeVoiceStreamingTTS:
    def __init__(
        self,
        model_path: str,
        voices_dir: str,
        device: str = "cuda",
        inference_steps: int = 5,
    ) -> None:
        self.model_path = Path(model_path)
        self.voices_dir = Path(voices_dir)
        self.inference_steps = inference_steps
        self.sample_rate = SAMPLE_RATE

        self.processor: Optional[VibeVoiceStreamingProcessor] = None
        self.model: Optional[VibeVoiceStreamingForConditionalGenerationInference] = None
        self.voice_presets: Dict[str, Path] = {}
        self.default_voice_key: Optional[str] = None
        self._voice_cache: Dict[str, object] = {}

        if device == "mpx":
            logging.info("Note: device 'mpx' detected, treating it as 'mps'.")
            device = "mps"
        if device == "mps" and not torch.backends.mps.is_available():
            logging.warning("Warning: MPS not available. Falling back to CPU.")
            device = "cpu"
        self.device = device
        self._torch_device = torch.device(device)

    def load(self) -> None:
        logging.info(f"[startup] Loading processor from {self.model_path}")
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(str(self.model_path))

        # Decide dtype & attention
        if self.device == "mps":
            load_dtype = torch.float32
            device_map = None
            attn_impl_primary = "sdpa"
        elif self.device == "cuda":
            load_dtype = torch.bfloat16
            device_map = "cuda"
            attn_impl_primary = "flash_attention_2"
        else:
            load_dtype = torch.float32
            device_map = "cpu"
            attn_impl_primary = "sdpa"
        logging.info(
            f"Using device: {device_map}, torch_dtype: {load_dtype}, attn_implementation: {attn_impl_primary}"
        )
        # Load model
        try:
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                str(self.model_path),
                torch_dtype=load_dtype,
                device_map=device_map,
                attn_implementation=attn_impl_primary,
            )

            if self.device == "mps":
                self.model.to("mps")
        except Exception as e:
            if attn_impl_primary == "flash_attention_2":
                logging.warning(
                    "Error loading the model. Trying to use SDPA. However, note that only flash_attention_2 has been fully tested, and using SDPA may result in lower audio quality."
                )

                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    str(self.model_path),
                    torch_dtype=load_dtype,
                    device_map=self.device,
                    attn_implementation="sdpa",
                )
                logging.info("Load model with SDPA successfully ")
            else:
                raise e

        self.model.eval()
        print_model_params(self.model)

        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type="sde-dpmsolver++",
            beta_schedule="squaredcos_cap_v2",
        )
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)

        self.voice_presets = self._load_voice_presets()
        preset_name = os.environ.get("VOICE_PRESET")
        self.default_voice_key = self._determine_voice_key(preset_name)
        self._ensure_voice_cached(self.default_voice_key)

    @property
    def default_voice_name(self) -> str:
        return self.default_voice_key

    def _load_voice_presets(self) -> Dict[str, Path]:
        if not self.voices_dir.exists():
            raise RuntimeError(f"Voices directory not found: {self.voices_dir}")

        presets: Dict[str, Path] = {}
        for pt_path in self.voices_dir.glob("*.pt"):
            presets[pt_path.stem] = pt_path

        if not presets:
            raise RuntimeError(f"No voice preset (.pt) files found in {self.voices_dir}")

        logging.info(f"[startup] Found {len(presets)} voice presets")
        return dict(sorted(presets.items()))

    def _determine_voice_key(self, name: Optional[str]) -> str:
        """
        if not voice name, default use first voice preset
        """
        if name and name in self.voice_presets:
            return name

        default_key = "en-WHTest_man"
        if default_key in self.voice_presets:
            return default_key

        first_key = next(iter(self.voice_presets))
        logging.info(f"[startup] Using fallback voice preset: {first_key}")
        return first_key

    def _ensure_voice_cached(self, key: str) -> object:
        if key not in self.voice_presets:
            raise RuntimeError(f"Voice preset {key!r} not found")

        if key not in self._voice_cache:
            preset_path = self.voice_presets[key]
            logging.info(
                f"[startup] Loading voice preset {key} from {preset_path} Loading prefilled prompt from {preset_path}"
            )
            prefilled_outputs = torch.load(
                preset_path,
                map_location=self._torch_device,
                weights_only=False,
            )
            self._voice_cache[key] = prefilled_outputs

        return self._voice_cache[key]

    def _get_voice_resources(self, requested_key: Optional[str]) -> Tuple[str, object]:
        key = (
            requested_key
            if requested_key and requested_key in self.voice_presets
            else self.default_voice_key
        )
        if key is None:
            key = next(iter(self.voice_presets))
            self.default_voice_key = key

        prefilled_outputs = self._ensure_voice_cached(key)
        return key, prefilled_outputs

    def _prepare_inputs(self, text: str, prefilled_outputs: object):
        if not self.processor or not self.model:
            raise RuntimeError("StreamingTTSService not initialized")

        processor_kwargs = {
            "text": text.strip(),
            "cached_prompt": prefilled_outputs,
            "padding": True,
            "return_tensors": "pt",
            "return_attention_mask": True,
        }

        processed = self.processor.process_input_with_cached_prompt(**processor_kwargs)

        prepared = {
            key: value.to(self._torch_device) if hasattr(value, "to") else value
            for key, value in processed.items()
        }
        return prepared

    def _run_generation(
        self,
        inputs,
        audio_streamer: AudioStreamer,
        errors,
        cfg_scale: float,
        do_sample: bool,
        temperature: float,
        top_p: float,
        refresh_negative: bool,
        prefilled_outputs,
        stop_event: threading.Event,
    ) -> None:
        try:
            self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={
                    "do_sample": do_sample,
                    "temperature": temperature if do_sample else 1.0,
                    "top_p": top_p if do_sample else 1.0,
                },
                audio_streamer=audio_streamer,
                stop_check_fn=stop_event.is_set,
                verbose=True,
                refresh_negative=refresh_negative,
                all_prefilled_outputs=copy.deepcopy(prefilled_outputs),
            )
        except Exception as exc:  # pragma: no cover - diagnostic logging
            errors.append(exc)
            traceback.print_exc()
            audio_streamer.end()

    def stream(
        self,
        text: str,
        cfg_scale: float = 1.5,
        do_sample: bool = False,
        temperature: float = 0.9,
        top_p: float = 0.9,
        refresh_negative: bool = True,
        inference_steps: Optional[int] = None,
        voice_key: Optional[str] = None,
        log_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[np.ndarray]:
        if not text.strip():
            return
        text = text.replace("’", "'")
        selected_voice, prefilled_outputs = self._get_voice_resources(voice_key)

        def emit(event: str, **payload: Any) -> None:
            if log_callback:
                try:
                    log_callback(event, **payload)
                except Exception as exc:
                    logging.error(f"[log_callback] Error while emitting {event}: {exc}")

        steps_to_use = self.inference_steps
        if inference_steps is not None:
            try:
                parsed_steps = int(inference_steps)
                if parsed_steps > 0:
                    steps_to_use = parsed_steps
            except (TypeError, ValueError):
                pass
        if self.model:
            self.model.set_ddpm_inference_steps(num_steps=steps_to_use)
        self.inference_steps = steps_to_use

        inputs = self._prepare_inputs(text, prefilled_outputs)
        audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
        errors: list = []
        stop_signal = stop_event or threading.Event()

        thread = threading.Thread(
            target=self._run_generation,
            kwargs={
                "inputs": inputs,
                "audio_streamer": audio_streamer,
                "errors": errors,
                "cfg_scale": cfg_scale,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": top_p,
                "refresh_negative": refresh_negative,
                "prefilled_outputs": prefilled_outputs,
                "stop_event": stop_signal,
            },
            daemon=True,
        )
        thread.start()

        generated_samples = 0

        try:
            stream = audio_streamer.get_stream(0)
            for audio_chunk in stream:
                if torch.is_tensor(audio_chunk):
                    audio_chunk = audio_chunk.detach().cpu().to(torch.float32).numpy()
                else:
                    audio_chunk = np.asarray(audio_chunk, dtype=np.float32)

                if audio_chunk.ndim > 1:
                    audio_chunk = audio_chunk.reshape(-1)

                peak = np.max(np.abs(audio_chunk)) if audio_chunk.size else 0.0
                if peak > 1.0:
                    audio_chunk = audio_chunk / peak

                generated_samples += int(audio_chunk.size)
                emit(
                    "model_progress",
                    generated_sec=generated_samples / self.sample_rate,
                    chunk_sec=audio_chunk.size / self.sample_rate,
                )

                chunk_to_yield = audio_chunk.astype(np.float32, copy=False)

                yield chunk_to_yield
        finally:
            stop_signal.set()
            audio_streamer.end()
            thread.join()
            if errors:
                emit("generation_error", message=str(errors[0]))
                raise errors[0]

    def np_to_pcm16_bytes(self, chunk: np.ndarray) -> bytes:
        chunk = np.clip(chunk, -1.0, 1.0)
        pcm = (chunk * 32767.0).astype(np.int16)
        return pcm.tobytes()


class VibeVoiceTTS(BaseTTS, ITts):
    TAG = "tts_vibevoice"
    SAMPLE_RATE = 24000

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**VibeVoiceTTSArgs().__dict__, **kwargs}

    def __init__(
        self,
        event_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        **args,
    ) -> None:
        super().__init__()
        # event callback e.g.: metrics, progress, error
        self.event_cb = event_cb

        self.args = VibeVoiceTTSArgs(**args)
        self.args.device = self.args.device or get_device()
        logging.debug(f"args:{self.args}")

        self.tts = VibeVoiceStreamingTTS(
            model_path=self.args.model_path,
            voices_dir=self.args.speaker_embedding_pt_dir,
            device=self.args.device,
            inference_steps=self.args.inference_steps,
        )
        self.tts.load()
        self.voices = self.tts.voice_presets

        self.warm_up()

    def warm_up(self):
        if not self.args.warm_up_text:
            logging.warning(f"No warm_up_text to Warm Up")
            return

        logging.info(f"Warming up {self.__class__.__name__} device: {self.tts.device}")

        if "cuda" in str(self.tts.device):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()

        for step in range(self.args.warmup_steps):
            times = []
            start_time = time.perf_counter()
            for _ in self.streaming_tts(
                self.args.warm_up_text,
                cfg_scale=self.args.cfg_scale,
                voice_key=self.args.voice,
                do_sample=self.args.do_sample,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                refresh_negative=self.args.refresh_negative,
                inference_steps=self.args.inference_steps,
                stop_event=self.stop_signal,
                log_callback=self.event_cb,
            ):
                times.append(time.perf_counter() - start_time)
                start_time = time.perf_counter()
            logging.info(f"step {step} warnup TTFT(chunk) time: {times[0]} s")

        if "cuda" in str(self.tts.device):
            end_event.record()
            torch.cuda.synchronize()
            logging.info(
                f"{self.__class__.__name__}:  warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s"
            )
        logging.info(f"End Warm Up")

    def set_voice(self, voice_name: str):
        """
        - voice_name: voice name
        """
        raise NotImplementedError("set_voice is not supported by VibeVoiceTTS yet.")

    def get_voices(self) -> list:
        return list(self.voices.keys())

    def get_stream_info(self) -> dict:
        return {
            "format": PYAUDIO_PAINT16,
            "channels": 1,
            "rate": self.SAMPLE_RATE,
            "sample_width": 2,
            "np_dtype": np.int16,
        }

    def streaming_tts(self, text: str, **kwargs) -> Iterator[np.ndarray]:
        yield from self.tts.stream(text, **kwargs)

    async def _inference(
        self, session: Session, text: str, **kwargs
    ) -> AsyncGenerator[bytes, None]:
        if "cuda" in str(self.tts.device):
            torch.cuda.empty_cache()
        seed = kwargs.get("seed", self.args.seed)

        set_all_random_seed(seed)

        cfg_scale = kwargs.get("cfg_scale", self.args.cfg_scale)
        voice = kwargs.get("voice", self.args.voice)
        do_sample = kwargs.get("do_sample", self.args.do_sample)
        temperature = kwargs.get("temperature", self.args.temperature)
        top_p = kwargs.get("top_p", self.args.top_p)
        refresh_negative = kwargs.get("refresh_negative", self.args.refresh_negative)
        steps = kwargs.get("steps", self.args.inference_steps)

        iterator = self.streaming_tts(
            text,
            cfg_scale=cfg_scale,
            voice_key=voice,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            refresh_negative=refresh_negative,
            inference_steps=steps,
            stop_event=self.stop_signal,
            log_callback=self.event_cb,
        )
        sentinel = object()

        logging.debug(f"{text=} {kwargs=}, Starting streaming generation...")
        self.start_speak()
        while True:
            # use async thread to run streaming tts, no block, yield chunk, if no chunk, return sentinel
            chunk = await asyncio.to_thread(next, iterator, sentinel)
            if chunk is sentinel:
                break
            chunk = cast(np.ndarray, chunk)
            payload = self.tts.np_to_pcm16_bytes(chunk)
            yield payload
