import logging
import os
import sys
import time
import asyncio
import tempfile
import urllib.request
from threading import Thread
import queue
from typing import AsyncGenerator, Any, Callable, Dict, Iterator, Optional, cast
from functools import partial

import numpy as np
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

try:
    cur_dir = os.path.dirname(__file__)
    if bool(os.getenv("ACHATBOT_PKG", "")):
        sys.path.insert(1, os.path.join(cur_dir, "../../../GLMTTS"))
    else:
        sys.path.insert(1, os.path.join(cur_dir, "../../../../deps/GLMTTS"))
    from deps.GLMTTS.cosyvoice.cli.frontend import TTSFrontEnd, SpeechTokenizer, TextFrontEnd
    from deps.GLMTTS.utils import file_utils, seed_util
    from deps.GLMTTS.utils import tts_model_util, yaml_util
    from deps.GLMTTS.llm.glmtts import GLMTTS
    from deps.GLMTTS.utils.audio import mel_spectrogram
except ModuleNotFoundError as e:
    logging.error(
        "In order to use glm-tts use transformer, you need to `pip install achatbot[tts_glm]`."
    )
    raise Exception(f"Missing module: {e}")

from src.common.random import set_all_random_seed
from src.common.types import PYAUDIO_PAFLOAT32, PYAUDIO_PAINT16
from src.common.utils.helper import get_device, print_model_params
from src.common.interface import ITts
from src.common.session import Session
from src.types.speech.tts.glm import GLMTTSTTSArgs
from .base import BaseTTS

SAMPLE_RATE = 24000
MAX_LLM_SEQ_INP_LEN = 750


def _assert_shape_and_get_len(token):
    assert token.ndim == 2 and token.shape[0] == 1
    token_len = torch.tensor([token.shape[1]], dtype=torch.int32).to(token.device)
    return token_len


class GLMTTSStreaming:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        inference_steps: int = 10,
        use_phoneme: bool = False,
        sample_rate: int = 24000,
    ):
        self.model_path = model_path
        self.device = device
        self.inference_steps = inference_steps
        self.frontend, self.text_frontend, self.speech_tokenizer, self.llm, self.token2wav = (
            self.load_models(use_phoneme=False, sample_rate=24000)
        )

        # prompt voices
        self.voices: dict[str, dict] = {}

    def load_frontends(
        self, speech_tokenizer, sample_rate=24000, use_phoneme=False, frontend_dir="frontend"
    ):
        if sample_rate == 32000:
            feat_extractor = partial(
                mel_spectrogram,
                sampling_rate=sample_rate,
                hop_size=640,
                n_fft=2560,
                num_mels=80,
                win_size=2560,
                fmin=0,
                fmax=8000,
                center=False,
            )
            print("Configured for 32kHz frontend.")
        elif sample_rate == 24000:
            feat_extractor = partial(
                mel_spectrogram,
                sampling_rate=sample_rate,
                hop_size=480,
                n_fft=1920,
                num_mels=80,
                win_size=1920,
                fmin=0,
                fmax=8000,
                center=False,
            )
            print("Configured for 24kHz frontend.")
        else:
            raise ValueError(f"Unsupported sampling_rate: {sample_rate}")

        glm_tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(self.model_path, "vq32k-phoneme-tokenizer"), trust_remote_code=True
        )

        def tokenize_fn(text):
            return glm_tokenizer.encode(text)

        frontend = TTSFrontEnd(
            tokenize_fn,
            speech_tokenizer,
            feat_extractor,
            os.path.join(frontend_dir, "campplus.onnx"),
            os.path.join(frontend_dir, "spk2info.pt"),
            self.device,
        )
        text_frontend = TextFrontEnd(use_phoneme)
        return frontend, text_frontend

    def get_special_token_ids(self, tokenize_fn):
        """
        Get special token IDs based on the tokenizer name.
        """
        _special_token_ids = {
            "ats": "<|audio_0|>",
            "ate": "<|audio_32767|>",
            "boa": "<|begin_of_audio|>",
            "eoa": "<|user|>",
            "pad": "<|endoftext|>",
        }

        special_token_ids = {}

        # Validation
        endoftext_id = tokenize_fn("<|endoftext|>")[0]
        for k, v in _special_token_ids.items():
            __ids = tokenize_fn(v)
            # Check 1: Special token length must be 1
            if len(__ids) != 1:
                raise AssertionError(f"Token '{k}' ({v}) encoded to multiple tokens: {__ids}")
            # Check 2: Special token ID must be >= endoftext_id
            if __ids[0] < endoftext_id:
                raise AssertionError(
                    f"Token '{k}' ({v}) ID {__ids[0]} is smaller than endoftext ID {endoftext_id}"
                )

            special_token_ids[k] = __ids[0]

        print(f"{special_token_ids=}")

        return special_token_ids

    def load_models(self, use_phoneme=False, sample_rate=24000):
        # Load Speech Tokenizer
        speech_tokenizer_path = os.path.join(self.model_path, "speech_tokenizer")
        _model, _feature_extractor = yaml_util.load_speech_tokenizer(speech_tokenizer_path)
        speech_tokenizer = SpeechTokenizer(_model, _feature_extractor)

        # Load Frontends
        frontend, text_frontend = self.load_frontends(
            speech_tokenizer,
            sample_rate=sample_rate,
            use_phoneme=use_phoneme,
            frontend_dir=os.path.join(self.model_path, "frontend"),
        )

        llama_path = os.path.join(self.model_path, "llm")

        llm = GLMTTS(
            llama_cfg_path=os.path.join(llama_path, "config.json"),
            mode="PRETRAIN",
            lora_adapter_config=os.path.join(
                self.model_path, "configs", "lora_adapter_configV3.1.json"
            ),
            spk_prompt_dict_path=os.path.join(self.model_path, "configs", "spk_prompt_dict.yaml"),
        )
        llm.llama = LlamaForCausalLM.from_pretrained(llama_path, dtype=torch.float32).to(
            self.device
        )

        llm.llama_embedding = llm.llama.model.embed_tokens

        special_token_ids = self.get_special_token_ids(frontend.tokenize_fn)
        print(f"special_token_ids: {special_token_ids}")
        llm.set_runtime_vars(special_token_ids=special_token_ids)

        flow_ckpt = os.path.join(self.model_path, "flow", "flow.pt")
        flow_config = os.path.join(self.model_path, "flow", "config.yaml")
        flow = yaml_util.load_flow_model(flow_ckpt, flow_config, self.device)

        ckpt_path = os.path.join(self.model_path, "hift", "hift.pt")
        if sample_rate == 32000:
            ckpt_path = os.path.join(self.model_path, "vocos2d", "generator_jit.ckpt")
        token2wav = tts_model_util.Token2Wav(
            flow,
            sample_rate=sample_rate,
            device=self.device,
            ckpt_path=ckpt_path,
        )

        return frontend, text_frontend, speech_tokenizer, llm, token2wav

    # --- Helper Function: Get Prompt from Cache ---
    def get_cached_prompt(self, cache, synth_text_token):
        """
        Constructs prompt tokens from the cache.
        Prunes the cache if the sequence length exceeds MAX_LLM_SEQ_INP_LEN.
        """
        cache_text = cache["cache_text"]
        cache_text_token = cache["cache_text_token"]
        cache_speech_token = cache["cache_speech_token"]

        def __len_cache_text_token():
            return sum(map(lambda x: x.shape[1], cache_text_token))

        def __len_cache_speech_token():
            return sum(map(len, cache_speech_token))

        # Estimate required length ratio
        # Avoid division by zero
        text_len = __len_cache_text_token()
        ta_ratio = __len_cache_speech_token() / (text_len if text_len > 0 else 1.0)

        __len_synth_text_token = synth_text_token.shape[1]
        __len_synth_audi_token_estim = int(ta_ratio * __len_synth_text_token)

        # Prune cache if too long.
        # Logic: Keep the first item (original prompt), remove from the second item onwards.
        while __len_cache_speech_token() + __len_synth_audi_token_estim > MAX_LLM_SEQ_INP_LEN:
            if len(cache_speech_token) <= 1:
                break  # Always keep at least the original prompt
            # logging.debug(f'[get_cached_prompt] Cache pop. Text count before: {len(cache_text)}')
            cache_text.pop(1)
            cache_text_token.pop(1)
            cache_speech_token.pop(1)

        # Construct Text Prompt
        prompt_text_token_from_cache = []
        for a_token in cache_text_token:
            prompt_text_token_from_cache.extend(a_token.squeeze().tolist())

        prompt_text_token = torch.tensor([prompt_text_token_from_cache]).to(self.device)

        # Construct Speech Prompt
        speech_tokens = []
        for a_cache_speech_token in cache_speech_token:
            speech_tokens.extend(a_cache_speech_token)

        llm_speech_token = torch.tensor([speech_tokens], dtype=torch.int32).to(self.device)

        return prompt_text_token, llm_speech_token

    def local_llm_forward_stream_generator(
        self,
        prompt_text_token,
        tts_text_token,
        prompt_speech_token,
        beam_size=1,
        sampling=25,
        sample_method="ras",
    ):
        token_queue = queue.Queue()
        prompt_text_token_len = _assert_shape_and_get_len(prompt_text_token)
        tts_text_token_len = _assert_shape_and_get_len(tts_text_token)
        prompt_speech_token_len = _assert_shape_and_get_len(prompt_speech_token)

        def llm_gen():
            try:
                self.llm.inference(
                    text=tts_text_token,
                    text_len=tts_text_token_len,
                    prompt_text=prompt_text_token,
                    prompt_text_len=prompt_text_token_len,
                    prompt_speech_token=prompt_speech_token,
                    prompt_speech_token_len=prompt_speech_token_len,
                    beam_size=beam_size,
                    sampling=sampling,
                    sample_method=sample_method,
                    spk=None,
                    queue=token_queue,
                )
            except Exception as e:
                print(f"Error in LLM inference: {e}")
                token_queue.put(None)

        thread = Thread(target=llm_gen)
        thread.start()

        while True:
            token_ids = token_queue.get()
            if token_ids is None:
                break
            print(f"{token_ids=}")
            yield token_ids

        thread.join()

    def local_flow_forward_stream_generator(
        self,
        token_list,
        prompt_speech_tokens,
        speech_feat,
        embedding,
        block_sizes,
        n_timesteps=10,
    ):
        wav_queue = queue.Queue()

        def stream_gen():
            try:
                self.token2wav.token2wav_stream(
                    token_list,
                    block_sizes=block_sizes,
                    prompt_token_list=prompt_speech_tokens,
                    prompt_feat_td=speech_feat,
                    embedding=embedding,
                    n_timesteps=n_timesteps,
                    queue=wav_queue,
                )
            except Exception as e:
                print(f"Error in streaming flow forward: {e}")
                wav_queue.put(None)

        thread = Thread(target=stream_gen)
        thread.start()

        while True:
            wav_np = wav_queue.get()
            if wav_np is None:
                break
            print(f"{wav_np.shape=}")
            yield wav_np

        thread.join()

    def stream(
        self,
        syn_text: str,
        cache: dict,
        embedding: torch.Tensor,
        seed: int = 0,
        sample_method: str = "ras",
        flow_prompt_token=None,
        speech_feat=None,
        use_phoneme: bool = False,
        n_timesteps: int = 10,
    ):
        text_tn_dict = {
            "syn_text": syn_text,
            "syn_text_tn": [],
            "syn_text_phoneme": [],
        }
        short_text_list = self.text_frontend.split_by_len(syn_text)

        for _, tts_text in enumerate(short_text_list):
            seed_util.set_seed(seed)
            tts_text_tn = self.text_frontend.text_normalize(
                tts_text
            )  # Normalize again after splitting
            text_tn_dict["syn_text_tn"].append(tts_text_tn)
            if use_phoneme:
                tts_text_tn = self.text_frontend.g2p_infer(tts_text_tn)
                text_tn_dict["syn_text_phoneme"].append(tts_text_tn)
            tts_text_token = self.frontend._extract_text_token(tts_text_tn)

            # Access cache references
            cache_text = cache["cache_text"]
            cache_text_token = cache["cache_text_token"]
            cache_speech_token = cache["cache_speech_token"]

            # Determine Prompts
            if cache["use_cache"] and len(cache_text_token) > 1:
                prompt_text_token, prompt_speech_token = self.get_cached_prompt(
                    cache, tts_text_token, self.device
                )
            else:
                # Initial prompt case
                prompt_text_token = cache_text_token[0].to(self.device)
                prompt_speech_token = torch.tensor([cache_speech_token[0]], dtype=torch.int32).to(
                    self.device
                )
                print("[generate_long] Using initial prompt (empty cache history)")

            # LLM Inference
            token_generator = self.local_llm_forward_stream_generator(
                prompt_text_token=prompt_text_token,
                tts_text_token=tts_text_token,
                prompt_speech_token=prompt_speech_token,
                sample_method=sample_method,
            )

            block_sizes = [25, 50, 200]
            all_tokens = []
            start_idx = 0
            block_idx = 0

            for token_id in token_generator:
                # normalize incoming token chunk to list and append to buffers
                all_tokens.append(token_id)

                # emit flows whenever we have enough tokens for the current block size
                while True:
                    cur_block = (
                        block_sizes[block_idx] if block_idx < len(block_sizes) else block_sizes[-1]
                    )
                    available = len(all_tokens) - start_idx
                    if available >= cur_block:
                        token_list = all_tokens[start_idx : start_idx + cur_block]
                        start_idx += cur_block
                        block_idx += 1

                        # Flow Inference for this block
                        wav_np_generator = self.local_flow_forward_stream_generator(
                            token_list=token_list,
                            prompt_speech_tokens=flow_prompt_token,
                            speech_feat=speech_feat,
                            embedding=embedding,
                            block_sizes=block_sizes,
                            n_timesteps=n_timesteps,
                        )
                        for wav_np in wav_np_generator:
                            yield wav_np
                        # loop to check if more full blocks are available now
                        continue
                    break

            # after token generator ends, if there are leftover tokens, process them as final block
            if len(all_tokens) - start_idx > 0:
                token_list = all_tokens[start_idx:]
                wav_np_generator = self.local_flow_forward_stream_generator(
                    token_list=token_list,
                    prompt_speech_tokens=flow_prompt_token,
                    speech_feat=speech_feat,
                    embedding=embedding,
                    block_sizes=block_sizes,
                    n_timesteps=n_timesteps,
                )
                for wav_np in wav_np_generator:
                    yield wav_np

            # Update Cache
            if cache is not None:
                cache_text.append(tts_text_tn)
                cache_text_token.append(tts_text_token)
                cache_speech_token.append(all_tokens)

    def set_prompt_ref_voice(self, name: str, prompt_text: str, prompt_speech_path: str):
        assert len(name) > 0 and len(prompt_text) > 0 and len(prompt_speech_path) > 0

        if not prompt_speech_path.startswith("http"):
            prompt_speech_path = os.path.join(self.model_path, "prompts", prompt_speech_path)
            assert os.path.exists(prompt_speech_path)
        else:
            # 从 URL 下载到本地临时文件
            url = prompt_speech_path
            # 从 URL 中提取文件扩展名
            suffix = os.path.splitext(url.split("?")[0])[-1] or ".wav"
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            try:
                logging.info(f"Downloading prompt speech from {url} to {temp_file.name}")
                urllib.request.urlretrieve(url, temp_file.name)
                prompt_speech_path = temp_file.name
            except Exception as e:
                temp_file.close()
                os.unlink(temp_file.name)
                raise RuntimeError(f"Failed to download prompt speech from {url}: {e}")
            finally:
                temp_file.close()

        # NT
        prompt_text = self.text_frontend.text_normalize(prompt_text)
        # text tokenizer
        prompt_text_token = self.frontend._extract_text_token(prompt_text + " ")
        # speech tokenizer
        prompt_speech_token = self.frontend._extract_speech_token(prompt_speech_path)
        # speech features extractor
        prompt_speech_feat = self.frontend._extract_speech_feat(
            prompt_speech_path, sample_rate=SAMPLE_RATE
        )
        # speech embedding
        prompt_speech_embedding = self.frontend._extract_spk_embedding(prompt_speech_path)

        self.voices[name] = {
            "prompt_text": prompt_text,
            "prompt_text_token": prompt_text_token,
            "prompt_speech_token": prompt_speech_token,
            "prompt_speech_feat": prompt_speech_feat,
            "prompt_speech_embedding": prompt_speech_embedding,
        }

    def get_prompt_ref_voice_names(self):
        return list(self.voices.keys())

    def get_prompt_ref_voice(self, name):
        return self.voices.get(name)

    def np_to_pcm16_bytes(self, chunk: np.ndarray) -> bytes:
        # chunk = chunk.astype(np.float32, copy=False)
        chunk = np.clip(chunk, -1.0, 1.0)
        pcm = (chunk * 32767.0).astype(np.int16)
        return pcm.tobytes()


class GLMTTSTTS(BaseTTS, ITts):
    TAG = "tts_glm"

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**GLMTTSTTSArgs().__dict__, **kwargs}

    def __init__(
        self,
        event_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        **args,
    ) -> None:
        super().__init__()
        # event callback e.g.: metrics, progress, error
        self.event_cb = event_cb

        self.args = GLMTTSTTSArgs(**args)
        self.args.device = self.args.device or get_device()
        logging.debug(f"args:{self.args}")

        self.tts = GLMTTSStreaming(
            model_path=self.args.model_path,
            device=self.args.device,
            inference_steps=self.args.inference_steps,
        )
        self.tts.set_prompt_ref_voice(
            self.args.default_voice_name,
            self.args.default_prompt_text,
            self.args.default_prompt_speech_path,
        )

        default_voice = self.tts.get_prompt_ref_voice(self.args.default_voice_name)
        assert default_voice is not None
        prompt_speech_token = default_voice.get("prompt_speech_token")
        prompt_text = default_voice.get("prompt_text")
        prompt_text_token = default_voice.get("prompt_text_token")
        embedding = default_voice.get("prompt_speech_embedding")
        speech_feat = default_voice.get("prompt_speech_feat")
        assert (
            prompt_speech_token is not None
            and prompt_text is not None
            and prompt_text_token is not None
            and embedding is not None
            and speech_feat is not None
        )
        self.voice = default_voice

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

        # Text Normalization
        synth_text = self.tts.text_frontend.text_normalize(self.args.warm_up_text)

        # Initialize Cache
        cache = {
            "cache_text": [self.voice.get("prompt_text")],
            "cache_text_token": [self.voice.get("prompt_text_token")],
            "cache_speech_token": [self.voice.get("prompt_speech_token").squeeze().tolist()],
            "use_cache": True,
        }

        for step in range(self.args.warmup_steps):
            times = []
            start_time = time.perf_counter()
            for _ in self.streaming_tts(
                syn_text=synth_text,
                cache=cache,
                embedding=self.voice.get("prompt_speech_embedding"),
                seed=self.args.seed,
                sample_method=self.args.sample_method,
                flow_prompt_token=torch.tensor(
                    self.voice.get("prompt_speech_token").squeeze().tolist(), dtype=torch.int32
                ).to(self.args.device),
                speech_feat=self.voice.get("prompt_speech_feat"),
                use_phoneme=self.args.use_phoneme,
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

    def set_voice(self, voice: str, **kwargs):
        """
        - voice_name: voice name
        - prompt_text: prompt text
        - prompt_speech_path: prompt speech path
        """
        voice_name = voice or kwargs.get("voice_name")
        prompt_text = kwargs.get("prompt_text")
        prompt_speech_path = kwargs.get("prompt_speech_path")
        self.tts.set_prompt_ref_voice(voice_name, prompt_text, prompt_speech_path)
        self.voice = self.tts.get_prompt_ref_voice(voice_name)
        assert self.voice is not None

    def get_voices(self) -> list:
        return self.tts.get_prompt_ref_voice_names()

    def get_stream_info(self) -> dict:
        return {
            "format": PYAUDIO_PAINT16,
            "channels": 1,
            "rate": SAMPLE_RATE,
            "sample_width": 2,
            "np_dtype": np.int16,
        }

    def streaming_tts(
        self, syn_text: str, cache: dict, embedding: torch.Tensor, **kwargs
    ) -> Iterator[np.ndarray]:
        yield from self.tts.stream(syn_text, cache, embedding, **kwargs)

    async def _inference(
        self, session: Session, text: str, **kwargs
    ) -> AsyncGenerator[bytes, None]:
        if "cuda" in str(self.tts.device):
            torch.cuda.empty_cache()
        seed = kwargs.get("seed", self.args.seed)

        set_all_random_seed(seed)

        # Text Normalization
        synth_text = self.tts.text_frontend.text_normalize(text)

        # Initialize Cache
        cache = {
            "cache_text": [self.voice.get("prompt_text")],
            "cache_text_token": [self.voice.get("prompt_text_token")],
            "cache_speech_token": [self.voice.get("prompt_speech_token").squeeze().tolist()],
            "use_cache": True,
        }

        iterator = self.streaming_tts(
            syn_text=synth_text,
            cache=cache,
            embedding=self.voice.get("prompt_speech_embedding"),
            seed=seed,
            sample_method=kwargs.get("sample_method", self.args.sample_method),
            flow_prompt_token=torch.tensor(
                self.voice.get("prompt_speech_token").squeeze().tolist(), dtype=torch.int32
            ).to(self.args.device),
            speech_feat=self.voice.get("prompt_speech_feat"),
            use_phoneme=kwargs.get("use_phoneme", self.args.use_phoneme),
            n_timesteps=kwargs.get("steps", self.args.inference_steps),
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
