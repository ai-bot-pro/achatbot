import os
import sys
import json
import asyncio
import subprocess
from threading import Thread
import queue


import modal


app = modal.App("step-audio")
IMAGE_GPU = os.getenv("IMAGE_GPU", None)
img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvcr.io/nvidia/cuda:13.0.2-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs")
    .run_commands(
        "git clone https://github.com/weedge/GLM-TTS.git -b main"
        " && cd /GLM-TTS"
        " && git checkout 24fffb4f74a2d9dd74001857e81549a6b0781672",
        "cd /GLM-TTS && pip install -r requirements.txt",
    )
    .pip_install("ffmpeg", "soxr")
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "ACHATBOT_PKG": "1",
            "LLM_MODEL": os.getenv("LLM_MODEL", "zai-org/GLM-TTS"),
        }
    )
)
img = img.run_commands(
    "cd /GLM-TTS && git pull origin main && git checkout 1284957d0b68d4864afa8ac594b5f2ed8fb668ed",
)

# img = img.pip_install(
#    f"achatbot==0.0.25.dev122",
#    extra_index_url=os.getenv("EXTRA_INDEX_URL", "https://test.pypi.org/simple/"),
# )


HF_MODEL_DIR = "/root/.achatbot/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/.achatbot/assets"
assets_vol = modal.Volume.from_name("assets", create_if_missing=True)
CONFIG_DIR = "/root/.achatbot/config"
config_vol = modal.Volume.from_name("config", create_if_missing=True)
RECORDS_DIR = "/root/.achatbot/records"
records_vol = modal.Volume.from_name("records", create_if_missing=True)

TORCH_CACHE_DIR = "/root/.cache/torch"
torch_cache_vol = modal.Volume.from_name("torch_cache", create_if_missing=True)

# gen audio output dir
GEN_AUDIO_DIR = "/root/gen_audio"
gen_audio_vol = modal.Volume.from_name("gen_audio", create_if_missing=True)


with img.imports():
    from functools import partial

    import torch
    from transformers import AutoTokenizer, LlamaForCausalLM

    sys.path.insert(1, "/GLM-TTS")

    from cosyvoice.cli.frontend import TTSFrontEnd, SpeechTokenizer, TextFrontEnd
    from utils import file_utils, seed_util
    from utils import tts_model_util, yaml_util
    from llm.glmtts import GLMTTS
    from utils.audio import mel_spectrogram

    MODEL_ID = os.getenv("LLM_MODEL", "zai-org/GLM-TTS")
    MODEL_PATH = os.path.join(HF_MODEL_DIR, MODEL_ID)
    os.makedirs(f"{ASSETS_DIR}/GLM-TTS", exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_LLM_SEQ_INP_LEN = 750

    # torch.set_float32_matmul_precision("high")


def print_model_params(model: torch.nn.Module, extra_info="", f=None):
    # print the number of parameters in the model
    model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(model, file=f)
    print(f"{extra_info} {model_million_params} M parameters", file=f)


@app.function(
    gpu=IMAGE_GPU,
    cpu=2.0,
    retries=0,
    image=img,
    secrets=[modal.Secret.from_name("achatbot")],
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_vol,
        GEN_AUDIO_DIR: gen_audio_vol,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=1,
)
async def run(func, **kwargs):
    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    gpu_prop = None
    if torch.cuda.is_available():
        gpu_prop = torch.cuda.get_device_properties("cuda")
        print(gpu_prop)

    if asyncio.iscoroutinefunction(func):
        await func(gpu_prop, **kwargs)
    else:
        func(gpu_prop, **kwargs)


def load_frontends(speech_tokenizer, sample_rate=24000, use_phoneme=False, frontend_dir="frontend"):
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
        os.path.join(MODEL_PATH, "vq32k-phoneme-tokenizer"), trust_remote_code=True
    )

    def tokenize_fn(text):
        return glm_tokenizer.encode(text)

    frontend = TTSFrontEnd(
        tokenize_fn,
        speech_tokenizer,
        feat_extractor,
        os.path.join(frontend_dir, "campplus.onnx"),
        os.path.join(frontend_dir, "spk2info.pt"),
        DEVICE,
    )
    text_frontend = TextFrontEnd(use_phoneme)
    return frontend, text_frontend


def get_special_token_ids(tokenize_fn):
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


def load_models(use_phoneme=False, sample_rate=24000):
    # Load Speech Tokenizer
    speech_tokenizer_path = os.path.join(MODEL_PATH, "speech_tokenizer")
    _model, _feature_extractor = yaml_util.load_speech_tokenizer(speech_tokenizer_path)
    speech_tokenizer = SpeechTokenizer(_model, _feature_extractor)

    # Load Frontends
    frontend, text_frontend = load_frontends(
        speech_tokenizer,
        sample_rate=sample_rate,
        use_phoneme=use_phoneme,
        frontend_dir=os.path.join(MODEL_PATH, "frontend"),
    )

    llama_path = os.path.join(MODEL_PATH, "llm")

    llm = GLMTTS(
        llama_cfg_path=os.path.join(llama_path, "config.json"),
        mode="PRETRAIN",
        lora_adapter_config=os.path.join(MODEL_PATH, "configs", "lora_adapter_configV3.1.json"),
        spk_prompt_dict_path=os.path.join(MODEL_PATH, "configs", "spk_prompt_dict.yaml"),
    )
    llm.llama = LlamaForCausalLM.from_pretrained(llama_path, dtype=torch.float32).to(DEVICE)

    llm.llama_embedding = llm.llama.model.embed_tokens

    special_token_ids = get_special_token_ids(frontend.tokenize_fn)
    print(f"special_token_ids: {special_token_ids}")
    llm.set_runtime_vars(special_token_ids=special_token_ids)

    flow_ckpt = os.path.join(MODEL_PATH, "flow", "flow.pt")
    flow_config = os.path.join(MODEL_PATH, "flow", "config.yaml")
    flow = yaml_util.load_flow_model(flow_ckpt, flow_config, DEVICE)

    ckpt_path = os.path.join(MODEL_PATH, "hift", "hift.pt")
    if sample_rate == 32000:
        ckpt_path = os.path.join(MODEL_PATH, "vocos2d", "generator_jit.ckpt")
    token2wav = tts_model_util.Token2Wav(
        flow,
        sample_rate=sample_rate,
        device=DEVICE,
        ckpt_path=ckpt_path,
    )

    return frontend, text_frontend, speech_tokenizer, llm, token2wav


def dump_model(gpu_prop, **kwargs):
    frontend, text_frontend, speech_tokenizer, llm, token2wav = load_models()

    print_model_params(speech_tokenizer.model, "Speech Tokenizer")  # for speech tokenizer
    print_model_params(llm.llama, f"{MODEL_ID}/llama")  # for LLM

    # speech token -> mel spectrogram -> waveform
    print_model_params(token2wav.flow, f"{MODEL_ID}/token2wav.flow")
    print_model_params(token2wav.vocoder.model, f"{MODEL_ID}/token2wav.vocoder")


def _assert_shape_and_get_len(token):
    assert token.ndim == 2 and token.shape[0] == 1
    token_len = torch.tensor([token.shape[1]], dtype=torch.int32).to(token.device)
    return token_len


def local_llm_forward(
    llm,
    prompt_text_token,
    tts_text_token,
    prompt_speech_token,
    beam_size=1,
    sampling=25,
    sample_method="ras",
):
    """
    Single LLM forward pass.
    """
    prompt_text_token_len = _assert_shape_and_get_len(prompt_text_token)
    tts_text_token_len = _assert_shape_and_get_len(tts_text_token)
    prompt_speech_token_len = _assert_shape_and_get_len(prompt_speech_token)

    tts_speech_token = llm.inference(
        text=tts_text_token,
        text_len=tts_text_token_len,
        prompt_text=prompt_text_token,
        prompt_text_len=prompt_text_token_len,
        prompt_speech_token=prompt_speech_token,
        prompt_speech_token_len=prompt_speech_token_len,
        beam_size=beam_size,
        sampling=sampling,
        sample_method=sample_method,
        spk=None,  # No specific speaker embedding needed for generic pretrain inference here
    )
    return tts_speech_token[0].tolist()


def local_llm_forward_stream_generator(
    llm,
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
            llm.inference(
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


def local_flow_forward(flow, token_list, prompt_speech_tokens, speech_feat, embedding):
    """
    Single Flow forward pass.
    """
    wav, full_mel = flow.token2wav_with_cache(
        token_list,
        prompt_token=prompt_speech_tokens,
        prompt_feat=speech_feat,
        embedding=embedding,
    )
    return wav.detach().cpu(), full_mel


def local_flow_forward_stream(flow, token_list, prompt_speech_tokens, speech_feat, embedding):
    """
    Single stream Flow forward pass.
    """
    wav, _, _, _, mel_list = flow.token2wav_stream(
        token_list,
        prompt_token_list=prompt_speech_tokens,
        prompt_feat_td=speech_feat,
        embedding=embedding,
    )
    full_mel = torch.cat(mel_list, dim=-1) if len(mel_list) > 0 else None
    return wav.detach().cpu(), full_mel


def local_flow_forward_stream_generator(
    flow, token_list, prompt_speech_tokens, speech_feat, embedding, block_sizes
):
    wav_queue = queue.Queue()

    def stream_gen():
        try:
            flow.token2wav_stream(
                token_list,
                block_sizes=block_sizes,
                prompt_token_list=prompt_speech_tokens,
                prompt_feat_td=speech_feat,
                embedding=embedding,
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


# --- Helper Function: Get Prompt from Cache ---
def get_cached_prompt(cache, synth_text_token, device):
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

    prompt_text_token = torch.tensor([prompt_text_token_from_cache]).to(device)

    # Construct Speech Prompt
    speech_tokens = []
    for a_cache_speech_token in cache_speech_token:
        speech_tokens.extend(a_cache_speech_token)

    llm_speech_token = torch.tensor([speech_tokens], dtype=torch.int32).to(device)

    return prompt_text_token, llm_speech_token


# --- Main Generation Logic ---


def generate_long(
    frontend,
    text_frontend,
    llm,
    flow,
    text_info,
    cache,
    device,
    embedding,
    seed=0,
    sample_method="ras",
    flow_prompt_token=None,
    speech_feat=None,
    local_llm_forward=local_llm_forward,
    local_flow_forward=local_flow_forward,
    use_phoneme=False,
):
    outputs = []
    full_mels = []
    output_token_list = []
    uttid = text_info[0]
    syn_text = text_info[1]
    text_tn_dict = {
        "uttid": uttid,
        "syn_text": syn_text,
        "syn_text_tn": [],
        "syn_text_phoneme": [],
    }
    short_text_list = text_frontend.split_by_len(syn_text)

    for _, tts_text in enumerate(short_text_list):
        seed_util.set_seed(seed)
        tts_text_tn = text_frontend.text_normalize(tts_text)  # Normalize again after splitting
        text_tn_dict["syn_text_tn"].append(tts_text_tn)
        if use_phoneme:
            tts_text_tn = text_frontend.g2p_infer(tts_text_tn)
            text_tn_dict["syn_text_phoneme"].append(tts_text_tn)
        tts_text_token = frontend._extract_text_token(tts_text_tn)

        # Access cache references
        cache_text = cache["cache_text"]
        cache_text_token = cache["cache_text_token"]
        cache_speech_token = cache["cache_speech_token"]

        # Determine Prompts
        if cache["use_cache"] and len(cache_text_token) > 1:
            prompt_text_token, prompt_speech_token = get_cached_prompt(
                cache, tts_text_token, device
            )
        else:
            # Initial prompt case
            prompt_text_token = cache_text_token[0].to(device)
            prompt_speech_token = torch.tensor([cache_speech_token[0]], dtype=torch.int32).to(
                device
            )
            print("[generate_long] Using initial prompt (empty cache history)")

        # LLM Inference
        token_list_res = local_llm_forward(
            llm=llm,
            prompt_text_token=prompt_text_token,
            tts_text_token=tts_text_token,
            prompt_speech_token=prompt_speech_token,
            sample_method=sample_method,
        )

        output_token_list.extend(token_list_res)

        # Flow Inference
        output, full_mel = local_flow_forward(
            flow=flow,
            token_list=token_list_res,
            prompt_speech_tokens=flow_prompt_token,
            speech_feat=speech_feat,
            embedding=embedding,
        )

        # Update Cache
        if cache is not None:
            cache_text.append(tts_text_tn)
            cache_text_token.append(tts_text_token)
            cache_speech_token.append(token_list_res)

        outputs.append(output)
        if full_mel is not None:
            full_mels.append(full_mel)

    print(f"[generate_long] Generated speech shape: {output.shape}")
    print(f"[generate_long] Generated mel shape: {full_mel.shape}")
    tts_speech = torch.concat(outputs, dim=1)
    tts_mel = torch.concat(full_mels, dim=-1) if full_mels else None

    return tts_speech, tts_mel, output_token_list, text_tn_dict


def stream_generate_long(
    frontend,
    text_frontend,
    llm,
    flow,
    text_info,
    cache,
    device,
    embedding,
    seed=0,
    sample_method="ras",
    flow_prompt_token=None,
    speech_feat=None,
    use_phoneme=False,
):
    uttid = text_info[0]
    syn_text = text_info[1]
    text_tn_dict = {
        "uttid": uttid,
        "syn_text": syn_text,
        "syn_text_tn": [],
        "syn_text_phoneme": [],
    }
    short_text_list = text_frontend.split_by_len(syn_text)

    for _, tts_text in enumerate(short_text_list):
        seed_util.set_seed(seed)
        tts_text_tn = text_frontend.text_normalize(tts_text)  # Normalize again after splitting
        text_tn_dict["syn_text_tn"].append(tts_text_tn)
        if use_phoneme:
            tts_text_tn = text_frontend.g2p_infer(tts_text_tn)
            text_tn_dict["syn_text_phoneme"].append(tts_text_tn)
        tts_text_token = frontend._extract_text_token(tts_text_tn)

        # Access cache references
        cache_text = cache["cache_text"]
        cache_text_token = cache["cache_text_token"]
        cache_speech_token = cache["cache_speech_token"]

        # Determine Prompts
        if cache["use_cache"] and len(cache_text_token) > 1:
            prompt_text_token, prompt_speech_token = get_cached_prompt(
                cache, tts_text_token, device
            )
        else:
            # Initial prompt case
            prompt_text_token = cache_text_token[0].to(device)
            prompt_speech_token = torch.tensor([cache_speech_token[0]], dtype=torch.int32).to(
                device
            )
            print("[generate_long] Using initial prompt (empty cache history)")

        # LLM Inference
        token_generator = local_llm_forward_stream_generator(
            llm=llm,
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
                    wav_np_generator = local_flow_forward_stream_generator(
                        flow=flow,
                        token_list=token_list,
                        prompt_speech_tokens=flow_prompt_token,
                        speech_feat=speech_feat,
                        embedding=embedding,
                        block_sizes=block_sizes,
                    )
                    for wav_np in wav_np_generator:
                        yield wav_np
                    # loop to check if more full blocks are available now
                    continue
                break

        # after token generator ends, if there are leftover tokens, process them as final block
        if len(all_tokens) - start_idx > 0:
            token_list = all_tokens[start_idx:]
            wav_np_generator = local_flow_forward_stream_generator(
                flow=flow,
                token_list=token_list,
                prompt_speech_tokens=flow_prompt_token,
                speech_feat=speech_feat,
                embedding=embedding,
                block_sizes=block_sizes,
            )
            for wav_np in wav_np_generator:
                yield wav_np

        # Update Cache
        if cache is not None:
            cache_text.append(tts_text_tn)
            cache_text_token.append(tts_text_token)
            cache_speech_token.append(all_tokens)


def stream_generate(gpu_prop, **kwargs):
    import torchaudio

    frontend, text_frontend, speech_tokenizer, llm, token2wav = load_models()
    # Create Output Directory
    data_name = kwargs.get("data_name", "example_zh")
    folder_path = os.path.join(GEN_AUDIO_DIR, f"glm_tts", data_name)
    print(f"Output folder: {folder_path}")
    os.makedirs(folder_path, exist_ok=True)

    # Run Inference (NO voice clone(audio ref with prompt wav))
    sample_rate = 24000
    seed = 0
    use_cache = True
    use_phoneme = False

    # Dataset path resolution
    jsonl_path = os.path.join("/GLM-TTS/examples", data_name + ".jsonl")
    print(f"Using jsonl: {jsonl_path}")
    item_list = file_utils.get_jsonl(jsonl_path)
    output_json_path = os.path.join(folder_path, "text_compare.jsonl")
    print(f"output_json_path: {output_json_path}")

    item = item_list[0]
    try:
        uttid = item["uttid"]
        wav_save_path = os.path.join(folder_path, f"{uttid}_stream.wav")
        item["prompt_speech"] = os.path.join("/GLM-TTS", item["prompt_speech"])

        # Text Normalization
        prompt_text = text_frontend.text_normalize(item["prompt_text"])
        synth_text = text_frontend.text_normalize(item["syn_text"])

        prompt_text_token = frontend._extract_text_token(prompt_text + " ")
        prompt_speech_token = frontend._extract_speech_token([item["prompt_speech"]])
        speech_feat = frontend._extract_speech_feat(item["prompt_speech"], sample_rate=sample_rate)
        embedding = frontend._extract_spk_embedding(item["prompt_speech"])
        cache_speech_token = [prompt_speech_token.squeeze().tolist()]
        flow_prompt_token = torch.tensor(cache_speech_token, dtype=torch.int32).to(DEVICE)

        # Initialize Cache
        cache = {
            "cache_text": [prompt_text],
            "cache_text_token": [prompt_text_token],
            "cache_speech_token": cache_speech_token,
            "use_cache": use_cache,
        }
        syn_text = item["syn_text"]
        print(f"Processing: {uttid}, Syn_text: {syn_text}")

        # Run Generation
        wav_np_generator = stream_generate_long(
            frontend=frontend,
            text_frontend=text_frontend,
            llm=llm,
            flow=token2wav,
            text_info=[uttid, synth_text],
            cache=cache,
            device=DEVICE,
            embedding=embedding,
            seed=seed,
            flow_prompt_token=flow_prompt_token,
            speech_feat=speech_feat,
            use_phoneme=use_phoneme,
        )

        wav_list = []
        for wav_np in wav_np_generator:
            wav_list.append(wav_np)

        if wav_list:
            wav_list = [torch.from_numpy(w) for w in wav_list]
            tts_speech = torch.cat(wav_list, dim=0)
            os.makedirs(os.path.dirname(wav_save_path), exist_ok=True)
            torchaudio.save(wav_save_path, tts_speech.unsqueeze(0), sample_rate)
            print(f"Saved audio to {wav_save_path}")

    except Exception as e:
        print(f"Error processing {item.get('uttid', 'unknown')}: {e}")
        import traceback

        traceback.print_exc()


def jsonl_generate(gpu_prop, **kwargs):
    import tqdm
    import torchaudio

    frontend, text_frontend, speech_tokenizer, llm, token2wav = load_models()
    # Create Output Directory
    data_name = kwargs.get("data_name", "example_zh")
    folder_path = os.path.join(GEN_AUDIO_DIR, f"glm_tts", data_name)
    print(f"Output folder: {folder_path}")
    os.makedirs(folder_path, exist_ok=True)

    # Run Inference (NO voice clone(audio ref with prompt wav))
    sample_rate = 24000
    seed = 0
    use_cache = True
    use_phoneme = False
    flow_forward = local_flow_forward
    if kwargs.get("is_stream_flow", False):
        flow_forward = local_flow_forward_stream
        print("use streaming flow forward!")

    # Dataset path resolution
    jsonl_path = os.path.join("/GLM-TTS/examples", data_name + ".jsonl")
    print(f"Using jsonl: {jsonl_path}")
    item_list = file_utils.get_jsonl(jsonl_path)
    output_json_path = os.path.join(folder_path, "text_compare.jsonl")
    print(f"output_json_path: {output_json_path}")

    with open(output_json_path, "w") as f_out:
        for item in tqdm.tqdm(item_list):
            try:
                uttid = item["uttid"]
                wav_save_path = os.path.join(folder_path, f"{uttid}.wav")
                item["prompt_speech"] = os.path.join("/GLM-TTS", item["prompt_speech"])

                # Text Normalization
                prompt_text = text_frontend.text_normalize(item["prompt_text"])
                synth_text = text_frontend.text_normalize(item["syn_text"])

                prompt_text_token = frontend._extract_text_token(prompt_text + " ")
                prompt_speech_token = frontend._extract_speech_token([item["prompt_speech"]])
                speech_feat = frontend._extract_speech_feat(
                    item["prompt_speech"], sample_rate=sample_rate
                )
                embedding = frontend._extract_spk_embedding(item["prompt_speech"])
                cache_speech_token = [prompt_speech_token.squeeze().tolist()]
                flow_prompt_token = torch.tensor(cache_speech_token, dtype=torch.int32).to(DEVICE)

                # Initialize Cache
                cache = {
                    "cache_text": [prompt_text],
                    "cache_text_token": [prompt_text_token],
                    "cache_speech_token": cache_speech_token,
                    "use_cache": use_cache,
                }
                syn_text = item["syn_text"]
                print(f"Processing: {uttid}, Syn_text: {syn_text}")

                # Run Generation
                tts_speech, _, _, text_tn_dict = generate_long(
                    frontend=frontend,
                    text_frontend=text_frontend,
                    llm=llm,
                    flow=token2wav,
                    text_info=[uttid, synth_text],
                    cache=cache,
                    embedding=embedding,
                    seed=seed,
                    flow_prompt_token=flow_prompt_token,
                    speech_feat=speech_feat,
                    device=DEVICE,
                    local_flow_forward=flow_forward,
                    use_phoneme=use_phoneme,
                )
                f_out.write(json.dumps(text_tn_dict, ensure_ascii=False, indent=2) + "\n")
                f_out.flush()
                # Save Wave and Tokens
                os.makedirs(os.path.dirname(wav_save_path), exist_ok=True)
                torchaudio.save(wav_save_path, tts_speech, sample_rate)
                break

                # Optinal: save prompt features as data input for RL
                # feat_root = os.path.join('grpo', 'data')

                # np.save(os.path.join(feat_root, 'prompt_speech_token', item['uttid']), prompt_speech_token.cpu().squeeze().numpy())
                # np.save(os.path.join(feat_root, 'prompt_speech_feat', item['uttid']), speech_feat.cpu().squeeze().numpy())
                # np.save(os.path.join(feat_root, 'embedding', item['uttid']), embedding.cpu().squeeze().numpy())

            except Exception as e:
                print(f"Error processing {item.get('uttid', 'unknown')}: {e}")
                import traceback

                traceback.print_exc()
                # Optional: raise e # Uncomment to stop on first error


"""

# download glm-tts model 
# 1. frontend campplus.onnx ckpt
modal run src/download_models.py::download_ckpts --ckpt-urls "https://raw.githubusercontent.com/zai-org/GLM-TTS/refs/heads/main/frontend/campplus.onnx" --dir-name zai-org/GLM-TTS/frontend
# 2. speech tokenizer, llm, flow, vocoder(hift) molels
modal run src/download_models.py --repo-ids "zai-org/GLM-TTS"
# 3. configs
modal run src/download_models.py::download_ckpts --ckpt-urls "https://raw.githubusercontent.com/zai-org/GLM-TTS/refs/heads/main/configs/spk_prompt_dict.yaml" --dir-name zai-org/GLM-TTS/configs
modal run src/download_models.py::download_ckpts --ckpt-urls "https://raw.githubusercontent.com/zai-org/GLM-TTS/refs/heads/main/configs/lora_adapter_configV3.1.json" --dir-name zai-org/GLM-TTS/configs

# dump model parameters
IMAGE_GPU=L4 modal run src/llm/transformers/glm_tts.py --task dump_model

# generate audio data from jsonl(example_zh.jsonl)
IMAGE_GPU=L4 modal run src/llm/transformers/glm_tts.py --task jsonl_generate --data-name example_zh
# stream flow generate audio data from jsonl(example_zh.jsonl)
IMAGE_GPU=L4 modal run src/llm/transformers/glm_tts.py --task jsonl_generate --data-name example_zh --is-stream-flow True

IMAGE_GPU=L4 modal run src/llm/transformers/glm_tts.py --task stream_generate --data-name example_zh
"""


@app.local_entrypoint()
def main(task: str = "dump_model", data_name="example_zh", is_stream_flow=False):
    tasks = {
        "dump_model": dump_model,
        "jsonl_generate": jsonl_generate,
        "stream_generate": stream_generate,
    }
    if task not in tasks:
        raise ValueError(f"task {task} not found")
    print(f"running task {task}")
    run.remote(
        tasks[task],
        data_name=data_name,
        is_stream_flow=is_stream_flow,
    )
