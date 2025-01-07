from dataclasses import dataclass, field


@dataclass
class E2TTSUNetTModelConfig:
    """
    E2-TTS U-Net Transformer's model config
    2021. UNETR: Transformers for 3D Medical Image Segmentation
    (UNETR (U-Net Transformer))
    2024. E2 TTS: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS
    """

    dim: int = 1024
    depth: int = 24
    heads: int = 16
    ff_mult: int = 4


@dataclass
class F5TTSDiTModelConfig:
    """
    F5-TTS DiT's model config
    2023. Scalable Diffusion Models with Transformers (DiT)
    2024. F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching

    NOTE: use thop.profile to compute FLOPs
    ~335M
    # FLOPs: 622.1 G, Params: 333.2 M
    transformer =     UNetT(dim = 1024, depth = 24, heads = 16, ff_mult = 4)
    # FLOPs: 363.4 G, Params: 335.8 M
    transformer = DiT(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    UNetT 和 DiT 两个模型参数量相近, 但是DiT计算量低于UNetT, DiT在计算效率上更优
    """

    dim: int = 1024
    depth: int = 22
    heads: int = 16
    ff_mult: int = 2
    text_dim: int = 512
    conv_layers: int = 4


@dataclass
class F5TTSArgs:
    """
    2024. F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching
    """

    model_type: str = "F5-TTS"  # F5-TTS with DiT | E2-TTS with U-Net Transformer
    model_ckpt_path: str = ""
    model_cfg: dict = field(default_factory=lambda: F5TTSDiTModelConfig().__dict__)

    vocoder_name: str = "vocos"  # gen mel_spec_type: vocos | bigvgan
    # if None from hf repo download
    # hf repo_id: charactr/vocos-mel-24khz | nvidia/bigvgan_v2_24khz_100band_256x
    vocoder_ckpt_dir: str = None

    # tokenizer to vocab_char_map default use vocab.text to inference with custom
    # default: f5_tts/infer/examples/vocab.txt
    vocab_file: str = ""

    # ODE solvers
    # from: https://github.com/rtqichen/torchdiffeq
    ode_method: str = "euler"  # Fixed-stepa 默认： euler (欧拉方法) | midpoint (中点法)

    # https://en.wikipedia.org/wiki/Moving_average
    # exponential moving average (EMA),指数移动平均线
    use_EMA: bool = True

    # "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device: str = None

    # inference ref
    # ref_audio_file default: f5_tts/infer/examples/basic/basic_ref_en.wav
    ref_audio_file: str = ""
    ref_text: str = ""
    preprocess_ref_audio_text:bool = False

    # inference params
    target_rms: float = 0.1
    cross_fade_duration: float = 0.15
    """
    Sway Sampling 是一种扩散模型（Diffusion Model）的采样策略。
    Sway Sampling 是一种在inference生成过程中对噪声进行调整的方法，目的是改善生成质量和效率。
    通过参数 sway_sampling_coef 来控制，这个参数在代码中默认值是 -1.0
    通过 sway_sampling_coef 参数来控制采样过程中的"摆动"程度：
    - 当系数为负值时，会产生一种来回"摆动"的效果
    - 当系数为0时，相当于标准采样
    - 当系数为正值时，会产生相反方向的"摆动"效果
    这种采样策略的主要优势是：
    - 可以帮助模型在生成过程中探索更多可能的状态
    - 有助于避免生成结果陷入局部最优
    - 可能提高生成音频的自然度和质量
    需要注意的是，不同的 sway_sampling_coef 值可能会对生成结果产生不同的影响，
    可以通过调整这个参数来获得最适合您需求的生成效果。
    """
    sway_sampling_coef: float = -1.0
    cfg_strength: float = 2.0
    nfe_step: int = 32
    speed: float = 1.5  # speak speed
    fix_duration: int = None
    seed: int = -1  # torch manual seed

    # extra param
    add_silence_chunk: bool = True
