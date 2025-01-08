from dataclasses import dataclass


@dataclass
class MeloTTSArgs:
    device: str = None
    language: str = "ZH"
    tts_config_path: str | None = None
    tts_ckpt_path: str | None = None

    # inference params
    speed: float = 1.0
    sdp_ratio: float = 0.2  # StochasticDurationPredictor and DurationPredictor
    noise_scale_w: float = 0.8  # StochasticDurationPredictor
    noise_scale: float = 0.6  # flow (TransformerCouplingBlock) and reverse flow

    # pbar=None
    # position = None
    quiet = True

    # format = None

    # stream
    tts_stream: bool = False
    chunk_length_seconds: int = 1

    # silence
    add_silence_chunk: bool = False


@dataclass
class OpenVoiceV2TTSArgs(MeloTTSArgs):
    """
    tone color converter with melo-tts
    """

    # tone color converter model ckpt
    enable_clone: bool = False
    converter_conf_path: str = ""
    converter_ckpt_path: str = ""
    # tone color converter params
    tau: float = 0.3  # PosteriorEncoder

    # src tone color feature ckpt
    src_se_ckpt_path: str = ""
    # target tone color feature ckpt
    target_se_ckpt_path: str = ""

    # watermark
    enable_watermark: bool = True
    watermark_name: str = "@achatbot"

    # is save src/target audio
    is_save: bool = False
