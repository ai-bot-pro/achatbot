from dataclasses import dataclass, field
import os

from src.common.types import MODELS_DIR


@dataclass
class Mega3TTSArgs:
    device: str = None

    ckpt_dir: str = os.path.join(MODELS_DIR, "ByteDance/MegaTTS3")

    ref_audio_file: str = ""
    # .npy ref_latent_file np.save np.darray from wavevae encoder encode audio
    ref_latent_file: str = ""

    # infer params
    time_step: int = 32  # Inference steps of Diffusion Transformer (DiT)
    p_w: float = 1.6  # Intelligibility Weight
    t_w: float = 2.5  # Similarity Weight
    dur_disturb: float = 0.1  # duration disturbance for duration prediction
    dur_alpha: float = 1.0  # duration alpha for duration prediction
