import logging
import time
import os
import shutil
import sys
import json
import subprocess as sp
from typing import Dict, Optional, List


import librosa
import numpy as np
from tqdm import tqdm


from src.common.time_utils import timeit
from src.common.types import RESOURCES_DIR, MODELS_DIR, ASSETS_DIR
from src.common.factory import EngineClass
from src.modules.avatar.interface import IFaceAvatar
from src.types.avatar.lam_audio2expression import LAMAudio2ExpressionAvatarArgs


try:
    cur_dir = os.path.dirname(__file__)
    deps_dir = os.path.join(cur_dir, "../../../deps/LAM_Audio2Expression")
    if bool(os.getenv("ACHATBOT_PKG", "")) is True:
        deps_dir = os.path.join(cur_dir, "../../LAM_Audio2Expression")
    sys.path.insert(0, deps_dir)

    import torch
    from deps.LAM_Audio2Expression.engines.defaults import (
        default_config_parser,
        default_setup,
    )
    from deps.LAM_Audio2Expression.engines.infer import (
        INFER,
        Audio2ExpressionInfer,
        export_blendshape_animation,
    )
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use Audio2Expression-avatar, you need to `pip install achatbot[lam_audio2expression_avatar]`. "
    )
    raise Exception(f"Missing module: {e}")


class LAMAudio2ExpressionAvatar(EngineClass):
    TAG = "lam_audio2expression_avatar"

    def __init__(self, **kwargs):
        super().__init__()
        self.args = LAMAudio2ExpressionAvatarArgs(**kwargs)
        logging.info(f"{self.args=}")
        self.infer: Audio2ExpressionInfer = None
        self.arkit_channels: List[str] = []

    def load(self):
        config_file = os.path.join(
            deps_dir,
            "configs",
            "lam_audio2exp_config_streaming.py",
        )
        wav2vec_config_file = os.path.join(
            deps_dir,
            "configs",
            "wav2vec2_config.json",
        )
        cfg = default_config_parser(
            config_file,
            {
                "weight": self.args.weight_path,
                "audio_sr": self.args.avatar_audio_sample_rate,
                "fps": self.args.fps,
                "model": {
                    "backbone": {
                        "pretrained_encoder_path": self.args.wav2vec_dir,
                        "wav2vec2_config_path": wav2vec_config_file,
                        "expression_dim": self.args.expression_dim,
                    }
                },
            },
        )
        self.cfg = default_setup(cfg)
        logging.info(f"{self.cfg=}")

        self.infer = INFER.build(dict(type=self.cfg.infer.type, cfg=self.cfg))
        self.infer.model.eval()
        arkit_channel_list_path = os.path.join(ASSETS_DIR, "arkit_face_channels.txt")
        self.arkit_channels.clear()
        with open(arkit_channel_list_path, "r") as f:
            for line in f:
                self.arkit_channels.append(line.strip())

        self.warm_up()

    def warm_up(self):
        t_start = time.monotonic()
        context: Optional[Dict] = None
        self.infer.infer_streaming_audio(
            context=context,
            audio=np.zeros([self.args.avatar_audio_sample_rate], dtype=np.float32),
            ssr=self.args.avatar_audio_sample_rate,
        )
        dur_warmup = time.monotonic() - t_start
        logging.info(f"LAM_Audio2Expression warmup finished in {dur_warmup * 1000} milliseconds.")

    def export_animation_json(self, bs_array, json_path=None):
        if json_path is not None:
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
        animation_data = export_blendshape_animation(
            bs_array, json_path, self.arkit_channels, fps=30.0
        )
        return json.dumps(animation_data)


"""
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/LAM_audio2exp_streaming.tar -P ./models/LAM_audio2exp/
tar -xzvf ./models/LAM_audio2exp/LAM_audio2exp_streaming.tar -C ./models/LAM_audio2exp && rm ./models/LAM_audio2exp/LAM_audio2exp_streaming.tar
git clone --depth 1 https://www.modelscope.cn/AI-ModelScope/wav2vec2-base-960h.git ./models/facebook/wav2vec2-base-960h

python -m src.modules.avatar.lam_audio2expression

python -m src.modules.avatar.lam_audio2expression \
    --audio_path test/audio_files/asr_example_zh.wav \
    --save_json_path resources/avatar/lam_audio2expression/expression.json
"""

if __name__ == "__main__":
    import argparse
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weight_path",
        type=str,
        default=os.path.join(
            MODELS_DIR, "LAM_audio2exp/pretrained_models/lam_audio2exp_streaming.tar"
        ),
    )
    parser.add_argument(
        "--wav2vec_dir", type=str, default=os.path.join(MODELS_DIR, "facebook/wav2vec2-base-960h")
    )
    parser.add_argument("--audio_sample_rate", type=int, default=16000)
    parser.add_argument(
        "--audio_path",
        type=str,
        default=os.path.join(deps_dir, "assets/sample_audio/BarackObama.wav"),
    )
    parser.add_argument(
        "--save_json_path",
        type=str,
        default=os.path.join(RESOURCES_DIR, "avatar/lam_audio2expression/expression.json"),
    )
    args = parser.parse_args()
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(funcName)s - %(message)s"
    logging.basicConfig(level="INFO", format=log_format)

    audio, sample_rate = librosa.load(args.audio_path, sr=16000)
    print(f"{audio=} {sample_rate=} {audio.shape[0]=}")

    avatar = LAMAudio2ExpressionAvatar(
        **LAMAudio2ExpressionAvatarArgs(
            weight_path=args.weight_path,
            wav2vec_dir=args.wav2vec_dir,
            speaker_audio_sample_rate=sample_rate,
            avatar_audio_sample_rate=args.audio_sample_rate,
        ).__dict__
    )
    avatar.load()

    context = None
    input_num = (audio.shape[0] + 15999) // sample_rate
    gap = sample_rate
    all_exp = []
    for i in tqdm(range(input_num)):
        start = time.time()
        # infer streaming audio with 30 fps
        output, context = avatar.infer.infer_streaming_audio(
            audio[i * gap : (i + 1) * gap], sample_rate, context
        )
        end = time.time()
        print("Inference time {}".format(end - start))
        print(f"{output['expression'].shape=}")
        all_exp.append(output["expression"])

    all_exp = np.concatenate(all_exp, axis=0)
    print(f"{all_exp.shape=}")

    animation_json_str = avatar.export_animation_json(all_exp, args.save_json_path)
    # print(f"{animation_json_str=}")
