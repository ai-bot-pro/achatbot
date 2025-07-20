import time
import logging
import os
import shutil
import sys
import subprocess as sp

import numpy as np

from src.common.time_utils import timeit
from src.common.types import RESOURCES_DIR, MODELS_DIR
from src.common.factory import EngineClass
from src.modules.avatar.interface import IFaceAvatar


try:
    cur_dir = os.path.dirname(__file__)
    if bool(os.getenv("ACHATBOT_PKG", "")):
        sys.path.insert(1, os.path.join(cur_dir, "../../LiteAvatar"))
    else:
        sys.path.insert(1, os.path.join(cur_dir, "../../../deps/LiteAvatar"))
    import torch
    from deps.LiteAvatar.lite_avatar import liteAvatar
    from src.types.avatar.lite_avatar import AvatarConfig, AvatarInitOption, AudioSlice
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error("In order to use lite-avatar, you need to `pip install achatbot[lite_avatar]`. ")
    raise Exception(f"Missing module: {e}")


class BgFrameCounter:
    def __init__(self, total_bg_count, step=1) -> None:
        logging.info(f"create bg frame counter with {total_bg_count} frames")
        self._increase_bg_index = True
        self._step = step
        self._current_bg_index = 0
        self._total_bg_count = total_bg_count

    def get_and_update_bg_index(self):
        """
        get bg index in a front-end-front loop
        """
        if self._total_bg_count <= 1:
            return 0
        bg_index = self._current_bg_index
        for i in range(int(self._step)):
            if self._increase_bg_index:
                if self._current_bg_index == self._total_bg_count - 1:
                    self._increase_bg_index = False
            else:
                if self._current_bg_index == 0:
                    self._increase_bg_index = True
            self._current_bg_index = (1 if self._increase_bg_index else -1) + self._current_bg_index
        return bg_index


class LiteAvatar(IFaceAvatar, EngineClass):
    TAG = "lite_avatar"
    TARGET_FPS = 30

    def __init__(self, **kwargs):
        super().__init__()
        self.tts2face: liteAvatar = None
        self._bg_counter: BgFrameCounter = None
        self._init_option = AvatarInitOption(**kwargs)

    @property
    def init_option(self) -> AvatarInitOption:
        return self._init_option

    def load(self):
        init_option = self._init_option
        data_dir = self._get_avatar_data_dir(init_option.avatar_name)
        self.tts2face = liteAvatar(
            data_dir=data_dir,
            fps=init_option.video_frame_rate,
            use_gpu=init_option.use_gpu,
            weight_dir=init_option.weight_dir,
        )
        bg_step = self.TARGET_FPS // init_option.video_frame_rate
        self.tts2face.load_dynamic_model(data_dir)
        self._bg_counter = BgFrameCounter(len(self.tts2face.ref_img_list), bg_step)
        self.warm_up()

    @timeit
    def audio2signal(self, audio_slice: AudioSlice) -> list:
        signal_list = self.tts2face.audio2param(
            input_audio_byte=audio_slice.algo_audio_data,
            prefix_padding_size=0,
            is_complete=audio_slice.end_of_speech,
        )
        return signal_list

    @timeit
    def signal2img(self, signal_data) -> tuple[torch.Tensor, int]:
        bg_frame_id = self._bg_counter.get_and_update_bg_index()
        mouth_img = self.tts2face.param2img(signal_data, bg_frame_id)
        return mouth_img, bg_frame_id

    @timeit
    def mouth2full(
        self, mouth_image: torch.Tensor, bg_frame_id: int, use_bg: bool = False
    ) -> np.ndarray:
        full_img, _ = self.tts2face.merge_mouth_to_bg(mouth_image, bg_frame_id, use_bg)
        return full_img

    def get_idle_signal(self, idle_frame_count: int) -> list:
        idle_param = self.tts2face.get_idle_param()
        idle_signal_list = []
        for _ in range(idle_frame_count):
            idle_signal_list.append(idle_param)
        return idle_signal_list

    def get_config(self):
        return AvatarConfig(input_audio_sample_rate=16000, input_audio_slice_duration=1)

    def _get_avatar_data_dir(self, avatar_name):
        logging.info(f"use avatar name {avatar_name}")
        avatar_zip_path = self._download_from_modelscope(avatar_name)
        avatar_dir = self.get_avatar_resource_dir()
        extract_dir = os.path.join(avatar_dir, os.path.dirname(avatar_name))
        avatar_data_dir = os.path.join(avatar_dir, avatar_name)
        if not os.path.exists(avatar_data_dir):
            # extract avatar data to dir
            logging.info(f"extract avatar data to dir {extract_dir}")
            assert os.path.exists(avatar_zip_path)
            shutil.unpack_archive(avatar_zip_path, extract_dir)
        assert os.path.exists(avatar_data_dir)
        return avatar_data_dir

    def _download_from_modelscope(self, avatar_name: str = "20250408/sample_data") -> str:
        """
        from https://modelscope.cn/models/HumanAIGC-Engineering/LiteAvatarGallery/files
        avatar_name (eg: 20250612/P1-64AzfrJY037WpS69RiUMw)
        download avatar data from modelscope to resources/avatar/liteavatar
        return avatar_zip_path (eg: resources/avatar/liteavatar/20250612/P1-64AzfrJY037WpS69RiUMw.zip)
        """
        if not avatar_name.endswith(".zip"):
            avatar_name = avatar_name + ".zip"
        avatar_dir = self.get_avatar_resource_dir()
        avatar_zip_path = os.path.join(avatar_dir, avatar_name)
        if not os.path.exists(avatar_zip_path):
            cmd = [
                "modelscope",
                "download",
                "--model",
                "HumanAIGC-Engineering/LiteAvatarGallery",
                avatar_name,
                "--local_dir",
                avatar_dir,
            ]
            logging.info(f"download avatar data from modelscope, cmd: {' '.join(cmd)}")
            sp.run(cmd)
        return avatar_zip_path

    @staticmethod
    def get_avatar_resource_dir():
        avatar_dir = os.path.join(RESOURCES_DIR, "avatar", "liteavatar")
        os.makedirs(avatar_dir, exist_ok=True)
        return avatar_dir

    def warm_up(self):
        for i in range(5):
            start = time.perf_counter()
            self.tts2face.audio2param(bytes(16000 * 2))
            logging.debug(f"warm up {i} step, cost time: {time.perf_counter() - start:0.3f}")


"""
python -m src.modules.avatar.lite_avatar --log_level debug
"""

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--avatar_name", type=str, default="20250408/sample_data")
    parser.add_argument("--audio_sample_rate", type=int, default=16000)
    parser.add_argument("--video_frame_rate", type=int, default=30)
    parser.add_argument(
        "--weight_dir", type=str, default=os.path.join(MODELS_DIR, "weege007/liteavatar")
    )
    parser.add_argument("--log_level", type=str, default="INFO")
    args = parser.parse_args()

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(funcName)s - %(message)s"
    logging.basicConfig(
        level=args.log_level.upper(),
        format=log_format,
    )

    use_gpu = True if torch.cuda.is_available() else False

    lite_avatar = LiteAvatar(
        **AvatarInitOption(
            audio_sample_rate=args.audio_sample_rate,
            video_frame_rate=args.video_frame_rate,
            avatar_name=args.avatar_name,
            weight_dir=args.weight_dir,
            use_gpu=use_gpu,
            is_show_video_debug_text=False,
            enable_fast_mode=False,
            is_flip=False,
        ).__dict__
    )

    lite_avatar._get_avatar_data_dir(args.avatar_name)

    lite_avatar.load()
