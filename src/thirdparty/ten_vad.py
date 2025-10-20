#
#  Copyright © 2025 Agora
#  This file is part of TEN Framework, an open source project.
#  Licensed under the Apache License, Version 2.0, with certain conditions.
#  Refer to the "LICENSE" file in the root directory for more information.
#
from ctypes import c_int, c_int32, c_float, c_size_t, CDLL, c_void_p, POINTER
import numpy as np
import os
import platform


class TenVad:
    """binding c lib with ctypes"""

    def __init__(self, hop_size: int = 256, threshold: float = 0.5, lib_path: str | None = None):
        self.hop_size = hop_size
        self.threshold = threshold
        self.vad_library = None
        if lib_path is None:
            # https://github.com/TEN-framework/ten-vad need build target platform lib
            if platform.system() == "Linux" and platform.machine() == "x86_64":
                lib_path = os.path.join(
                    os.path.dirname(os.path.relpath(__file__)), "lib/Linux/x64/libten_vad.so"
                )
            elif platform.system() == "Darwin":
                lib_path = os.path.join(
                    os.path.dirname(os.path.relpath(__file__)),
                    "lib/macOS/ten_vad.framework/Versions/A/ten_vad",
                )
            elif platform.system().upper() == "WINDOWS":
                if platform.machine().upper() in ["X64", "X86_64", "AMD64"]:
                    lib_path = os.path.join(
                        os.path.dirname(os.path.realpath(__file__)), "lib/Windows/x64/ten_vad.dll"
                    )
                else:
                    lib_path = os.path.join(
                        os.path.dirname(os.path.realpath(__file__)), "lib/Windows/x86/ten_vad.dll"
                    )
            else:
                raise NotImplementedError(
                    f"Unsupported platform: {platform.system()} {platform.machine()}"
                )
        if not os.path.exists(lib_path):
            raise NotImplementedError(f"{lib_path} not exists!")

        self.vad_library = CDLL(lib_path)
        self.vad_handler = c_void_p(0)
        self.out_probability = c_float()
        self.out_flags = c_int32()

        self.vad_library.ten_vad_create.argtypes = [
            POINTER(c_void_p),
            c_size_t,
            c_float,
        ]
        self.vad_library.ten_vad_create.restype = c_int

        self.vad_library.ten_vad_destroy.argtypes = [POINTER(c_void_p)]
        self.vad_library.ten_vad_destroy.restype = c_int

        self.vad_library.ten_vad_process.argtypes = [
            c_void_p,
            c_void_p,
            c_size_t,
            POINTER(c_float),
            POINTER(c_int32),
        ]
        self.vad_library.ten_vad_process.restype = c_int
        self.create_and_init_handler()

    def create_and_init_handler(self):
        assert (
            self.vad_library.ten_vad_create(
                POINTER(c_void_p)(self.vad_handler),
                c_size_t(self.hop_size),
                c_float(self.threshold),
            )
            == 0
        ), "[TEN VAD]: create handler failure!"

    def __del__(self):
        assert self.vad_library.ten_vad_destroy(POINTER(c_void_p)(self.vad_handler)) == 0, (
            "[TEN VAD]: destroy handler failure!"
        )

    def get_input_data(self, audio_data: np.ndarray):
        audio_data = np.squeeze(audio_data)
        assert len(audio_data.shape) == 1 and audio_data.shape[0] == self.hop_size, (
            "[TEN VAD]: audio data shape should be [%d]" % (self.hop_size)
        )
        assert isinstance(audio_data[0], np.int16), (
            "[TEN VAD]: audio data type error, must be int16"
        )
        data_pointer = audio_data.__array_interface__["data"][0]
        return c_void_p(data_pointer)

    def process(self, audio_data: np.ndarray):
        input_pointer = self.get_input_data(audio_data)
        self.vad_library.ten_vad_process(
            self.vad_handler,
            input_pointer,
            c_size_t(self.hop_size),
            POINTER(c_float)(self.out_probability),
            POINTER(c_int32)(self.out_flags),
        )
        return self.out_probability.value, self.out_flags.value
