#
#  Copyright © 2025 Agora
#  This file is part of TEN Framework, an open source project.
#  Licensed under the Apache License, Version 2.0, with certain conditions.
#  Refer to the "LICENSE" file in the root directory for more information.
#
import sys
import os

import scipy.io.wavfile as Wavfile

from src.thirdparty.ten_vad import TenVad

"""
TEN_VAD_LIB_PATH=/Users/wuyong/project/ten-vad/lib/macOS/ten_vad.framework/ten_vad python -m demo.vad_ten ./records/asr_example_zh.wav | less
"""

if __name__ == "__main__":
    lib_path = os.getenv("TEN_VAD_LIB_PATH")
    assert len(sys.argv) > 1
    input_file = sys.argv[1]
    sr, data = Wavfile.read(input_file)
    # hop_size = 256  # 16 ms per frame for 16K hz sample rate
    hop_size = 160  # 10 ms per frame for 16K hz sample rate
    threshold = 0.5
    ten_vad_instance = TenVad(hop_size, threshold, lib_path=lib_path)  # Create a TenVad instance
    num_frames = data.shape[0] // hop_size
    # Streaming inference
    for i in range(num_frames):
        audio_data = data[i * hop_size : (i + 1) * hop_size]
        out_probability, out_flag = ten_vad_instance.process(
            audio_data
        )  #  Out_flag is speech indicator (0 for non-speech signal, 1 for speech signal)
        print("[%d] %0.6f, %d" % (i, out_probability, out_flag))
