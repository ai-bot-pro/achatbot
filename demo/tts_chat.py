r"""
mkdir -p records && python -m demo.tts_chat
mkdir -p records && STEAM=1 python -m demo.tts_chat
"""

import logging
import os

import torch
import torchaudio
import numpy as np

import deps.ChatTTS.ChatTTS as ChatTTS
from deps.ChatTTS.ChatTTS.core import Chat
from deps.ChatTTS.tools.normalizer.en import normalizer_en_nemo_text
from deps.ChatTTS.tools.normalizer.zh import normalizer_zh_tn


def load_normalizer(chat: Chat):
    # try to load normalizer
    try:
        chat.normalizer.register("en", normalizer_en_nemo_text())
    except ValueError as e:
        logging.error(e)
    except BaseException:
        logging.warning("Package nemo_text_processing not found!")
        logging.warning(
            "Run: conda install -c conda-forge pynini=2.1.5 && pip install nemo_text_processing",
        )
    try:
        chat.normalizer.register("zh", normalizer_zh_tn())
    except ValueError as e:
        logging.error(e)
    except BaseException:
        logging.warning("Package WeTextProcessing not found!")
        logging.warning(
            "Run: conda install -c conda-forge pynini=2.1.5 && pip install WeTextProcessing",
        )


if __name__ == "__main__":
    is_stream = bool(os.getenv("STREAM", ""))

    chat = ChatTTS.Chat()
    load_normalizer(chat)

    chat.load(compile=True, source="custom", custom_path="./models/2Noise/ChatTTS")

    texts = ["你好，我是机器人", "我是机器人"]
    ###################################
    # Sample a speaker from Gaussian.
    rand_spk = chat.sample_random_speaker()
    params_infer_code = Chat.InferCodeParams(
        spk_emb=rand_spk,  # add sampled speaker
        top_P=0.7,  # top P decode
        top_K=20,  # top K decode
        temperature=0.3,  # using custom temperature
    )

    ###################################
    # For sentence level manual control.

    # use oral_(0-9), laugh_(0-2), break_(0-7)
    # to generate special token in text to synthesize.
    params_refine_text = Chat.RefineTextParams(
        prompt="[oral_2][laugh_0][break_6]",
    )

    if is_stream is False:
        wavs = chat.infer(
            texts,
            params_refine_text=params_refine_text,
            params_infer_code=params_infer_code,
            do_text_normalization=False,
        )
        print("wavs[0] type", type(wavs[0]), "wavs[0] shape", wavs[0].shape, "shape", wavs.shape)

        # Assuming 'wavs' is a NumPy array of shape (num_samples,) or (num_channels, num_samples)
        wav_tensor = torch.from_numpy(wavs[0].astype(np.float32))  # Ensure data is float32

        # If 'wavs' is 1D, add a channel dimension
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)
        print("wav tensor shape", wav_tensor.shape, "type", wav_tensor.dtype)

        torchaudio.save("./records/chat_tts_output1.wav", wav_tensor, 24000)

        ###################################
        # For word level manual control.
        text = "What is [uv_break]your favorite english food.[laugh][lbreak]"
        wav = chat.infer(
            text,
            skip_refine_text=True,
            params_refine_text=params_refine_text,
            params_infer_code=params_infer_code,
            do_text_normalization=False,
        )
        print("wav[0] type", type(wav[0]), "shape", wav.shape)

        # Assuming 'wavs' is a NumPy array of shape (num_samples,) or (num_channels, num_samples)
        wav_tensor = torch.from_numpy(wav[0].astype(np.float32))  # Ensure data is float32

        # If 'wavs' is 1D, add a channel dimension
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)
        print("wav tensor shape", wav_tensor.shape, "type", wav_tensor.dtype)

        torchaudio.save("./records/chat_tts_output2.wav", wav_tensor, 24000)

    else:
        ############ stream infer ############
        texts = ["你好，我是机器人", "我是机器人一号", "我是机器人二号"]
        wavs_iter = chat.infer(
            texts,
            stream=True,
            params_refine_text=params_refine_text,
            params_infer_code=params_infer_code,
            do_text_normalization=False,
        )
        res = None
        for wavs in wavs_iter:
            if res is None:
                res = wavs
            else:
                res = np.concatenate([res, wavs], axis=1)

            print("wavs type", type(wavs), "shape", wavs.shape)

        print("res type", type(res), "shape", res.shape)

        for i in range(res.shape[0]):
            # Assuming 'wavs' is a NumPy array of shape (num_samples,) or
            # (num_channels, num_samples)
            wav_tensor = torch.from_numpy(res[i].astype(np.float32))  # Ensure data is float32

            # If 'wavs' is 1D, add a channel dimension
            if wav_tensor.dim() == 1:
                wav_tensor = wav_tensor.unsqueeze(0)
            print("wav tensor shape", wav_tensor.shape, "type", wav_tensor.dtype)

            torchaudio.save(f"./records/chat_tts_stream{i}.wav", wav_tensor, 24000)
