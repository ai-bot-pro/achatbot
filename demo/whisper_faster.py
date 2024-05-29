#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
see: https://github.com/SYSTRAN/faster-whisper
1. use https://github.com/OpenNMT/CTranslate2 to convert the model
eg: (base size model)
ct2-transformers-converter --model openai/whisper-base --output_dir faster-whisper-base \
--copy_files tokenizer.json - -quantization float16

2. faster_whisper model ckpt from https://huggingface.co/collections/guillaumekln/faster-whisper-64f9c349b3115b4f51434976
"""

from faster_whisper import WhisperModel
from cuda import CUDAInfo


def faster_whisper_transcribe(audio_path, download_root, model_size="base", target_lang="zh"):
    """
    https://github.com/SYSTRAN/faster-whisper?#whisper
    https://opennmt.net/CTranslate2/quantization.html#implicit-type-conversion-on-load
    """
    info = CUDAInfo()
    if info.is_cuda:
        # this worked fast and reliably on NVIDIA L40
        model = WhisperModel(model_size, device="cuda",
                             compute_type="float16" if info.compute_capability_major >= 7 else "float32", download_root=download_root)

        # or run on GPU with INT8
        # tested: the transcripts were different, probably worse than with FP16, and it was slightly (appx 20%) slower
        # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    else:
        # or run on CPU with INT8
        # tested: works, but slow, appx 10-times than cuda FP16
        #        model = WhisperModel(modelsize, device="cpu", compute_type="int8") #, download_root="faster-disk-cache-dir/")
        model = WhisperModel(model_size, device="cpu",
                             compute_type="float32", download_root=download_root)

    segmentsIter, transcriptionInfo = model.transcribe(audio_path, language=target_lang,
                                                       beam_size=5, word_timestamps=True, condition_on_previous_text=True)
    print(transcriptionInfo)
    for segment in segmentsIter:
        print(segment)
        print("[%.2fs -> %.2fs] %s" %
              (segment.start, segment.end, segment.text))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', "-t", type=str,
                        default="whisper", help='choice whisper | whisper_timestamped')
    parser.add_argument('--audio_path', "-a", type=str,
                        default="./records/tmp.wav", help='audio path')
    parser.add_argument('--model_size', "-s", type=str,
                        default="base", help='model size')
    parser.add_argument('--model_path', "-m", type=str,
                        default="./models", help='model root path')
    parser.add_argument('--lang', "-l", type=str,
                        default="zh", help='target language')
    args = parser.parse_args()

    faster_whisper_transcribe(args.audio_path, args.model_path,
                              args.model_size, args.lang)
