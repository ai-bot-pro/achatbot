r"""
- mlx: https://ml-explore.github.io/mlx/build/html/index.html
- mlx whisper example: https://github.com/ml-explore/mlx-examples/blob/main/whisper/README.md
- mlx whisper model ckpt: https://huggingface.co/collections/mlx-community/whisper-663256f9964fbb1177db93dc
"""

import mlx_whisper


def mlx_whisper_transcribe(
    audio_path, path_or_hf_repo="mlx-community/whisper-base-mlx", target_lang="zh"
):
    # help(mlx_whisper.transcribe)

    transcribe_kargs = {}
    transcribe_kargs["language"] = target_lang
    outputs = mlx_whisper.transcribe(
        audio_path, path_or_hf_repo=path_or_hf_repo, word_timestamps=True, **transcribe_kargs
    )
    print(outputs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        "-t",
        type=str,
        default="whisper",
        help="choice whisper | whisper_timestamped",
    )
    parser.add_argument(
        "--audio_path", "-a", type=str, default="./records/tmp.wav", help="audio path"
    )
    parser.add_argument(
        "--path_or_hf_repo",
        "-m",
        type=str,
        default="./models/mlx-community/whisper-base-mlx",
        help="model path or hf repo",
    )
    parser.add_argument("--lang", "-l", type=str, default="zh", help="target language")
    args = parser.parse_args()
    mlx_whisper_transcribe(args.audio_path, args.path_or_hf_repo, args.lang)
