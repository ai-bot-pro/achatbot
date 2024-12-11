import torch
from device_cuda import CUDAInfo
from transformers import pipeline


def pipe_whisper_transcribe(audio_path, model_size="base", target_lang="zh"):
    info = CUDAInfo()

    # Initialize the ASR pipeline
    if info.is_cuda:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_size,
            device="cuda:0",
            torch_dtype=torch.float16 if info.compute_capability_major >= 7 else torch.float32,
            model_kwargs={"use_flash_attention_2": info.compute_capability_major >= 8},
        )

        if info.compute_capability_major == 7 or info.compute_capability_major == 6:
            pipe.model = pipe.model.to_bettertransformer()
    else:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_size,
            device="cpu",
            torch_dtype=torch.float32,
        )

        # for Word-level timestamps batch-size must be 1.
        # https://huggingface.co/openai/whisper-large-v3/discussions/12
        outputs = pipe(
            audio_path,
            chunk_length_s=30,
            batch_size=1,
            generate_kwargs={
                "language": target_lang,
            },
            return_timestamps="word",
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
        "--model_path_or_size",
        "-m",
        type=str,
        default="./models/openai/whisper-base",
        help="model path or size",
    )
    parser.add_argument("--lang", "-l", type=str, default="zh", help="target language")
    args = parser.parse_args()
    pipe_whisper_transcribe(args.audio_path, args.model_path_or_size, args.lang)
