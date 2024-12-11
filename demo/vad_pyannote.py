r"""
pyannote
- https://www.pyannote.ai/
- https://github.com/pyannote/pyannote-audio
hf-repo-ckpt:
- https://huggingface.co/pyannote/segmentation
- https://huggingface.co/pyannote/segmentation-3.0
- https://huggingface.co/pyannote/speaker-diarization-3.0
- https://huggingface.co/pyannote/speaker-diarization-3.1

paper:
- [pyannote.audio: neural building blocks for speaker diarization](https://arxiv.org/abs/1911.01255)
- [End-to-end speaker segmentation for overlap-aware resegmentation](https://arxiv.org/abs/2104.04045)
- [Powerset multi-class cross entropy loss for neural speaker diarization](https://arxiv.org/abs/2310.13025)
- [PYANNOTE.AUDIO 2.1 SPEAKER DIARIZATION PIPELINE: PRINCIPLE, BENCHMARK, AND RECIPE](https://huggingface.co/paris-iea/speaker-diarization/resolve/main/technical_report_2.1.pdf)
"""

import os


def init_model(
    hf_auth_token,
    path_or_hf_repo="pyannote/segmentation-3.0",
):
    from pyannote.audio import Model

    auth_token = os.environ.get("HF_TOKEN") if os.environ.get("HF_TOKEN") else hf_auth_token

    # 1. visit hf.co/pyannote/segmentation-3.0 and accept user conditions
    # 2. visit hf.co/settings/tokens to create an access token
    # 3. instantiate pretrained model
    model = Model.from_pretrained(path_or_hf_repo, use_auth_token=auth_token)

    return model


def pyannote_vad_pipeline(
    audio_path,
    hf_auth_token,
    path_or_hf_repo="pyannote/segmentation-3.0",
    model_type="segmentation-3.0",
):
    r"""
    voice activity detection (语音活动识别)
    """
    from pyannote.audio.pipelines import VoiceActivityDetection

    pipeline = VoiceActivityDetection(segmentation=init_model(hf_auth_token, path_or_hf_repo))
    HYPER_PARAMETERS = {
        # remove speech regions shorter than that many seconds.
        "min_duration_on": 0.0,
        # fill non-speech regions shorter than that many seconds.
        "min_duration_off": 0.0,
    }
    # if use pyannote/segmentation open onset/offset activation thresholds
    if model_type == "segmentation":
        HYPER_PARAMETERS["onset"] = 0.5
        HYPER_PARAMETERS["offset"] = 0.5
    pipeline.instantiate(HYPER_PARAMETERS)
    vad_res = pipeline(audio_path)
    print(type(vad_res), vad_res)
    # `vad_res` is a pyannote.core.Annotation instance containing speech regions
    for segment in vad_res.itersegments():
        print(type(segment), segment, segment.start, segment.end)


def pyannote_osd_pipeline(
    audio_path,
    hf_auth_token,
    path_or_hf_repo="pyannote/segmentation-3.0",
    model_type="segmentation-3.0",
):
    r"""
    Overlapped speech detection (重叠语音检测)
    """
    from pyannote.audio.pipelines import OverlappedSpeechDetection

    pipeline = OverlappedSpeechDetection(segmentation=init_model(hf_auth_token, path_or_hf_repo))
    HYPER_PARAMETERS = {
        # remove speech regions shorter than that many seconds.
        "min_duration_on": 0.0,
        # fill non-speech regions shorter than that many seconds.
        "min_duration_off": 0.0,
    }
    # if use pyannote/segmentation open onset/offset activation thresholds
    if model_type == "segmentation":
        HYPER_PARAMETERS["onset"] = 0.5
        HYPER_PARAMETERS["offset"] = 0.5
    pipeline.instantiate(HYPER_PARAMETERS)
    vad_res = pipeline(audio_path)
    print(type(vad_res), vad_res)
    # `vad_res` is a pyannote.core.Annotation instance containing speech regions
    for segment in vad_res.itersegments():
        print(type(segment), segment, segment.start, segment.end)


def pyannote_diarization_pipeline(
    audio_path,
    hf_auth_token,
    path_or_hf_repo="pyannote/speaker-diarization-3.1",
    diarization_path="./records/diarization_audio.rttm",
):
    r"""
    Speaker diarization (说话人分割或说话人辨识)
    """
    from pyannote.audio import Pipeline

    auth_token = os.environ.get("HF_TOKEN") if os.environ.get("HF_TOKEN") else hf_auth_token
    # instantiate the pipeline
    pipeline = Pipeline.from_pretrained(path_or_hf_repo, use_auth_token=auth_token)

    # run the pipeline on an audio file
    diarization = pipeline(audio_path)
    print(type(diarization), diarization)

    # Pre-loading audio files in memory may result in faster processing:
    # import torchaudio
    # waveform, sample_rate = torchaudio.load("audio.wav")
    # diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    # Monitoring progress Hooks are available to monitor the progress of the pipeline:
    # from pyannote.audio.pipelines.utils.hook import ProgressHook
    # with ProgressHook() as hook:
    #    diarization = pipeline("audio.wav", hook=hook)

    # Controlling the number of speakers
    # diarization = pipeline("audio.wav", num_speakers=2)
    # diarization = pipeline("audio.wav", min_speakers=2, max_speakers=5)

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(type(turn), turn, f"start: {turn.start:.3f}", f"end: {turn.end:.3f}")
        print(type(speaker), speaker)

    # dump the diarization output to disk using RTTM format
    with open(diarization_path, "w") as rttm:
        diarization.write_rttm(rttm)


if __name__ == "__main__":
    r"""
    python demo/vad_pyannote.py -m ./models/pyannote/segmentation/pytorch_model.bin -mt segmentation
    python demo/vad_pyannote.py -dt osd
    python demo/vad_pyannote.py -m pyannote/speaker-diarization-3.1 -dt diarization -dp ./records/diarization_audio.rttm
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_path", "-a", type=str, default="./records/tmp.wav", help="audio path"
    )
    parser.add_argument("--auth_token", "-t", type=str, default="", help="hf auth token")
    parser.add_argument(
        "--path_or_hf_repo",
        "-m",
        type=str,
        default="./models/pyannote/segmentation-3.0/pytorch_model.bin",
        help="model ckpt file path or hf repo",
    )
    parser.add_argument(
        "--model_type",
        "-mt",
        type=str,
        default="segmentation-3.0",
        choices=["segmentation-3.0", "segmentation", "diarization"],
        help="choice segmentation or segmentation-3.0 or diarization",
    )
    parser.add_argument(
        "--detect_type",
        "-dt",
        type=str,
        default="vad",
        choices=["vad", "osd", "diarization"],
        help="choice vad, osd, diarization",
    )
    parser.add_argument(
        "--diarization_path",
        "-dp",
        type=str,
        default="./records/diarization_audio.rttm",
        help="diarization rttm file path",
    )
    args = parser.parse_args()

    if args.detect_type == "diarization":
        pyannote_diarization_pipeline(
            args.audio_path, args.auth_token, args.path_or_hf_repo, args.diarization_path
        )
    elif args.detect_type == "osd":
        pyannote_osd_pipeline(
            args.audio_path, args.auth_token, args.path_or_hf_repo, args.model_type
        )
    else:
        pyannote_vad_pipeline(
            args.audio_path, args.auth_token, args.path_or_hf_repo, args.model_type
        )
