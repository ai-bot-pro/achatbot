r"""
pyannote
- https://www.pyannote.ai/
- https://github.com/pyannote/pyannote-audio
hf-repo-ckpt:
- https://huggingface.co/pyannote/segmentation
- https://huggingface.co/pyannote/segmentation-3.0
- https://huggingface.co/pyannote/speaker-diarization-3.0

paper: 
- [pyannote.audio: neural building blocks for speaker diarization](https://arxiv.org/abs/1911.01255)
- [End-to-end speaker segmentation for overlap-aware resegmentation](https://arxiv.org/abs/2104.04045)
- [Powerset multi-class cross entropy loss for neural speaker diarization](https://arxiv.org/abs/2310.13025)
- [PYANNOTE.AUDIO 2.1 SPEAKER DIARIZATION PIPELINE: PRINCIPLE, BENCHMARK, AND RECIPE](https://huggingface.co/paris-iea/speaker-diarization/resolve/main/technical_report_2.1.pdf)
"""

import os
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Model


def pyannote_vad_pipeline(audio_path, hf_auth_token,
                          path_or_hf_repo="pyannote/segmentation-3.0", model_type="segmentation-3.0"):
    auth_token = os.environ.get('HF_TOKEN') if os.environ.get(
        'HF_TOKEN') else hf_auth_token

    # 1. visit hf.co/pyannote/segmentation-3.0 and accept user conditions
    # 2. visit hf.co/settings/tokens to create an access token
    # 3. instantiate pretrained model
    model = Model.from_pretrained(
        path_or_hf_repo, use_auth_token=auth_token)

    pipeline = VoiceActivityDetection(segmentation=model)
    HYPER_PARAMETERS = {
        # remove speech regions shorter than that many seconds.
        "min_duration_on": 0.0,
        # fill non-speech regions shorter than that many seconds.
        "min_duration_off": 0.0
    }
    # if use pyannote/segmentation open onset/offset activation thresholds
    if model_type == "segmentation":
        HYPER_PARAMETERS["onset"] = 0.5
        HYPER_PARAMETERS["offset"] = 0.5
    pipeline.instantiate(HYPER_PARAMETERS)
    vad_results = pipeline(audio_path)
    print(vad_results)
    # `vad_results` is a pyannote.core.Annotation instance containing speech regions
    for segment in vad_results.itersegments():
        print(segment, segment.start, segment.end)


if __name__ == '__main__':
    r"""
    python demo/vad_pyannote.py -m ./models/pyannote/segmentation/pytorch_model.bin -mt segmentation
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', "-a", type=str,
                        default="./records/tmp.wav", help='audio path')
    parser.add_argument('--auth_token', "-t", type=str,
                        default="", help='hf auth token')
    parser.add_argument('--path_or_hf_repo', "-m", type=str,
                        default="./models/pyannote/segmentation-3.0/pytorch_model.bin",
                        help='model ckpt file path or hf repo')
    parser.add_argument('--model_type', "-mt", type=str,
                        default="segmentation-3.0", choices=["segmentation-3.0", "segmentation"],
                        help='choice segmentation or segmentation-3.0')
    args = parser.parse_args()
    pyannote_vad_pipeline(
        args.audio_path, args.auth_token, args.path_or_hf_repo, args.model_type)
