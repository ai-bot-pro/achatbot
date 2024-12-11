import logging
import os


from src.common.session import Session
from src.common.interface import IDetector
from src.common.types import MODELS_DIR, RATE, CHUNK
from src.common.factory import EngineClass


class PyannoteDetector(EngineClass):
    @staticmethod
    def load_model(
        hf_auth_token, path_or_hf_repo="pyannote/segmentation-3.0", model_type="segmentation-3.0"
    ):
        auth_token = os.environ.get("HF_TOKEN") if os.environ.get("HF_TOKEN") else hf_auth_token
        if auth_token is None:
            raise ValueError(
                "Missing required auth_token, env var in HF_TOKEN or from hf_auth_token param"
            )

        if model_type == "diarization":
            from pyannote.audio import Pipeline

            auth_token = os.environ.get("HF_TOKEN") if os.environ.get("HF_TOKEN") else hf_auth_token
            # instantiate the pipeline
            model = Pipeline.from_pretrained(path_or_hf_repo, use_auth_token=auth_token)
        else:
            # 1. visit hf.co/pyannote/segmentation-3.0 and accept user conditions
            # 2. visit hf.co/settings/tokens to create an access token
            # 3. instantiate pretrained model
            from pyannote.audio import Model

            model = Model.from_pretrained(path_or_hf_repo, use_auth_token=auth_token)

        return model

    def __init__(self, **args) -> None:
        from src.types.speech.detector.pyannote import PyannoteDetectorArgs

        self.args = PyannoteDetectorArgs(**args)
        self.hyper_parameters = {
            # remove speech regions shorter than that many seconds.
            "min_duration_on": self.args.min_duration_on,
            # fill non-speech regions shorter than that many seconds.
            "min_duration_off": self.args.min_duration_off,
        }
        # if use pyannote/segmentation open onset/offset activation thresholds
        if self.args.model_type == "segmentation":
            self.hyper_parameters["onset"] = self.args.onset
            self.hyper_parameters["offset"] = self.args.offset

        self.model = PyannoteDetector.load_model(
            self.args.hf_auth_token, self.args.path_or_hf_repo, self.args.model_type
        )

    def set_audio_data(self, audio_data):
        from pyannote.audio.core.io import AudioFile

        if isinstance(audio_data, AudioFile):
            self.args.vad_pyannote_audio = audio_data

    def get_sample_info(self):
        return RATE, CHUNK

    def close(self):
        pass


class PyannoteVAD(PyannoteDetector, IDetector):
    r"""
    voice activity detection (语音活动识别)
    """

    TAG = "pyannote_vad"

    def __init__(self, **args) -> None:
        super().__init__(**args)

        from pyannote.audio.pipelines import VoiceActivityDetection

        self.pipeline = VoiceActivityDetection(segmentation=self.model)
        self.pipeline.instantiate(self.hyper_parameters)

    async def detect(self, session: Session):
        vad_res = self.pipeline(self.args.vad_pyannote_audio)
        # `vad_res` is a pyannote.core.Annotation instance containing speech regions
        vad_segments = []
        for segment in vad_res.itersegments():
            logging.debug(f"vad_segment: {segment}")
            vad_segments.append({"start": segment.start, "end": segment.end, "confidence": 1.0})
        logging.debug(f"vad_segments: {vad_segments}")
        return vad_segments


class PyannoteOSD(PyannoteDetector, IDetector):
    r"""
    Overlapped speech detection (重叠语音检测)
    """

    TAG = "pyannote_osd"

    def __init__(self, **args) -> None:
        super().__init__(**args)

        from pyannote.audio.pipelines import OverlappedSpeechDetection

        self.pipeline = OverlappedSpeechDetection(segmentation=self.model)
        self.pipeline.instantiate(self.hyper_parameters)

    async def detect(self, session: Session) -> None:
        vad_res = self.pipeline(self.args.vad_pyannote_audio)
        logging.debug(f"vad_res: {vad_res}")
        # `vad_res` is a pyannote.core.Annotation instance containing speech regions
        vad_segments = []
        if len(vad_res) > 0:
            vad_segments = [
                {"start": segment.start, "end": segment.end, "confidence": 1.0}
                for segment in vad_res.itersegments()
            ]
        logging.debug(f"vad_segments: {vad_segments}")
        return vad_segments


class PyannoteDiarization(PyannoteDetector, IDetector):
    r"""
    Speaker diarization (说话人分割或说话人辨识)
    """

    TAG = "pyannote_diarization"

    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.pipeline = self.model

    async def detect(self, session: Session):
        # run the pipeline on an audio file
        diarization = self.pipeline(self.args.vad_pyannote_audio)
        logging.debug(f"diarization: {diarization}")

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

        # dump the diarization output to disk using RTTM format
        # with open(diarization_path, "w") as rttm:
        #    diarization.write_rttm(rttm)

        vad_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            logging.debug(
                f"turn:{turn} start: {turn.start:.3f} end: {turn.end:.3f}, speaker: {speaker}"
            )
            vad_segments.append({"start": turn.start, "end": turn.end, "confidence": 1.0})

        return vad_segments
