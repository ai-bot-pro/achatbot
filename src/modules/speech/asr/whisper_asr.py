
from src.common.utils.audio_utils import bytes2NpArrayWith16, bytes2TorchTensorWith16
from src.common.factory import EngineClass
from src.common.session import Session
from src.common.interface import IAsr
from src.common.types import WhisperASRArgs
from src.common.device_cuda import CUDAInfo


class WhisperASRBase(EngineClass):
    TAG = "whisper_asr_base"

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**WhisperASRArgs().__dict__, **kwargs}

    def __init__(self, **args) -> None:
        self.args = WhisperASRArgs(**args)
        self.asr_audio = None


class WhisperAsr(WhisperASRBase, IAsr):
    TAG = "whisper_asr"

    def __init__(self, **args) -> None:
        import whisper
        super().__init__(**args)
        self.model = whisper.load_model(
            self.args.model_name_or_path, download_root=self.args.download_path)

    def set_audio_data(self, audio_data):
        if isinstance(audio_data, (bytes, bytearray)):
            self.asr_audio = bytes2NpArrayWith16(audio_data)
        if isinstance(audio_data, str):
            self.asr_audio = audio_data
        return

    async def transcribe(self, session: Session) -> dict:
        transcription = self.model.transcribe(
            self.asr_audio, verbose=self.args.verbose,
            language=self.args.language, word_timestamps=True,
            condition_on_previous_text=True)
        flattened_words = [
            word for segment in transcription["segments"] for word in segment["words"]]
        res = {
            "language": self.args.language,
            "language_probability": transcription["language"],
            "text": transcription["text"].strip(),
            "words": flattened_words,
        }
        return res


class WhisperTimestampedAsr(WhisperAsr):
    TAG = "whisper_timestamped_asr"

    async def transcribe(self, session: Session) -> dict:
        from whisper_timestamped import transcribe_timestamped
        transcription = transcribe_timestamped(
            self.model,
            self.asr_audio,
            language=self.args.language,
            condition_on_previous_text=True,
            verbose=self.args.verbose)
        flattened_words = [
            word for segment in transcription["segments"] for word in segment["words"]]
        res = {"language": self.args.language,
               "language_probability": transcription["language"],
               "text": transcription["text"].strip(),
               "words": [{'text': item['text'],
                          'start': item['start'],
                          'end': item['end'],
                          'probability': item['confidence']} for item in flattened_words],
               }
        return res


class WhisperFasterAsr(WhisperASRBase, IAsr):
    TAG = "whisper_faster_asr"

    def __init__(self, **args) -> None:
        """
        https://github.com/SYSTRAN/faster-whisper?#whisper
        https://opennmt.net/CTranslate2/quantization.html#implicit-type-conversion-on-load
        """
        super().__init__(**args)
        from faster_whisper import WhisperModel
        info = CUDAInfo()
        if info.is_cuda:
            # this worked fast and reliably on NVIDIA L40
            self.model = WhisperModel(
                self.args.model_name_or_path, device="cuda",
                compute_type="float16" if info.compute_capability_major >= 7 else "float32",
                download_root=self.args.download_path)
        else:
            self.model = WhisperModel(
                self.args.model_name_or_path, device="cpu",
                compute_type="float32",
                download_root=self.args.download_path)
        self.asr_audio = None

    def set_audio_data(self, audio_data):
        if isinstance(audio_data, (bytes, bytearray)):
            self.asr_audio = bytes2NpArrayWith16(audio_data)
        if isinstance(audio_data, str):
            self.asr_audio = audio_data
        return

    async def transcribe(self, session: Session) -> dict:
        segmentsIter, info = self.model.transcribe(
            self.asr_audio, language=self.args.language,
            beam_size=5, word_timestamps=True,
            condition_on_previous_text=True)
        # The transcription will actually run here.
        segments = list(segmentsIter)
        flattened_words = [
            word for segment in segments for word in segment.words]

        # print(type(flattened_words[0]))
        res = {
            "language": info.language,
            "language_probability": info.language_probability,
            "text": ' '.join([s.text.strip() for s in segments]),
            "words":
            [{"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                for w in flattened_words]
        }
        return res


class WhisperTransformersAsr(WhisperASRBase, IAsr):
    TAG = "whisper_transformers_asr"

    def __init__(self, **args) -> None:
        super().__init__(**args)
        from transformers import pipeline
        import torch
        info = CUDAInfo()
        # Initialize the ASR pipeline
        if info.is_cuda:
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.args.model_name_or_path,
                device="cuda:0",
                torch_dtype=torch.float16 if info.compute_capability_major >= 7 else torch.float32,
                model_kwargs={
                    "use_flash_attention_2": info.compute_capability_major >= 8})

            if info.compute_capability_major == 7 or info.compute_capability_major == 6:
                self.pipe.model = self.pipe.model.to_bettertransformer()
        else:
            self.pipe = pipeline("automatic-speech-recognition",
                                 model=self.args.model_name_or_path,
                                 device="cpu",
                                 torch_dtype=torch.float32)

    def set_audio_data(self, audio_data):
        if isinstance(audio_data, (bytes, bytearray)):
            self.asr_audio = bytes2TorchTensorWith16(audio_data)
        if isinstance(audio_data, str):
            self.asr_audio = audio_data
        return

    async def transcribe(self, session: Session) -> dict:
        # for Word-level timestamps batch-size must be 1.
        # https://huggingface.co/openai/whisper-large-v3/discussions/12
        outputs = self.pipe(
            self.asr_audio, chunk_length_s=30, batch_size=1,
            generate_kwargs={"language": self.args.language},
            return_timestamps="word")
        res = {"language": self.args.language,
               "language_probability": None,
               "text": outputs['text'].strip(),
               "words": [{"text": item["text"],
                          "start": item["timestamp"][0],
                          "end": item["timestamp"][1]} for item in outputs["chunks"]]}
        return res


class WhisperMLXAsr(WhisperASRBase, IAsr):
    TAG = "whisper_mlx_asr"

    def set_audio_data(self, audio_data):
        if isinstance(audio_data, (bytes, bytearray)):
            self.asr_audio = bytes2NpArrayWith16(audio_data)
        if isinstance(audio_data, str):
            self.asr_audio = audio_data
        return

    async def transcribe(self, session: Session) -> dict:
        import mlx_whisper

        transcribe_kargs = {}
        transcribe_kargs["language"] = self.args.language
        outputs = mlx_whisper.transcribe(
            self.asr_audio,
            path_or_hf_repo=self.args.model_name_or_path,
            word_timestamps=True, **transcribe_kargs)
        res = {"language": self.args.language,
               "language_probability": outputs["language"],
               "text": outputs["text"].strip(),
               "words": [{'text': item['text'],
                          'start': item['start'],
                          'end': item['end'],
                          'probability': item['confidence']} for item in outputs['words']],
               }
        return res
