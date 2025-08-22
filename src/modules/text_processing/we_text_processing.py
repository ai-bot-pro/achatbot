from itn.chinese.inverse_normalizer import InverseNormalizer
from tn.chinese.normalizer import Normalizer as ZhNormalizer
from tn.english.normalizer import Normalizer as EnNormalizer

from src.common.session import Session
from src.common.factory import EngineClass
from src.common.interface import ITextProcessing


class WeTextProcessing(EngineClass, ITextProcessing):
    """
    - https://github.com/wenet-e2e/WeTextProcessing
    - https://github.com/wenet-e2e/WeTextProcessing/blob/master/tn/README.md
    - https://github.com/wenet-e2e/WeTextProcessing/blob/master/itn/README.md
    """

    TAG = "we_text_processing"

    def __init__(self, cache_dir=None, overwrite_cache=False, **kwargs):
        super().__init__()
        language = kwargs.pop("language", "zh")  # zh | en
        normalize_type = kwargs.pop("normalize_type", "tn")  # tn | itn
        model_type = language + "_" + normalize_type
        self.model = None
        if model_type == "zh_tn":
            self.model = ZhNormalizer(
                cache_dir=cache_dir, overwrite_cache=overwrite_cache, **kwargs
            )
        if model_type == "zh_itn":
            self.model = InverseNormalizer(
                cache_dir=cache_dir, overwrite_cache=overwrite_cache, **kwargs
            )
        if model_type == "en_tn":
            self.model = EnNormalizer(
                cache_dir=cache_dir, overwrite_cache=overwrite_cache, **kwargs
            )

        assert self.model is not None, f"{model_type} is not support"

    def normalize(self, session, **kwargs):
        text = session.ctx.state["text"]
        return self.model.normalize(text)
