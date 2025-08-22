import os

from funasr import AutoModel

from src.common.session import Session
from src.common.factory import EngineClass
from src.common.interface import IPunc
from src.common.types import MODELS_DIR


class CTTransformerPunc(EngineClass, IPunc):
    """
    pytorch realtime Punctuation Restoration
    - https://www.modelscope.cn/models/iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727
    - modelscope download --local_dir ./models/iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727 iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727
    """

    TAG = "punc_ct_tranformer"

    def __init__(self, **kwargs):
        """
        - model
        """
        super().__init__()
        model = kwargs.get(
            "model",
            os.path.join(
                MODELS_DIR, "iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727"
            ),
        )
        self.model = AutoModel(
            model=model,
            disable_pbar=True,
            disable_log=True,
        )

    def generate(self, session: Session, **kwargs):
        text = session.ctx.state.get("text", None)
        punc_cache = session.ctx.state.get("punc_cache", {})
        rec_result = self.model.generate(input=text, cache=punc_cache, disable_pbar=True)

        return rec_result[0]["text"]
