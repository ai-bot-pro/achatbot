from funasr import AutoModel

from src.common.session import Session
from src.common.factory import EngineClass
from src.common.interface import IPunc


class CTTransformerPunc(EngineClass, IPunc):
    """
    - https://www.modelscope.cn/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch
    """

    TAG = "punc_ct_tranformer"

    def __init__(self, **kwargs):
        super().__init__()
        model = kwargs.get("model", "iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727")
        self.model = AutoModel(model=model)

    def generate(self, session: Session, **kwargs):
        text = session.ctx.state.get("text", None)
        punc_cache = session.ctx.state.get("punc_cache", {})
        rec_result = self.model.generate(input=text, cache=punc_cache, disable_pbar=True)

        return rec_result[0]["text"]
