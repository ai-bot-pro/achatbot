import os

from funasr_onnx import CT_Transformer_VadRealtime, CT_Transformer

from src.common.session import Session
from src.common.factory import EngineClass
from src.common.interface import IPunc
from src.common.types import MODELS_DIR


class CTTransformerRealtimePuncONNX(EngineClass, IPunc):
    """
    realtime punctuation engine
    - https://www.modelscope.cn/models/iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx
    - modelscope download --local_dir ./models/iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727 iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727
    """

    TAG = "punc_ct_tranformer_onnx"

    def __init__(self, **kwargs):
        """
        - model_dir: model_name in modelscope or local path downloaded from modelscope. If the local path is set, it should contain model.onnx, config.yaml, am.mvn
        - device_id: -1 (Default), infer on CPU. If you want to infer with GPU, set it to gpu_id (Please make sure that you have install the onnxruntime-gpu)
        - quantize: False (Default), load the model of model.onnx in model_dir. If set True, load the model of model_quant.onnx in model_dir
        - intra_op_num_threads: 4 (Default), sets the number of threads used for intraop parallelism on CPU
        """
        super().__init__()
        model_dir = kwargs.get(
            "model_dir",
            os.path.join(
                MODELS_DIR, "iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727"
            ),
        )
        device_id = kwargs.get("device_id", -1)
        quantize = kwargs.get(
            "quantize", True
        )  # False: pt export to onnx, True, export quantized onnx
        intra_op_num_threads = kwargs.get("intra_op_num_threads", 4)
        self.model = CT_Transformer_VadRealtime(
            model_dir=model_dir,
            device_id=device_id,
            quantize=quantize,
            intra_op_num_threads=intra_op_num_threads,
            disable_pbar=True,
            disable_log=True,
        )

    def generate(self, session: Session, **kwargs):
        text = session.ctx.state.get("text", None)
        punc_cache = session.ctx.state.get("punc_cache", [])
        param_dict = {"cache": punc_cache}
        rec_result = self.model(text, param_dict=param_dict)
        session.ctx.state.set("punc_cache", param_dict["cache"])

        return rec_result[0]


class CTTransformerPuncONNX(EngineClass, IPunc):
    """
    onnx offline Punctuation Restoration, pytorch bin -> onnx
    - https://modelscope.cn/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary
    - modelscope download --local_dir ./models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch
    """

    TAG = "punc_ct_tranformer_onnx_offline"

    def __init__(self, **kwargs):
        """
        - model_dir: model_name in modelscope or local path downloaded from modelscope. If the local path is set, it should contain model.onnx, config.yaml, am.mvn
        - device_id: -1 (Default), infer on CPU. If you want to infer with GPU, set it to gpu_id (Please make sure that you have install the onnxruntime-gpu)
        - quantize: False (Default), load the model of model.onnx in model_dir. If set True, load the model of model_quant.onnx in model_dir
        - intra_op_num_threads: 4 (Default), sets the number of threads used for intraop parallelism on CPU
        """
        super().__init__()
        model_dir = kwargs.get(
            "model_dir",
            os.path.join(MODELS_DIR, "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"),
        )
        device_id = kwargs.get("device_id", -1)
        quantize = kwargs.get(
            "quantize", True
        )  # False: pt export to onnx, True, export quantized onnx
        intra_op_num_threads = kwargs.get("intra_op_num_threads", 4)
        self.model = CT_Transformer(
            model_dir=model_dir,
            device_id=device_id,
            quantize=quantize,
            intra_op_num_threads=intra_op_num_threads,
            disable_pbar=True,
            disable_log=True,
        )

    def generate(self, session: Session, **kwargs):
        text = session.ctx.state.get("text", None)
        assert text is not None, "text is None"
        split_size = session.ctx.state.get("split_size", 20)
        rec_result = self.model(text, split_size=split_size)

        return rec_result[0]
