import os

import torch
import onnx
from onnxsim import simplify

from src.common.types import MODELS_DIR


def onnx_convert(stream_model, device=torch.device("cpu"), file=f"{MODELS_DIR}/gtcrn.onnx"):
    """ONNX Conversion"""

    conv_cache = torch.zeros(2, 1, 16, 16, 33).to(device)
    tra_cache = torch.zeros(2, 3, 1, 1, 16).to(device)
    inter_cache = torch.zeros(2, 1, 33, 16).to(device)
    if not os.path.exists(file):
        input = torch.randn(1, 257, 1, 2, device=device)
        torch.onnx.export(
            stream_model,
            (input, conv_cache, tra_cache, inter_cache),
            file,
            input_names=["mix", "conv_cache", "tra_cache", "inter_cache"],
            output_names=["enh", "conv_cache_out", "tra_cache_out", "inter_cache_out"],
            opset_version=11,
            verbose=False,
        )

        onnx_model = onnx.load(file)
        onnx.checker.check_model(onnx_model)

    # simplify onnx model
    if not os.path.exists(file.split(".onnx")[0] + "_simple.onnx"):
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, file.split(".onnx")[0] + "_simple.onnx")
