# -*- coding: utf-8 -*-
import re
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.dlpack import from_dlpack, to_dlpack
import tensorrt_llm.logger as logger
import triton_python_backend_utils as pb_utils

from .tokenizer import get_tokenizer
from .whisper_trtllm import WhisperTRTLLM
from .whisper_utils import mel_filters, log_mel_spectrogram


class FeatureExtractor(torch.nn.Module):
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def __init__(self, n_mels: int = 128, mel_filters_dir: str = None):
        self.device = torch.device("cuda")
        self.n_mels = n_mels
        self.mel_filters_dir = mel_filters_dir
        self.filters = mel_filters(
            self.device,
            n_mels=self.n_mels,
            mel_filters_dir=mel_filters_dir,
        )

    def compute_feature(self, wav, target: int = 3000):
        mel = log_mel_spectrogram(wav, self.filters, mel_filters_dir=self.mel_filters_dir)
        assert mel.shape[1] <= target, f"{mel.shape[1]} > {target}, audio is too long"
        if mel.shape[1] < target:
            mel = F.pad(mel, (0, target - mel.shape[1]), mode="constant")
        mel = mel.unsqueeze(0)
        return mel


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self.model_config = model_config = json.loads(args["model_config"])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(model_config, "TRANSCRIPTS")
        # Convert Triton types to numpy types
        self.out0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

        self.device = torch.device("cuda")
        parameters = self.model_config["parameters"]
        for key, value in parameters.items():
            parameters[key] = value["string_value"]
        engine_dir = parameters["engine_dir"]
        config_path = engine_dir + "/encoder" + "/config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        self.num_languages = config["pretrained_config"]["num_languages"]
        tokenizer_dir = parameters["tokenizer_dir"]
        self.tokenizer = get_tokenizer(
            num_languages=self.num_languages, tokenizer_dir=tokenizer_dir
        )
        self.blank = self.tokenizer.encode(" ", allowed_special=self.tokenizer.special_tokens_set)[
            0
        ]
        n_mels = int(parameters["n_mels"])
        mel_filters_dir = parameters["mel_filters_dir"]
        self.init_model(n_mels, engine_dir, mel_filters_dir)

    def init_model(self, n_mels, engine_dir, mel_filters_dir: str = None):
        self.model = WhisperTRTLLM(engine_dir)
        self.feature_extractor = FeatureExtractor(n_mels=n_mels, mel_filters_dir=mel_filters_dir)

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them. You
        # should avoid storing any of the input Tensors in the class attributes
        # as they will be overridden in subsequent inference requests. You can
        # make a copy of the underlying NumPy array and store it if it is
        # required.
        mel_list, text_prefix_list, repetition_penalty_list = [], [], []
        for request in requests:
            # Perform inference on the request and append it to responses list...
            in_0 = pb_utils.get_input_tensor_by_name(request, "TEXT_PREFIX")
            in_1 = pb_utils.get_input_tensor_by_name(request, "WAV")
            in_2 = pb_utils.get_input_tensor_by_name(request, "REPETITION_PENALTY")

            wav = in_1.as_numpy()

            assert wav.shape[0] == 1, "Only support batch size 1"
            wav = torch.from_numpy(wav[0]).to(self.device)
            mel = self.feature_extractor.compute_feature(wav)
            mel_list.append(mel)

            text_prefix_list.append(in_0.as_numpy().tolist())
            repetition_penalty_list.append(in_2.as_numpy()[0])

        # concat tensors in batch dimension
        features = torch.cat(mel_list, dim=0)
        features = features.to(self.device)
        features_input_lengths = torch.full(
            (features.shape[0],), features.shape[2], dtype=torch.int32, device=features.device
        )

        prompt_ids = []
        for text_prefix in text_prefix_list:
            text_prefix = text_prefix[0][0].decode("utf-8")
            if text_prefix == "":
                text_prefix = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
            prompt_id = self.tokenizer.encode(
                text_prefix, allowed_special=self.tokenizer.special_tokens_set
            )
            # convert prompt_id to tensor, tensor shape is [Seq]
            prompt_id = torch.tensor(prompt_id)
            prompt_ids.append(prompt_id)
        # convert prompt_ids to tensor, tensor shape is [Batch, Seq], left padding with self.blank
        tokens = torch.nn.utils.rnn.pad_sequence(
            prompt_ids, batch_first=True, padding_value=self.blank
        )
        tokens = tokens.to(features.device)
        logger.info(
            f"features.shape: {features.shape}, repetition_penalty: {repetition_penalty_list}"
        )
        output_ids = self.model.process_batch(
            features, features_input_lengths, tokens, repetition_penalty=repetition_penalty_list
        )
        results = [output_ids[i][0] for i in range(len(output_ids))]

        responses = []
        for result in results:
            s = self.tokenizer.decode(result)
            s = re.sub(r"<\|startofprev\|>.*?<\|startoftranscript\|>", "", s)
            s = re.sub(r"<\|.*?\|>", "", s)
            sentence = np.array([s])
            out0 = pb_utils.Tensor("TRANSCRIPTS", sentence.astype(self.out0_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out0])
            responses.append(inference_response)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
