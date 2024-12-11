import os
from src.types.llm.transformers import TransformersLMArgs
from transformers import HfArgumentParser


class TransformersLLMInit:
    @staticmethod
    def get_transformers_llm_args(parse_file: str = "") -> dict:
        parser = HfArgumentParser(
            TransformersLMArgs,
        )
        if len(parse_file) > 0 and parse_file.endswith(".json"):
            kwargs, _ = parser.parse_json_file(json_file=os.path.abspath(parse_file))
        elif len(parse_file) > 0 and parse_file.endswith(".yaml"):
            kwargs, _ = parser.parse_yaml_file(yaml_file=os.path.abspath(parse_file))
        else:
            kwargs, _ = parser.parse_args_into_dataclasses()
        return vars(kwargs)
