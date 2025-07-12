from typing import Any, Dict, List

from pydantic import BaseModel, PrivateAttr


class RTVIServiceOptionConfig(BaseModel):
    name: str
    value: Any


class RTVIServiceConfig(BaseModel):
    service: str
    options: List[RTVIServiceOptionConfig]


class RTVIConfig(BaseModel):
    config_list: List[RTVIServiceConfig]
    _arguments_dict: Dict[str, Any] = PrivateAttr(default={})

    def model_post_init(self, __context: Any) -> None:
        self._arguments_dict = {}
        for conf in self.config_list:
            opt_dict = {}
            for opt in conf.options:
                opt_dict[opt.name] = opt.value
            self._arguments_dict[conf.service] = opt_dict
        return super().model_post_init(__context)
