
from src.common.factory import EngineClass


class BaseLLM(EngineClass):
    def model_name(self):
        if hasattr(self.args, "model_name"):
            return self.args.model_name
        return ""
