
from abc import ABC, abstractmethod


class IModel(ABC):
    @abstractmethod
    def load_model(self, **kwargs):
        raise NotImplemented("must be implemented in the child class")


class IRecorder(ABC):
    @abstractmethod
    def record_audio(self, session):
        raise NotImplemented("must be implemented in the child class")


class IBuffering(ABC):
    @abstractmethod
    def process_audio(self, session):
        raise NotImplemented("must be implemented in the child class")


class IDetector(ABC):
    @abstractmethod
    async def detect(self, session):
        raise NotImplemented("must be implemented in the child class")


class IAsr(ABC):
    @abstractmethod
    async def transcribe(self, session):
        raise NotImplemented("must be implemented in the child class")


class ILlm(ABC):
    @abstractmethod
    def generate(self, session):
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def chat_completion(self, session):
        raise NotImplemented("must be implemented in the child class")


class IFunction(ABC):
    @abstractmethod
    def excute(self, session):
        raise NotImplemented("must be implemented in the child class")


class ITts(ABC):
    @abstractmethod
    def inference(self, session):
        raise NotImplemented("must be implemented in the child class")
