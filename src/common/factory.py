from abc import ABCMeta
import logging

logger = logging.getLogger(__name__)


class EngineClass(object):
    # the same as ABC(Abstract Base Classe)
    __metaclass__ = ABCMeta

    args = None
    TAG = ""

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return kwargs

    def set_args(self, **args):
        if self.args is not None \
                and hasattr(self.args, '__dict__') \
                and hasattr(self.args, '__class__'):
            self.args = self.args.__class__(**{**self.args.__dict__, **args})

    @classmethod
    def get_instance(cls, **kwargs):
        dict_args = cls.get_args(**kwargs)
        logging.info(f"get_instance {cls.TAG} args: {dict_args}")
        instance = cls(**dict_args)
        return instance


class EngineFactory:
    @staticmethod
    def get_engine_by_tag(cls, tag: str, **kwargs):
        if not tag or type(tag) is not str:
            raise TypeError(f"empty tag")

        selected_engines = list(
            filter(
                lambda engine: hasattr(engine, "TAG") and engine.TAG == tag,
                EngineFactory.get_engines(cls),
            )
        )

        if len(selected_engines) == 0:
            raise ValueError(f"error: can't find {tag} engine")
        else:
            if len(selected_engines) > 1:
                logger.warning(f"have multi {tag}, just use first one")
            engine = selected_engines[0]
            logger.info(f"use {engine.TAG} engine")
            return engine.get_instance(**kwargs)

    @staticmethod
    def get_engines(cls):
        def get_subclasses(cls):
            subclasses = set()
            for subclass in cls.__subclasses__():
                subclasses.add(subclass)
                subclasses.update(get_subclasses(subclass))
            return subclasses

        return [
            engine
            for engine in list(get_subclasses(cls))
        ]
