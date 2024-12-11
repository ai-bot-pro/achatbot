from abc import ABCMeta
import logging
import inspect


class EngineClass(object):
    # the same as ABC(Abstract Base Classe)
    __metaclass__ = ABCMeta

    args = None
    TAG = ""
    SELECTED_TAG = ""

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return kwargs

    def set_args(self, **args):
        if (
            self.args is not None
            and hasattr(self.args, "__dict__")
            and hasattr(self.args, "__class__")
        ):
            self.args = self.args.__class__(**{**self.args.__dict__, **args})

    def get_args_dict(self) -> dict:
        if self.args is not None and hasattr(self.args, "__dict__"):
            return self.args.__dict__
        return {}

    @classmethod
    def get_instance(cls, **kwargs):
        dict_args = cls.get_args(**kwargs)
        logging.info(f"class: {cls} args: {dict_args}")
        instance = cls(**dict_args)
        return instance

    def __str__(self):
        return f"TAG:{self.TAG} | {self.__class__.__name__}"


class EngineFactory:
    @staticmethod
    def get_engine_by_tag(cls, tag: str, **kwargs):
        if not tag or (type(tag) is not str and type(tag) is not list):
            raise TypeError("empty tag")

        def filter_tag(engine):
            if hasattr(engine, "TAG") is False:
                return False
            if engine.TAG == tag:
                return True
            if type(engine.TAG) is list and tag in engine.TAG:
                return True

        selected_engines = list(
            filter(
                filter_tag,
                EngineFactory.get_engines(cls),
            )
        )

        if len(selected_engines) == 0:
            raise ValueError(f"error: can't find {tag} engine")
        else:
            if len(selected_engines) > 1:
                logging.warning(f"have multi {tag}, just use first one")
            engine = selected_engines[0]
            logging.info(f"use {tag} engine")
            engine.SELECTED_TAG = tag
            return engine.get_instance(**kwargs)

    @staticmethod
    def get_engines(cls):
        def get_subclasses(cls):
            subclasses = set()
            for subclass in cls.__subclasses__():
                subclasses.add(subclass)
                subclasses.update(get_subclasses(subclass))
            return subclasses

        return [engine for engine in list(get_subclasses(cls))]

    @staticmethod
    def get_init_engines(object) -> dict:
        engines = {}
        for name, obj in inspect.getmembers(object, inspect.isfunction):
            if "init" not in name and "Engine" not in name:
                continue
            engines[name] = obj()
        return engines
