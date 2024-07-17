import os
import logging

from src.common.factory import EngineFactory, EngineClass
from src.common import interface
from src.modules.functions.function import FunctionManager
from src.common.types import OpenWeatherMapArgs
import src.modules.functions.weather


class WeatherFuncEnvInit:
    engine: interface.IFunction | EngineClass = None

    @staticmethod
    def initWeatherEngine() -> interface.IFunction | EngineClass:
        if WeatherFuncEnvInit.engine is not None:
            logging.info(
                f"WeatherFuncEnvInit.engine already initialized {WeatherFuncEnvInit.engine}")
            return WeatherFuncEnvInit.engine
        tag = os.getenv('FUNC_WEATHER_TAG', "openweathermap_api")
        kwargs = WeatherFuncEnvInit.map_config_func[tag]()
        WeatherFuncEnvInit.engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        logging.info(f"initWeatherEngine: {tag}, {WeatherFuncEnvInit.engine}")
        return WeatherFuncEnvInit.engine

    @staticmethod
    def get_openweathermap_weather_args():
        return OpenWeatherMapArgs(
            units=os.getenv("OPEN_WEATHER_MAP_UNITS", "metric"),
            lang=os.getenv("OPEN_WEATHER_MAP_LANG", "zh_cn"),
        ).__dict__

    # TAG : config
    map_config_func = {
        'openweathermap_api': get_openweathermap_weather_args,
    }


@FunctionManager.functions.register('get_weather')
class WeatherFunc:
    @staticmethod
    def get_tool_call():
        return WeatherFuncEnvInit.initWeatherEngine().get_tool_call()

    @staticmethod
    def execute(session, **args):
        return WeatherFuncEnvInit.initWeatherEngine().execute(session, **args)
