import os
import logging

from src.common.http import HTTPRequest
from src.common.factory import EngineClass, EngineFactory
from src.common import interface
from src.modules.functions.function import FunctionManager
from src.common.types import SearchApiArgs, Search1ApiArgs, SerperApiArgs
import src.modules.functions.search


class SearchBaseApi(EngineClass, interface.IFunction):
    def __init__(self) -> None:
        self.requests = HTTPRequest()

    def get_tool_call(self):
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "web search by query",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "web search query"}},
                    "required": ["query"],
                },
            },
        }

    def execute(self, session, **args):
        return self._web_search(session, **args)

    def _web_search(self, session, **args) -> str:
        pass


class SearchFuncEnvInit:
    engine: interface.IFunction | EngineClass = None

    @staticmethod
    def initSearchEngine() -> interface.IFunction | EngineClass:
        if SearchFuncEnvInit.engine is not None:
            logging.info(f"SearchFuncEnvInit.engine already initialized {SearchFuncEnvInit.engine}")
            return SearchFuncEnvInit.engine
        tag = os.getenv("FUNC_SEARCH_TAG", "search_api")
        kwargs = SearchFuncEnvInit.map_config_func[tag]()
        SearchFuncEnvInit.engine = EngineFactory.get_engine_by_tag(EngineClass, tag, **kwargs)
        logging.info(f"initSearchEngine: {tag}, {SearchFuncEnvInit.engine}")
        return SearchFuncEnvInit.engine

    @staticmethod
    def get_serper_api_args():
        return SerperApiArgs(
            gl=os.getenv("SERPER_GL", "cn"),
            hl=os.getenv("SERPER_HL", "zh-cn"),
            page=int(os.getenv("SERPER_PAGE", "1")),
            num=int(os.getenv("SERPER_NUM", "5")),
        ).__dict__

    @staticmethod
    def get_search_api_args():
        return SearchApiArgs(
            engine=os.getenv("SEARCH_ENGINE", "google"),
            gl=os.getenv("SEARCH_GL", "cn"),
            hl=os.getenv("SEARCH_HL", "zh-cn"),
            page=int(os.getenv("SERPER_PAGE", "1")),
            num=int(os.getenv("SERPER_NUM", "5")),
        ).__dict__

    @staticmethod
    def get_search1_api_args():
        return Search1ApiArgs(
            search_service=os.getenv("SEARCH1_ENGINE", "google"),
            image=bool(os.getenv("SEARCH1_IMAGE", "")),
            crawl_results=int(os.getenv("CRAWL_RESULTS", "0")),
            max_results=int(os.getenv("MAX_RESULTS", "5")),
        ).__dict__

    # TAG : config
    map_config_func = {
        "search_api": get_search_api_args,
        "search1_api": get_search1_api_args,
        "serper_api": get_serper_api_args,
    }


@FunctionManager.functions.register("web_search")
class SearchFunc:
    @staticmethod
    def get_tool_call():
        return SearchFuncEnvInit.initSearchEngine().get_tool_call()

    @staticmethod
    def execute(session, **args):
        return SearchFuncEnvInit.initSearchEngine().execute(session, **args)
