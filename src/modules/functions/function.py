import logging

from src.common.register import Register


class FunctionManager:
    """
    FunctionManager just as a namespace to use static methods.
    """

    functions = Register("llm_function_callings")

    def __init__(self):
        raise RuntimeError("FunctionManager is not intended to be instantiated")

    @staticmethod
    def get_tool_calls() -> list[dict]:
        tool_calls = []
        for _, func_cls in FunctionManager.functions.items():
            tool_calls.append(func_cls.get_tool_call())
        return tool_calls

    @staticmethod
    def get_tool_calls_by_names(names: list[str]) -> list[dict]:
        tool_calls = []
        func_list = FunctionManager.functions.items()
        for name, func_cls in func_list:
            if name in names:
                tool_calls.append(func_cls.get_tool_call())
        return tool_calls

    @staticmethod
    def execute(name: str, session, **args):
        func_cls = FunctionManager.functions[name]
        return func_cls.execute(session, **args)


"""
python -m src.modules.functions.function
"""
if __name__ == "__main__":
    import src.modules.functions.search.api
    import src.modules.functions.weather.api
    from src.modules.functions.function import FunctionManager

    items = FunctionManager.functions.items()
    print(items)
    tool_calls = FunctionManager.get_tool_calls_by_names(
        ["web_search", "web_search2", "get_weather"]
    )
    print(tool_calls)
