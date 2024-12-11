import logging

from src.common.register import Register


class FunctionManager:
    """
    FunctionManager just as a namespace to use static methods.
    """

    functions = Register("llm_function_calling")

    def __init__(self):
        raise RuntimeError("FunctionManager is not intended to be instantiated")

    @staticmethod
    def get_tool_calls() -> list[dict]:
        tool_calls = []
        for _, func_cls in FunctionManager.functions.items():
            tool_calls.append(func_cls.get_tool_call())
        return tool_calls

    @staticmethod
    def execute(name: str, session, **args):
        func_cls = FunctionManager.functions[name]
        return func_cls.execute(session, **args)
