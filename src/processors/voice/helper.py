import json


def extract_function_info(tool_calls_token: str) -> tuple:
    """
    从 tool_calls_token 字符串中提取 function_name 和 function_args

    参数格式示例：
    'function\nweb_search\n{"query": "2025年8月28日 上证指数 开盘价"}'

    返回: (function_name, function_args_dict)
    """
    # 按换行符分割字符串
    parts = tool_calls_token.split("\n")

    # 验证格式是否正确
    if len(parts) < 3 or parts[0] != "function":
        raise ValueError("无效的 tool_calls_token 格式")

    # 提取函数名
    function_name = parts[1]

    try:
        # 合并剩余部分作为 JSON 字符串（处理可能的多行 JSON）
        json_str = "\n".join(parts[2:])
        function_args = json.loads(json_str)
    except json.JSONDecodeError:
        raise ValueError("无法解析 function_args JSON")

    return function_name, function_args


# TODO: MCP
