class ChatHistory:
    """
    buffer the local chat hostory with limit size using to avoid llm context too long.
    - if size is None, no limit
    - if size < 0, no history

    """

    def __init__(
        self, size: int | None = None, init_chat_message: dict = None, init_chat_tools: dict = None
    ):
        self.size = size
        self.init_chat_message = init_chat_message
        self.init_chat_tools = init_chat_tools
        # maxlen is necessary pair,
        # since a each new step we add an prompt and assitant answer
        self.buffer = []

    def set_size(self, size: int | None):
        self.size = size

    def clear(self):
        self.buffer.clear()

    def append(self, item):
        if self.size and self.size < 0:
            return

        self.buffer.append(item)
        if self.size is None:
            return

        if len(self.buffer) == 2 * (self.size + 1):
            self.buffer.pop(0)
            self.buffer.pop(0)

    def pop(self, index: int = -1):
        if self.size and self.size < 0:
            return
        if len(self.buffer) > 0:
            self.buffer.pop(index)

    def init(self, init_chat_message: dict):
        self.init_chat_message = init_chat_message

    def init_tools(self, tools: dict):
        self.init_chat_tools = tools

    def to_list(self) -> list:
        if self.init_chat_message:
            if self.init_chat_tools:
                return [self.init_chat_message, self.init_chat_tools] + self.buffer
            else:
                return [self.init_chat_message] + self.buffer
        else:
            return self.buffer

    def __getstate__(self):
        return {
            "size": self.size,
            "init_chat_message": self.init_chat_message,
            "init_chat_tools": self.init_chat_tools,
            "buffer": self.buffer,
        }

    def __setstate__(self, state):
        self.size = state["size"]
        self.init_chat_message = state["init_chat_message"]
        self.init_chat_tools = state["init_chat_tools"]
        self.buffer = state["buffer"]

    def __repr__(self) -> str:
        chat_history = self.__getstate__()
        return f"{chat_history}"


"""
python src/common/chat_history.py
"""
if __name__ == "__main__":
    chat_history = ChatHistory(size=2)
    print(chat_history)
    chat_history.append({"role": "user", "content": "Hello 0"})
    chat_history.append({"role": "assistant", "content": "Hi, how can I help you 0?"})
    print(chat_history)
    chat_history.append({"role": "user", "content": "Hello 1"})
    chat_history.append({"role": "assistant", "content": "Hi, how can I help you 1?"})
    print(chat_history)
    chat_history.pop(-1)
    chat_history.append({"role": "assistant", "content": "Hi, how can I help you 1.1?"})

    chat_history.append({"role": "user", "content": "Hello 2"})
    chat_history.append({"role": "assistant", "content": "Hi, how can I help you 2?"})
    print(chat_history)

    chat_history.append({"role": "user", "content": "Hello 3"})
    chat_history.append({"role": "assistant", "content": "Hi, how can I help you 3?"})
    print(chat_history)
