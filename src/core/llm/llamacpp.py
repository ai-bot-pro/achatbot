
from src.common.interface import ILlm
from src.common.factory import EngineClass
from src.common.session import Session
from src.common.types import LLamcppLLMArgs


class LLamacppLLM(ILlm, EngineClass):
    TAG = "llm_llamacpp"

    @classmethod
    def get_args(cls, **kwargs) -> dict:
        return {**(LLamcppLLMArgs).__dict__, **kwargs}

    def __init__(self, **args) -> None:
        self.args = LLamcppLLMArgs(**args)
        from llama_cpp import Llama
        if "chat" in self.args.model_type.lower():
            self.model = Llama(
                model_path=self.args.model_path,
                n_ctx=self.args.n_ctx,
                n_batch=self.args.n_batch,
                n_threads=self.args.n_threads,
                n_gpu_layers=self.args.n_gpu_lay,
                chat_format=self.args.chat_format)
        else:
            self.model = Llama(
                model_path=self.args.model_path,
                n_ctx=self.args.n_ctx,
                n_batch=self.args.n_batch,
                n_threads=self.args.n_threads,
                n_gpu_layers=self.args.n_gpu_layers)

    def generate(self, session: Session):
        prompt = session.ctx.state["prompt"]
        prompt = session.ctx.llm_prompt_tpl % (prompt,)
        output = self.model(
            prompt,
            max_tokens=session.ctx.llm_max_tokens,  # Generate up to 256 tokens
            stop=session.ctx.llm_stop,
            # echo=True,  # Whether to echo the prompt
            stream=session.ctx.llm_stream,
            temperature=session.ctx.llm_temperature,
            top_p=session.ctx.llm_top_p,
        )
        session.ctx.state["llm_text"] = ""
        if session.ctx.llm_stream:
            for item in output:
                session.ctx.state["llm_text"] += item['choices'][0]['text']
                yield item['choices'][0]['text']
        else:
            session.ctx.state["llm_text"] += output['choices'][0]['text']
            yield output['choices'][0]['text']

    def chat_completion(self, session: Session):
        query = session.ctx.state["prompt"]
        output = self.model.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": session.ctx.llm_chat_system,
                },
                {"role": "user", "content": query},
            ],
            # response_format={"type": "json_object"},
            max_tokens=session.ctx.llm_max_tokens,  # Generate up to 256 tokens
            stop=session.ctx.llm_stop,
            stream=session.ctx.llm_stream,
            temperature=session.ctx.llm_temperature,
            top_p=session.ctx.llm_top_p,
        )
        session.ctx.state["llm_text"] = ""
        if session.ctx.llm_stream:
            for item in output:
                if 'content' in item['choices'][0]['delta']:
                    content = item['choices'][0]['delta']['content']
                    session.ctx.state["llm_text"] += content
                    res += content
                    for char in ['.', '。', ',', '，', ';', '；']:
                        if char in content:
                            res += '\n'
                            yield res
                            res = ""
                            break
        else:
            session.ctx.state["llm_text"] += item['choices'][0]['message']['content'] if 'content' in item['choices'][0]['message'] else ""
            yield session.ctx.state["llm_text"]
