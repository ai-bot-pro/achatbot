from ..llamacpp import LLamacppLLM

from src.common.session import Session


class LlamacppGenerator(LLamacppLLM):
    """
    token_ids -> llm generate stream -> token_ids
    """

    TAG = "llm_llamacpp_generator"

    def generate(self, session: Session, **kwargs):
        assert session.ctx.state["token_ids"] is not None
        assert isinstance(session.ctx.state["token_ids"], list)
        token_ids = session.ctx.state["token_ids"]
        generator = self.model.generate(
            token_ids,
            temp=kwargs.get("temperature", self.args.llm_temperature),
            top_k=kwargs.get("top_k", self.args.llm_top_k),
            top_p=kwargs.get("top_p", self.args.llm_top_p),
            repeat_penalty=kwargs.get("repeat_penalty", self.args.llm_repeat_penalty),
        )
        for token_id in generator:
            yield token_id


"""
MODEL=./models/Qwen/Qwen2.5-0.5B python -m src.core.llm.llamacpp.generator 
"""
if __name__ == "__main__":
    from src.common.types import SessionCtx
    import uuid
    import os

    generator = LlamacppGenerator()

    session = Session(**SessionCtx(str(uuid.uuid4().hex)).__dict__)
    gen_iter = generator.generate(session, max_new_tokens=3)
    for token_id in gen_iter:
        print(token_id)
