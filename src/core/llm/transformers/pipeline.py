from threading import Thread

from transformers import TextIteratorStreamer

from src.common.session import Session
from src.common.interface import ILlm
from src.types.speech.language import TO_LLM_LANGUAGE
from .base import TransformersBaseLLM


class TransformersPipelineLLM(TransformersBaseLLM, ILlm):
    TAG = "llm_transformers_pipeline"

    def init(self):
        from transformers import pipeline

        # https://huggingface.co/docs/transformers/main_classes/pipelines
        self._pipeline = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            device=self._model.device,
        )

    def warmup(self):
        dummy_input_text = self.args.warnup_prompt
        dummy_msgs = [{"role": self.args.user_role, "content": dummy_input_text}]
        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        warmup_gen_kwargs = dict(
            text_inputs=dummy_msgs,
            streamer=streamer,
            return_full_text=False,
            do_sample=self.args.lm_gen_do_sample,
            top_k=self.args.lm_gen_top_k,
            top_p=self.args.lm_gen_top_p,
            temperature=self.args.lm_gen_temperature,
            repetition_penalty=self.args.lm_gen_repetition_penalty,
            min_new_tokens=self.args.lm_gen_min_new_tokens,
            max_new_tokens=self.args.lm_gen_max_new_tokens,
        )

        self._warmup(target=self._pipeline, kwargs=warmup_gen_kwargs, streamer=streamer)

    def generate(self, session: Session):
        prompt = session.ctx.state["prompt"]
        if isinstance(prompt, tuple):
            prompt, language_code = prompt
            if isinstance(prompt, str):
                prompt = (
                    f"Please reply to my message in {TO_LLM_LANGUAGE[language_code]}. " + prompt
                )

        msgs = [{"role": self.args.user_role, "content": prompt}]

        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            text_inputs=msgs,
            return_full_text=False,
            streamer=streamer,
            do_sample=self.args.lm_gen_do_sample,
            top_k=self.args.lm_gen_top_k,
            top_p=self.args.lm_gen_top_p,
            temperature=self.args.lm_gen_temperature,
            repetition_penalty=self.args.lm_gen_repetition_penalty,
            min_new_tokens=self.args.lm_gen_min_new_tokens,
            max_new_tokens=self.args.lm_gen_max_new_tokens,
        )
        thread = Thread(target=self._pipeline, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text
