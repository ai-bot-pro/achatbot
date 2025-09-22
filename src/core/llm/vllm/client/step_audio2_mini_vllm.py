import os
import re
import io
import sys
import wave
import json
import base64
from typing import BinaryIO

import requests

try:
    import torch
    import numpy

    cur_dir = os.path.dirname(__file__)
    if bool(os.getenv("ACHATBOT_PKG", "")):
        sys.path.insert(1, os.path.join(cur_dir, "../../../../StepAudio2"))
    else:
        sys.path.insert(1, os.path.join(cur_dir, "../../../../../deps/StepAudio2"))

    from deps.StepAudio2.utils import load_audio
except ModuleNotFoundError as e:
    raise Exception(f"Missing module: {e}")


class StepAudio2MiniVLLMClient:
    audio_token_re = re.compile(r"<audio_(\d+)>")

    def __init__(self, api_url, model_name, tokenizer_path: str = None):
        self.api_url = api_url
        self.model_name = model_name

        self.llm_tokenizer = None
        if tokenizer_path:
            from transformers import AutoTokenizer

            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, trust_remote_code=True, padding_side="right", use_fast=True
            )

    def __call__(self, messages, **kwargs):
        return next(self.stream(messages, **kwargs, stream=False))

    def stream(self, messages, stream=True, **kwargs):
        """
        kwargs: class ChatCompletionRequest(OpenAIBaseModel):
        # Ordered by official OpenAI API documentation
        # https://platform.openai.com/docs/api-reference/chat/create
        """
        headers = {"Content-Type": "application/json"}
        payload = kwargs
        payload["messages"] = self.apply_chat_template(messages)
        payload["model"] = self.model_name
        payload["stream"] = stream
        if (payload["messages"][-1].get("role", None) == "assistant") and (
            payload["messages"][-1].get("content", None) is None
        ):
            payload["messages"].pop(-1)
            payload["continue_final_message"] = False
            payload["add_generation_prompt"] = True
        elif payload["messages"][-1].get("eot", True):
            payload["continue_final_message"] = False
            payload["add_generation_prompt"] = True
        else:
            payload["continue_final_message"] = True
            payload["add_generation_prompt"] = False
        with requests.post(self.api_url, headers=headers, json=payload, stream=stream) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line == b"":
                    continue
                line = line.decode("utf-8")[6:] if stream else line.decode("utf-8")
                if line == "[DONE]":
                    break
                line = json.loads(line)["choices"][0]["delta" if stream else "message"]
                text = line.get("tts_content", {}).get("tts_text", None)
                text = text if text else line["content"]
                audio = line.get("tts_content", {}).get("tts_audio", None)
                audio_id = (
                    [int(i) for i in StepAudio2MiniVLLMClient.audio_token_re.findall(audio)]
                    if audio
                    else None
                )

                token_ids = None
                if text and self.llm_tokenizer:
                    token_ids = self.llm_tokenizer.encode(text)
                elif audio and self.llm_tokenizer:
                    token_ids = self.llm_tokenizer.encode(audio)

                yield (line, text, audio_id, token_ids)

    def process_content_item(self, item):
        if item["type"] == "audio":
            audio = item["audio"]
            if isinstance(audio, (BinaryIO, str, os.PathLike)):
                audio = load_audio(item["audio"], target_rate=16000)
            elif isinstance(audio, numpy.ndarray):
                audio = torch.from_numpy(audio, dtype=torch.float32)
            assert isinstance(audio, torch.Tensor), f"Unsupported audio type: {type(audio)}"
            if len(audio.shape) > 1:  # [1, size]
                audio = audio.squeeze(0)  # [size]
            chunks = []
            for i in range(0, audio.shape[0], 25 * 16000):
                chunk = audio[i : i + 25 * 16000]
                if len(chunk.numpy()) == 0:
                    continue
                chunk_int16 = (chunk.numpy().clip(-1.0, 1.0) * 32767.0).astype("int16")
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(chunk_int16.tobytes())
                chunks.append(
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(buf.getvalue()).decode("utf-8"),
                            "format": "wav",
                        },
                    }
                )
            return chunks
        return [item]

    def apply_chat_template(self, messages):
        out = []
        # print(f"{messages=}")
        for m in messages:
            if m["role"] == "human" and isinstance(m["content"], list):
                out.append(
                    {
                        "role": m["role"],
                        "content": [j for i in m["content"] for j in self.process_content_item(i)],
                    }
                )
            else:
                out.append(m)
        return out
