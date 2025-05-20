import io
import logging
import os
import sys
from threading import Thread
from dotenv import load_dotenv

from PIL import Image
import numpy as np

from src.common.chat_history import ChatHistory
from src.common.session import Session
from src.common.utils.helper import get_device, print_model_params
from src.core.llm.transformers.base import TransformersBaseLLM
from src.types.llm.transformers import TransformersLMArgs
from src.common.random import set_all_random_seed

load_dotenv(override=True)

try:
    cur_dir = os.path.dirname(__file__)
    if bool(os.getenv("ACHATBOT_PKG", "")):
        sys.path.insert(1, os.path.join(cur_dir, "../../../Janus"))
    else:
        sys.path.insert(1, os.path.join(cur_dir, "../../../../deps/Janus"))
    import torch
    from transformers import AutoConfig, TextIteratorStreamer
    from deps.Janus.janus.janusflow.models import MultiModalityCausalLM, VLChatProcessor
    from diffusers.models import AutoencoderKL
except ModuleNotFoundError as e:
    logging.error(
        "In order to use DeepSeek janus, you need to `pip install achatbot[llm_transformers_manual_vision_img_janus]`."
    )
    raise Exception(f"Missing module: {e}")


class TransformersManualJanusFlow(TransformersBaseLLM):
    r"""
    base class for Multimodal Understanding + Text-to-Image Generation
    https://github.com/deepseek-ai/Janus

    vl_chat_processor.tokenizer + AR LM model + gen_vision_model
    """

    def __init__(self, **args) -> None:
        self.args = TransformersLMArgs(**args)
        self.args.lm_device = self.args.lm_device or get_device()
        logging.info("TransformersLMArgs: %s", self.args)

        # https://huggingface.co/deepseek-ai/JanusFlow-1.3B/blob/main/config.json
        config = AutoConfig.from_pretrained(self.args.lm_model_name_or_path)
        language_config = config.language_config
        language_config._attn_implementation = "eager"
        self._model: MultiModalityCausalLM = MultiModalityCausalLM.from_pretrained(
            self.args.lm_model_name_or_path,
            language_config=language_config,
            trust_remote_code=True,
        )
        print_model_params(self._model, self.TAG)

        self._model = self._model.to(
            self.args.lm_device,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        ).eval()

        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(
            self.args.lm_model_name_or_path
        )
        self._tokenizer = self.vl_chat_processor.tokenizer

        self._chat_history = ChatHistory(self.args.chat_history_size)
        self.warmup()


class TransformersManualVisionJanusFlow(TransformersManualJanusFlow):
    r"""
    Multimodal Understanding
    https://github.com/deepseek-ai/Janus

    vl_chat_processor.tokenizer.encode + AR LM model + vl_chat_processor.tokenizer.decode
    """

    TAG = "llm_transformers_manual_vision_janus_flow"

    def warmup(self):
        dummy_input_text = self.args.warnup_prompt
        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>\n{dummy_input_text}",
            },
            {"role": "Assistant", "content": ""},
        ]

        dummy_pil_images = [Image.new("RGB", (100, 100), color="white")]
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=dummy_pil_images, force_batchify=True
        ).to(
            self._model.device,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        )
        # logging.debug(f"prepare_inputs: {prepare_inputs}")

        # input embeddings
        inputs_embeds = self._model.prepare_inputs_embeds(**prepare_inputs)
        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        # AR lm generate with streamer
        warmup_gen_kwargs = dict(
            inputs_embeds=inputs_embeds,
            streamer=streamer,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self._tokenizer.eos_token_id,
            bos_token_id=self._tokenizer.bos_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            max_new_tokens=self.args.lm_gen_max_new_tokens,
            do_sample=False if self.args.lm_gen_temperature == 0 else True,
            use_cache=True,
            temperature=self.args.lm_gen_temperature,
            top_p=self.args.lm_gen_top_p,
        )

        self._warmup(
            target=self._model.language_model.generate,
            kwargs=warmup_gen_kwargs,
            streamer=streamer,
        )

    @torch.inference_mode()
    def generate(self, session: Session, **kwargs):
        logging.debug(f"kwargs: {kwargs}")
        if "cuda" in str(self._model.device):
            torch.cuda.empty_cache()
        seed = kwargs.get("seed", self.args.lm_gen_seed)
        set_all_random_seed(seed)

        assert isinstance(session.ctx.state["prompt"], list)
        assert len(session.ctx.state["prompt"]) > 1
        question = session.ctx.state["prompt"][-1]
        pil_images = session.ctx.state["prompt"][:-1]

        lm_gen_max_new_tokens = kwargs.get("lm_gen_max_new_tokens", self.args.lm_gen_max_new_tokens)
        lm_gen_temperature = kwargs.get("lm_gen_temperature", self.args.lm_gen_temperature)
        lm_gen_top_p = kwargs.get("lm_gen_top_p", self.args.lm_gen_top_p)

        # conversation
        message = {
            "role": "User",
            "content": f"<image_placeholder>\n{question}",
            # "images": [image_data],
        }
        self._chat_history.append(message)
        chat_history = self._chat_history.to_list()
        logging.debug(f"chat_history:{chat_history}")
        conversation = chat_history + [{"role": "Assistant", "content": ""}]

        # inputs
        # pil_images = [Image.open(io.BytesIO(image_data))]
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(
            self._model.device,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        )
        # logging.debug(f"prepare_inputs: {prepare_inputs}")

        # input embeddings
        inputs_embeds = self._model.prepare_inputs_embeds(**prepare_inputs)

        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        # AR lm generate with streamer
        generation_kwargs = dict(
            inputs_embeds=inputs_embeds,
            streamer=streamer,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self._tokenizer.eos_token_id,
            bos_token_id=self._tokenizer.bos_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            max_new_tokens=lm_gen_max_new_tokens,
            do_sample=False if lm_gen_temperature == 0 else True,
            use_cache=True,
            temperature=lm_gen_temperature,
            top_p=lm_gen_top_p,
        )
        thread = Thread(target=self._model.language_model.generate, kwargs=generation_kwargs)
        thread.start()

        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            yield new_text
        self._chat_history.append({"role": "Assistant", "content": generated_text})
        torch.cuda.empty_cache()


class TransformersManualGenImageJanusFlow(TransformersManualJanusFlow):
    r"""
    Text-to-Image Generation image(384*384)
    https://github.com/deepseek-ai/Janus

    vl_chat_processor.tokenizer.encode + AR LM model + (vision_gen_dec_model + sdxl vae)
    """

    TAG = "llm_transformers_manual_image_janus_flow"

    def __init__(self, **args) -> None:
        vae_model_name_or_path = "stabilityai/sdxl-vae"
        if args.get("vae_model_name_or_path"):
            vae_model_name_or_path = args.pop("vae_model_name_or_path")

        super().__init__(**args)
        self.vae = AutoencoderKL.from_pretrained(vae_model_name_or_path)
        print_model_params(self.vae, "VAE")

        self.vae = self.vae.to(
            self.args.lm_device,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        ).eval()

    def warmup(self):
        pass

    @torch.no_grad()
    def _gen_image(
        self,
        prompt: str,
        cfg_weight: float = 5.0,
        num_inference_steps: int = 30,
        batch_size: int = 5,
    ):
        input_ids = self._tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids)

        tokens = torch.stack([input_ids] * 2 * batch_size).to(self._model.device)
        tokens[batch_size:, 1:] = self.vl_chat_processor.pad_id
        inputs_embeds = self._model.language_model.get_input_embeddings()(tokens)

        # we remove the last <bog> token and replace it with t_emb later
        inputs_embeds = inputs_embeds[:, :-1, :]

        # generate with rectified flow ode
        # step 1: encode with vision_gen_enc
        z = torch.randn((batch_size, 4, 48, 48), dtype=torch.bfloat16).to(self._model.device)

        dt = 1.0 / num_inference_steps
        dt = (
            torch.zeros_like(z)
            .cuda()
            .to(
                self._model.device,
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
            )
            + dt
        )

        # step 2: run ode
        attention_mask = torch.ones((2 * batch_size, inputs_embeds.shape[1] + 577)).to(
            self._model.device
        )
        attention_mask[batch_size:, 1 : inputs_embeds.shape[1]] = 0
        attention_mask = attention_mask.int()
        for step in range(num_inference_steps):
            # prepare inputs for the llm
            z_input = torch.cat([z, z], dim=0)  # for cfg
            t = step / num_inference_steps * 1000.0
            t = torch.tensor([t] * z_input.shape[0]).to(dt)
            z_enc = self._model.vision_gen_enc_model(z_input, t)
            z_emb, t_emb, hs = z_enc[0], z_enc[1], z_enc[2]
            z_emb = z_emb.view(z_emb.shape[0], z_emb.shape[1], -1).permute(0, 2, 1)
            z_emb = self._model.vision_gen_enc_aligner(z_emb)
            llm_emb = torch.cat([inputs_embeds, t_emb.unsqueeze(1), z_emb], dim=1)

            # input to the llm
            # we apply attention mask for CFG: 1 for tokens that are not masked, 0 for tokens that are masked.
            if step == 0:
                outputs = self._model.language_model.model(
                    inputs_embeds=llm_emb,
                    use_cache=True,
                    attention_mask=attention_mask,
                    past_key_values=None,
                )
                past_key_values = []
                for kv_cache in past_key_values:
                    k, v = kv_cache[0], kv_cache[1]
                    past_key_values.append(
                        (k[:, :, : inputs_embeds.shape[1], :], v[:, :, : inputs_embeds.shape[1], :])
                    )
                past_key_values = tuple(past_key_values)
            else:
                outputs = self._model.language_model.model(
                    inputs_embeds=llm_emb,
                    use_cache=True,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                )
            hidden_states = outputs.last_hidden_state

            # transform hidden_states back to v
            hidden_states = self._model.vision_gen_dec_aligner(
                self._model.vision_gen_dec_aligner_norm(hidden_states[:, -576:, :])
            )
            hidden_states = hidden_states.reshape(z_emb.shape[0], 24, 24, 768).permute(0, 3, 1, 2)
            v = self._model.vision_gen_dec_model(hidden_states, hs, t_emb)
            v_cond, v_uncond = torch.chunk(v, 2)
            v = cfg_weight * v_cond - (cfg_weight - 1.0) * v_uncond
            z = z + dt * v

        # step 3: decode with vision_gen_dec and sdxl vae
        decoded_image = self.vae.decode(z / self.vae.config.scaling_factor).sample
        # unpack
        decoded_image = decoded_image.clip_(-1.0, 1.0) * 0.5 + 0.5

        return decoded_image.detach().cpu().type(torch.float32).numpy()

    @torch.inference_mode()
    def generate(self, session: Session, **kwargs):
        logging.debug(f"kwargs: {kwargs}")
        prompt = session.ctx.state["prompt"]
        if "cuda" in str(self._model.device):
            torch.cuda.empty_cache()
        seed = kwargs.get("seed", self.args.lm_gen_seed)
        set_all_random_seed(seed)

        guidance = kwargs.get("guidance", 5.0)
        num_inference_steps = kwargs.get("num_inference_steps", 30)
        batch_size = kwargs.get("batch_size", 1)
        gen_width = kwargs.get("gen_width", 1024)
        gen_height = kwargs.get("gen_height", 1024)
        np_imgs = self._gen_image(
            prompt,
            cfg_weight=guidance,
            num_inference_steps=num_inference_steps,
            batch_size=batch_size,
        )
        logging.debug(f"Generated image shape: {np_imgs.shape}")

        for i in range(np_imgs.shape[0]):
            np_img = np_imgs[i]
            # numpy array -> PIL Image
            img = Image.fromarray((np_img * 255).astype(np.uint8).transpose(1, 2, 0)).resize(
                (gen_width, gen_height), Image.LANCZOS
            )
            # PIL Image -> bytes
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            yield buf.read()


class TransformersManualVisionGenImageJanusFlow(
    TransformersManualGenImageJanusFlow, TransformersManualVisionJanusFlow
):
    r"""
    Multimodal Understanding + Text-to-Image Generation
    https://github.com/deepseek-ai/Janus

    vl_chat_processor.tokenizer.encode + AR LM model + (vision_gen_dec_model + sdxl vae)
    """

    def generate(self, session: Session, **kwargs):
        r"""
        According to Python's MRO (Method Resolution Order) rules:
        In multiple inheritance, Python searches parent classes from left to right
        Using the generate method from the first parent class, we need to define the generate method
        to distinguish based on parameters
        """
        if isinstance(session.ctx.state.get("prompt"), list):
            return TransformersManualVisionJanusFlow.generate(self, session, **kwargs)
        else:
            return TransformersManualGenImageJanusFlow.generate(self, session, **kwargs)
