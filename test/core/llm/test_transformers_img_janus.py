import os
import io
import logging

from time import perf_counter
import unittest
from dotenv import load_dotenv
from PIL import Image
import numpy as np

from src.core.llm.transformers.base import TransformersBaseLLM
from src.common.logger import Logger
from src.common.session import Session
from src.common.types import MODELS_DIR, TEST_DIR, SessionCtx
from src.core.llm import LLMEnvInit

load_dotenv(override=True)

r"""
LLM_DEVICE=cuda LLM_MODEL_NAME_OR_PATH=./models/deepseek-ai/Janus-Pro-1B \
    python -m unittest test.core.llm.test_transformers_img_janus.TestTransformersImgJanus.test_gen_imgs

LLM_DEVICE=cuda LLM_TAG=llm_transformers_manual_image_janus_flow \
    LLM_MODEL_NAME_OR_PATH=./models/deepseek-ai/JanusFlow-1.3B \
    VAE_MODEL_NAME_OR_PATH=./models/stabilityai/sdxl-vae \
    python -m unittest test.core.llm.test_transformers_img_janus.TestTransformersImgJanus.test_gen_imgs
"""


class TestTransformersImgJanus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed = int(os.getenv("SEED", "1234"))
        cls.parallel_size = int(os.getenv("PARALLEL_SIZE", "1"))
        cls.guidance = float(os.getenv("GUIDANCE", "5.0"))
        cls.img_size = os.getenv("IMAGE_SIZE", "1024x1024")
        cls.gen_width, cls.gen_height = map(int, cls.img_size.split("x"))

        # janus flow
        cls.batch_size = int(os.getenv("BATCH_SIZE", "1"))
        cls.num_inference_steps = int(os.getenv("NUM_INFERENCE_STEPS", "30"))

        cls.llm_tag = os.getenv("LLM_TAG", "llm_transformers_manual_image_janus")
        Logger.init(os.getenv("LOG_LEVEL", "debug").upper(), is_file=False)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.session = Session(**SessionCtx(f"test_{self.llm_tag}_client_id").__dict__)

        engine = LLMEnvInit.initLLMEngine(self.llm_tag)
        self.assertIsInstance(engine, TransformersBaseLLM)
        self.engine: TransformersBaseLLM = engine

    def tearDown(self):
        pass

    def test_gen_imgs(self):
        prompt_cases = [
            "钢铁侠与超人，两位 titan 级的超级英雄, 钢铁侠的战甲: 流线型、充满科技感、金属光泽、损伤、能量指示灯闪烁; 超人的斗篷：鲜红色、迎风飘扬、破损；他们于末日废墟中展开殊死搏斗。钢铁侠的 repulsor 光束划破昏暗的天空，直击超人坚毅的面庞；超人则以雷霆万钧之势，挥舞着钢铁之躯，将钢铁侠的战甲撞击得火花四溅。背景是残垣断壁，浓烟滚滚，烘托出战斗的激烈与悲壮。",
            "Iron Man and Superman, two titan-class superheroes, engage in a deadly battle amidst apocalyptic ruins. Iron Man's armor, streamlined, technologically advanced, with metallic sheen, shows signs of damage, and its energy indicators flicker. Superman's crimson cape billows in the wind, tattered and torn. Iron Man's repulsor beams pierce the gloomy sky, striking Superman's resolute face; Superman retaliates with thunderous force, swinging his steel body, causing Iron Man's armor to spark and crack. The backdrop is one of shattered buildings and billowing smoke, emphasizing the intensity and tragedy of the fight.",
            "A stunning princess from kabul in red, white traditional clothing, blue eyes, brown hair",
        ]

        generate_args = {
            "seed": self.seed,
            "parallel_size": self.parallel_size,
            "guidance": self.guidance,
            "gen_width": self.gen_width,
            "gen_height": self.gen_height,
        }
        if "flow" in self.engine.TAG:
            generate_args = {
                "seed": self.seed,
                "guidance": self.guidance,
                "num_inference_steps": self.num_inference_steps,
                "batch_size": self.batch_size,
                "gen_width": self.gen_width,
                "gen_height": self.gen_height,
            }

        os.makedirs("generated_samples", exist_ok=True)
        i = 0
        for prompt in prompt_cases:
            print("\n--------test prompt: ", prompt, "--------\n")
            with self.subTest(prompt=prompt):
                self.session.ctx.state["prompt"] = prompt
                logging.debug(self.session.ctx)
                logging.debug(self.engine.args)
                iter = self.engine.generate(self.session, **generate_args)

                times = []
                start_time = perf_counter()
                j = 0
                for item in iter:
                    times.append(perf_counter() - start_time)
                    start_time = perf_counter()

                    save_path = os.path.join("generated_samples", f"{self.llm_tag}_img_{i}_{j}.jpg")
                    if isinstance(item, bytes):
                        img = Image.open(io.BytesIO(item))
                        img.save(save_path)

                    j += 1
                i += 1
                logging.debug(f"generate first image time: {times[0]} s")
                self.assertGreater(j, 0)
