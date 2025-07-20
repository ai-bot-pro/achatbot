import asyncio
from io import BytesIO
import os
import logging
from typing import Dict
from contextlib import asynccontextmanager
import multiprocessing

import unittest
from apipeline.pipeline.pipeline import Pipeline
from apipeline.pipeline.task import PipelineTask, PipelineParams
from apipeline.pipeline.runner import PipelineRunner
from apipeline.frames import EndFrame, AudioRawFrame
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketState
import uvicorn
import aiohttp

from src.common.utils.wav import read_wav_to_bytes
from src.common.logger import Logger
from src.common.types import TEST_DIR, MODELS_DIR
from src.types.frames.control_frames import (
    AvatarArgsUpdateFrame,
    AvatarLanguageUpdateFrame,
    AvatarModelUpdateFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from src.types.avatar.lam_audio2expression import LAMAudio2ExpressionAvatarArgs
from src.types.network.fastapi_websocket import AudioCameraParams, FastapiWebsocketServerParams
from src.transports.fastapi_websocket_server import FastapiWebsocketTransport
from src.serializers.avatar_protobuf import AvatarProtobufFrameSerializer


from dotenv import load_dotenv


load_dotenv(override=True)

Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False)

"""
# download model weights
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/LAM_audio2exp_streaming.tar -P ./models/LAM_audio2exp/
tar -xzvf ./models/LAM_audio2exp/LAM_audio2exp_streaming.tar -C ./models/LAM_audio2exp && rm ./models/LAM_audio2exp/LAM_audio2exp_streaming.tar
git clone --depth 1 https://www.modelscope.cn/AI-ModelScope/wav2vec2-base-960h.git ./models/facebook/wav2vec2-base-960h

python -m unittest test.integration.processors.test_lam_audio2expression_avatar_processor.TestLamAudio2ExpressionProcessor.test_gen

DAILY_ROOM_URL=https://weedge.daily.co/jk5g4mFlZkPHvOyaEZe5 \
    WEIGHT_PATH=./models/LAM_audio2exp/pretrained_models/lam_audio2exp_streaming.tar \
    WAV2VEC_DIR=./models/facebook/wav2vec2-base-960h \
    SLEEP_TO_END_TIME_S=12 \
    python -m unittest test.integration.processors.test_lam_audio2expression_avatar_processor.TestLamAudio2ExpressionProcessor.test_gen

WEIGHT_PATH=./models/LAM_audio2exp/pretrained_models/lam_audio2exp_streaming.tar \
    WAV2VEC_DIR=./models/facebook/wav2vec2-base-960h \
    SLEEP_TO_END_TIME_S=7 \
    python -m unittest test.integration.processors.test_lam_audio2expression_avatar_processor.TestLamAudio2ExpressionFastApiWebsocketProcessor.test_gen
"""


class TestLamAudio2ExpressionBaseProcessor(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/asr/test_audio/asr_example_zh.wav
        audio_file = os.path.join(TEST_DIR, "audio_files/asr_example_zh.wav")
        cls.audio_file = os.getenv("AUDIO_FILE", audio_file)
        cls.data_bytes, cls.sr = read_wav_to_bytes(cls.audio_file)

        weight_path = os.path.join(
            MODELS_DIR, "LAM_audio2exp/pretrained_models/lam_audio2exp_streaming.tar"
        )
        cls.weight_path = os.getenv("WEIGHT_PATH", weight_path)

        wav2vec_dir = os.path.join(MODELS_DIR, "facebook/wav2vec2-base-960h")
        cls.wav2vec_dir = os.getenv("WAV2VEC_DIR", wav2vec_dir)

        cls.sleep_to_end_time_s = int(os.getenv("SLEEP_TO_END_TIME_S", "12"))

        cls.avatar_processor = None
        cls.task = None
        cls.runner = None
        cls.end_task: asyncio.Task = None

    @classmethod
    def tearDownClass(cls):
        pass

    async def out_cb(self, frame):
        # await asyncio.sleep(1)
        logging.info(f"sink_callback print frame: {frame}")

    async def end(self):
        await asyncio.sleep(self.sleep_to_end_time_s)
        await self.task.queue_frame(EndFrame())

    async def asyncSetUp(self):
        from src.processors.avatar.lam_audio2expression_avatar_processor import (
            LAMAudio2ExpressionAvatarProcessor,
        )
        from src.modules.avatar.lam_audio2expression import LAMAudio2ExpressionAvatar

        avatar = LAMAudio2ExpressionAvatar(
            **LAMAudio2ExpressionAvatarArgs(
                weight_path=self.weight_path,
                wav2vec_dir=self.wav2vec_dir,
                audio_sample_rate=self.sr,
            ).__dict__
        )
        self.avatar_processor = LAMAudio2ExpressionAvatarProcessor(avatar)

    async def asyncTearDown(self):
        self.end_task.cancel()
        await self.end_task

    async def test_gen(self):
        # ctrl + C to stop
        await self.task.queue_frames(
            [
                UserStartedSpeakingFrame(),
                AudioRawFrame(
                    audio=self.data_bytes,
                    sample_rate=self.sr,
                ),
                UserStoppedSpeakingFrame(),
                # EndFrame(),
            ]
        )
        await self.runner.run(self.task)


class TestLamAudio2ExpressionProcessor(TestLamAudio2ExpressionBaseProcessor):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.room_url = os.getenv("DAILY_ROOM_URL", "https://weedge.daily.co/chat-room")

    async def asyncSetUp(self):
        from src.transports.daily import DailyTransport
        from src.common.types import DailyParams

        await super().asyncSetUp()
        bot_name = "avatar-bot"
        transport = DailyTransport(
            self.room_url,
            None,
            bot_name,
            DailyParams(
                audio_out_enabled=True,
                audio_out_sample_rate=self.sr,
                camera_out_enabled=False,
                camera_out_width=1024,
                camera_out_height=1408,
                camera_out_is_live=True,
            ),
        )

        pipeline = Pipeline(
            [
                self.avatar_processor,
                # OutputFrameProcessor(cb=self.out_cb),
                transport.output_processor(),
            ]
        )

        self.task = PipelineTask(
            pipeline,
            PipelineParams(
                allow_interruptions=False,
                enable_metrics=False,
            ),
        )
        self.runner = PipelineRunner()

        self.end_task: asyncio.Task = asyncio.get_event_loop().create_task(self.end())


class TestLamAudio2ExpressionFastApiWebsocketProcessor(TestLamAudio2ExpressionBaseProcessor):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.serializer = AvatarProtobufFrameSerializer()

        # ws server
        cls.host = "localhost"
        cls.port = os.environ.get("WEBSOCKET_PORT", 4321)
        cls.endpoint = f"ws://localhost:{cls.port}/"

        # Store websocket connection
        cls.ws_map: Dict[str, WebSocket] = {}

        # https://fastapi.tiangolo.com/advanced/events/#lifespan
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            yield  # Run app

            # app life end to clear resources
            # clear websocket connection
            coros = [
                ws.close()
                for ws in cls.ws_map.values()
                if ws.client_state == WebSocketState.CONNECTED
            ]
            await asyncio.gather(*coros)
            cls.ws_map.clear()
            print(f"websocket connections clear success")

        cls.app = FastAPI(lifespan=lifespan)

        cls.server_task: asyncio.Task = None

    async def asyncTearDown(self):
        self.end_task.cancel()
        await self.end_task
        self.server_task.cancel()
        await self.server_task

    async def asyncSetUp(self):
        await super().asyncSetUp()

        params = FastapiWebsocketServerParams(
            audio_in_enabled=False,
            audio_out_enabled=True,
            # audio_frame_size=6400,  # output 200ms with 16K hz 1 channel 2 sample_width
            serializer=self.serializer,  # audio_expression frame serializer
        )

        @self.app.websocket("/")
        async def websocket_endpoint(websocket: WebSocket):
            try:
                await websocket.accept()
                key = f"{websocket.client.host}:{websocket.client.port}"
                self.ws_map[key] = websocket
                logging.info(f"accept client: {websocket.client}")
                # just for one test session
                transport = FastapiWebsocketTransport(
                    websocket=websocket,
                    params=params,
                )

                pipeline = Pipeline(
                    [
                        self.avatar_processor,
                        # OutputFrameProcessor(cb=self.out_cb),
                        transport.output_processor(),
                    ]
                )

                self.task = PipelineTask(
                    pipeline,
                    PipelineParams(
                        allow_interruptions=False,
                        enable_metrics=False,
                    ),
                )
                self.runner = PipelineRunner()
                self.end_task = asyncio.get_event_loop().create_task(self.end())
                await self.task.queue_frames(
                    [
                        UserStartedSpeakingFrame(),
                        AudioRawFrame(
                            audio=self.data_bytes,
                            sample_rate=self.sr,
                        ),
                        UserStoppedSpeakingFrame(),
                        # EndFrame(),
                    ]
                )
                await self.runner.run(self.task)
                logging.info(f"{self.task.name} is DONE!!")
                await self.ws_map[key].close()
            except Exception as e:
                logging.error(f"Failed to initialize pipeline: {e}", exc_info=True)

        self.server_task = asyncio.get_event_loop().create_task(self.run_server())

    async def run_server(self):
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
        )
        server = uvicorn.Server(config)
        try:
            await server.serve()
        except asyncio.CancelledError:
            logging.warning("Server task cancelled")

    async def connect2receive(self):
        await asyncio.sleep(1)  # sleep to wait websocket server is ok
        # start connect
        async with aiohttp.ClientSession() as session:
            try:
                async with session.ws_connect(self.endpoint) as ws:
                    logging.info("Connection established.")
                    # start receive msg
                    while True:
                        msg = await ws.receive()
                        print(f"{msg.type=}")
                        if msg.type in [
                            aiohttp.WSMsgType.CLOSE,
                            aiohttp.WSMsgType.CLOSING,
                            aiohttp.WSMsgType.CLOSED,
                        ]:
                            logging.info("Connection closed by server.")
                            break
                        if msg.type in [
                            aiohttp.WSMsgType.ERROR,
                        ]:
                            logging.error("Connection error by server.")
                            break
                        msg_bytes = msg.data
                        assert isinstance(msg_bytes, bytes)
                        assert len(msg_bytes) > 0
                        # receive frame
                        frame = self.serializer.deserialize(msg_bytes)
                        assert frame
                    await ws.close()
                    logging.info(f"---{ws} Connection closed---")
            except aiohttp.ClientError as e:
                logging.error(f"Connection error: {e}")
            except Exception as e:
                logging.error(f"Error in connect2receive: {e}")

    async def test_gen(self):
        await self.connect2receive()
