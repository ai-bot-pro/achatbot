import modal
import os


class ContainerRuntimeConfig:
    images = {
        "default": (
            modal.Image.debian_slim(python_version="3.11")
            .apt_install("git", "git-lfs", "ffmpeg")
            .pip_install(
                [
                    "achatbot["
                    "fastapi_bot_server,"
                    "websocket_server_transport,"
                    "silero_vad_analyzer,"
                    "moshi_voice_processor"
                    "]~=0.0.7.10",
                    "huggingface_hub[hf_transfer]==0.24.7",
                ],
                extra_index_url="https://pypi.org/simple/")
            .env({
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "ACHATBOT_PKG": "1",
                "BOT_CONFIG_NAME": os.getenv("BOT_CONFIG_NAME",
                                             "fastapi_websocket_moshi_voice_bot"),
                "LOG_LEVEL": os.getenv("LOG_LEVEL", "info"),
                "IMAGE_NAME": os.getenv("IMAGE_NAME", "default"),
            })
        ),
    }

    @staticmethod
    def get_img(image_name: str = None):
        image_name = image_name or os.getenv("IMAGE_NAME", "default")
        if image_name not in ContainerRuntimeConfig.images:
            raise Exception(f"image name {image_name} not found")
        print(f"use image:{image_name}")
        return ContainerRuntimeConfig.images[image_name]


img = ContainerRuntimeConfig.get_img()
with img.imports():
    import asyncio
    import logging
    import pathlib
    import os

    from fastapi import WebSocket
    from huggingface_hub import hf_hub_download
    from moshi.models import loaders

    from achatbot.cmd.bots.base_fastapi_websocket_server import AIFastapiWebsocketBot
    from achatbot.cmd.http.server.fastapi_daily_bot_serve import app as fastapi_app
    from achatbot.cmd.bots.bot_loader import BotLoader
    from achatbot.common.logger import Logger

# ----------------------- app -------------------------------
app = modal.App("fastapi_ws_moshi_voice_bot")

volume = modal.Volume.from_name("bot_config", create_if_missing=True)


# 128 MiB of memory and 0.125 CPU cores by default container runtime
@app.cls(
    image=img,
    # secrets=[modal.Secret.from_name("achatbot")],
    volumes={"/bots": volume},
    gpu="A10G",
    container_idle_timeout=1200,
    timeout=600,
    allow_concurrent_inputs=1,
)
class Srv:
    @modal.build()
    def setup(self):
        Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)
        # https://huggingface.co/docs/huggingface_hub/guides/download
        repo_id = loaders.DEFAULT_REPO
        bot_config_name = os.getenv("BOT_CONFIG_NAME", "")
        if "moshika" in bot_config_name:
            repo_id = "kyutai/moshika-pytorch-bf16"
        hf_hub_download(repo_id, loaders.MOSHI_NAME)
        hf_hub_download(repo_id, loaders.MIMI_NAME)
        hf_hub_download(repo_id, loaders.TEXT_TOKENIZER_NAME)
        logging.info(f"download model done, config name:{bot_config_name}, hf_repo:{repo_id}")

    @modal.enter()
    def enter(self):
        volume.reload()
        bot_config_name = os.getenv("BOT_CONFIG_NAME", "")
        self.config_path = pathlib.Path(os.path.join("/bots", f"{bot_config_name}.json"))
        # self.run_bot: AIFastapiWebsocketBot = asyncio.run(
        #    BotLoader.load_bot(self.config_path, bot_type="fastapi_ws_bot"))
        logging.info(f"start enter get run_bot from conifg:{self.config_path}")

    @modal.asgi_app()
    def app(self):

        @fastapi_app.websocket("/")
        async def websocket_endpoint(websocket: WebSocket):
            self.run_bot: AIFastapiWebsocketBot = await BotLoader.load_bot(  # type: ignore
                self.config_path, bot_type="fastapi_ws_bot")

            # NOTE: after init, websocket to accept connection, then to run
            await websocket.accept()
            self.run_bot.set_fastapi_websocket(websocket)
            logging.info(f"accept client: {websocket.client}")
            await self.run_bot.try_run()

        return fastapi_app
