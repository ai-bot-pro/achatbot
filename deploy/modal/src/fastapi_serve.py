import modal
import os


class ContainerRuntimeConfig:
    images = {
        "default": (
            modal.Image.debian_slim(python_version="3.11")
            .pip_install(
                "achatbot[fastapi_bot_server]~=0.0.7.10", extra_index_url="https://pypi.org/simple/"
            )
            .apt_install()
            .env({})
        ),
    }

    @staticmethod
    def get_img(image_name: str = None):
        image_name = image_name or os.getenv("IMAGE_NAME", "default")
        if image_name not in ContainerRuntimeConfig.images:
            raise Exception(f"image name {image_name} not found")
        print(f"use image:{image_name}")
        return ContainerRuntimeConfig.images[image_name]


# ----------------------- app -------------------------------
app = modal.App("fastapi")


# 128 MiB of memory and 0.125 CPU cores by default
@app.cls(
    image=ContainerRuntimeConfig.get_img(),
    # cpu=2.0,
    scaledown_window=300,
    timeout=600,
    allow_concurrent_inputs=1,
)
class Srv:
    @modal.build()
    def download_model(self):
        print("start downloading model")

    @modal.enter()
    def enter(self):
        print("start entrer")

    @modal.asgi_app()
    def app(self):
        from achatbot.cmd.http.server.fastapi_daily_bot_serve import app

        return app


if __name__ == "__main__":
    pass
