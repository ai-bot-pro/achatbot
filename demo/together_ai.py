import base64
import io
import os
import logging
from typing import List

import typer
from PIL import Image
from together import Together
from dotenv import load_dotenv


client = Together()

# Load environment variables from .env file
load_dotenv(override=True)

app = typer.Typer()


@app.command("gen_image")
def gen_image(
    prompt: str,
    width: int = 640,
    height: int = 480,
    steps: int = 4,
    n: int = 1,
) -> List[str]:
    response = client.images.generate(
        model="black-forest-labs/FLUX.1-schnell-Free",
        prompt=prompt,
        width=width,
        height=height,
        steps=steps,
        n=n,
        response_format="b64_json",
    )

    res = []
    if len(response.data) > 0:
        res.append(response.data[0].b64_json)

    return res


@app.command("save_gen_image")
def save_gen_image(
    prompt: str,
    file_name: str,
    save_dir: str = "./images",
    width: int = 640,
    height: int = 480,
    n: int = 1,
):
    base64_imgs = gen_image(prompt, width=width, height=height, steps=4, n=n)
    if not base64_imgs:
        raise Exception("no gen image")
    i = 0
    for base64_img in base64_imgs:
        img_bytes = io.BytesIO(base64.b64decode(base64_img))
        img = Image.open(img_bytes)
        save_path = os.path.join(save_dir, f"{i}_{file_name}.png")
        # img.show()
        img.save(save_path, "PNG")
        return save_path


r"""
python -m demo.image_together_flux gen_image "llama, sitting in a field of flowers"
python -m demo.image_together_flux save_gen_image "llama, sitting in a field of flowers" 123 --n 2
"""
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(funcName)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    app()
