import logging
import os
import io
from PIL import Image

import typer

from demo.aws.upload import r2_upload

app = typer.Typer()


@app.command("compress_img")
def compress_img(
    img_path: str,
    file_name: str = "",
    save_dir: str = "./images",
    quality: int = 60,
):
    """Compresses a single image.

    Args:
        img_path: Path to the input image.
        file_name: Desired file name for the compressed image.
        save_dir: Directory to save the compressed image.
        quality: Compression quality (0-100).
    """
    if not os.path.isfile(img_path):
        raise Exception(f"image not found: {img_path}")

    f = os.path.basename(img_path).split(".")
    ext = f".{f[-1]}"
    if not file_name:
        file_name = f[0] + f"_{quality}"
    save_name = os.path.join(save_dir, file_name)
    img = Image.open(img_path)
    if img.format == "PNG":
        img = img.convert("RGB")
        ext = ".jpg"
    else:
        img = img.quantize(colors=256)
    save_path = save_name + ext
    img.save(save_path, optimize=True, quality=quality, subsampling=-1)
    return save_path


@app.command("compress_imgs")
def compress_imgs(
    img_dir: str = "./images",
    save_dir: str = "images",
    quality: int = 60,
):
    """Compresses and uploads all JPG images in a directory to R2.

    Args:
        img_dir: Directory containing the images to compress and upload.
        remote_folder: Remote folder in R2 to upload to.
        quality: Compression quality (0-100).
    """
    if not os.path.isdir(img_dir) and not os.path.exists(save_dir):
        raise Exception(f"image directory not found: {img_dir} or {save_dir}")
    for file_name in os.listdir(img_dir):
        if file_name.endswith(".png"):
            img_path = os.path.join(img_dir, file_name)
            save_path = compress_img(
                img_path=img_path,
                save_dir=img_dir,
                quality=quality,
            )
            logging.info(f"compress {img_path} to {save_path}")
        else:
            logging.info(f"skip {file_name}")


@app.command("upload_imgs")
def upload_imgs(img_dir: str = "./images", remote_folder: str = "podcast"):
    """Upload all JPG images in a directory to R2.

    Args:
        img_dir: Directory containing the images to compress and upload.
        remote_folder: Remote folder in R2 to upload to.
    """
    if not os.path.isdir(img_dir):
        raise Exception(f"image directory not found: {img_dir}")
    for file_name in os.listdir(img_dir):
        if file_name.endswith(".jpg"):
            img_path = os.path.join(img_dir, file_name)
            # Upload the image
            r2_url = r2_upload(remote_folder=remote_folder, local_file=img_path)
            logging.info(f"Uploaded {img_path} to R2: {r2_url}")
        else:
            logging.info(f"skip {file_name}")


r"""
python -m demo.image_compression compress_img images/0_123.png
python -m demo.image_compression compress_img images/0_123.png --quality 60

python -m demo.image_compression compress_and_upload_imgs
python -m demo.image_compression upload_imgs
"""
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(funcName)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    app()
