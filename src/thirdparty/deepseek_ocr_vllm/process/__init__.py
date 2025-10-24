import logging

from PIL import Image, ImageOps, ImageDraw, ImageFont


def load_image(image_path) -> Image.Image:
    try:
        image = image_path
        if not isinstance(image_path, Image.Image):
            image = Image.open(image_path)
        corrected_image = ImageOps.exif_transpose(image)
        return corrected_image
    except Exception as e:
        logging.error(f"error: {e}")
        try:
            return Image.open(image_path)
        except Exception:
            return None
