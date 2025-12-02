from .base import ImageGenProcessor


def get_video_gen_processor(tag, **kwargs) -> ImageGenProcessor:
    if tag == "ComfyUIAPIVideoGenProcessor":
        from .comfyui_video_gen_processor import ComfyUIAPIVideoGenProcessor

        return ComfyUIAPIVideoGenProcessor(**kwargs)

    return None
