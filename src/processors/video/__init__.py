from .base import VideoGenProcessor


def get_video_gen_processor(tag, **kwargs) -> VideoGenProcessor:
    if tag == "ComfyUIAPIVideoGenProcessor":
        from .comfyui_video_gen_processor import ComfyUIAPIVideoGenProcessor

        return ComfyUIAPIVideoGenProcessor(**kwargs)

    return None
