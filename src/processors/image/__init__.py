from .base import ImageGenProcessor


def get_image_gen_processor(tag, **kwargs) -> ImageGenProcessor:
    if tag == "HFApiInferenceImageGenProcessor":
        from .hf_img_gen_processor import HFApiInferenceImageGenProcessor

        return HFApiInferenceImageGenProcessor(**kwargs)
    if tag == "OpenAIImageGenProcessor":
        from .openai_img_gen_processor import OpenAIImageGenProcessor

        return OpenAIImageGenProcessor(**kwargs)
    if tag == "TogetherImageGenProcessor":
        from .together_img_gen_processor import TogetherImageGenProcessor

        return TogetherImageGenProcessor(**kwargs)
    if tag == "HFStableDiffusionImageGenProcessor":
        from .diffusers_img_gen_processor import HFStableDiffusionImageGenProcessor

        return HFStableDiffusionImageGenProcessor(**kwargs)
