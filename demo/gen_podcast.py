import os
import re
import logging
import asyncio
from datetime import datetime
from typing import Generator, List, Union
from urllib.parse import urlparse

import typer

from demo.content_parser.table import podcast
from demo.content_parser_tts import instruct_content_tts
from demo.insert_podcast import insert_podcast_to_d1


app = typer.Typer()


def is_url(source: str) -> bool:
    try:
        # If the source doesn't start with a scheme, add 'https://'
        if not source.startswith(("http://", "https://")):
            source = "https://" + source

        result = urlparse(source)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def is_file(source: str) -> bool:
    return os.path.isfile(source)


@app.command("get_source_type")
def get_source_type(source: str):
    src_type = ""
    isUrl = is_url(source)
    if isUrl:
        if any(pattern in source for pattern in ["youtube.com", "youtu.be"]):
            src_type = "youtube"
        else:
            src_type = "website"
    elif is_file(source):
        sp = source.split(".")
        src_type = sp[-1] if len(sp) > 0 else "file"

    return src_type


@app.command("run")
def run(
    sources: List[str],
    role_tts_voices: List[str] = ["en-US-JennyNeural", "en-US-EricNeural"],
    language: str = "en",
    save_dir: str = "./audios/podcast",
    category: int = 0,
    is_published: bool = False,
):
    data_list = instruct_content_tts(
        sources,
        role_tts_voices=role_tts_voices,
        language=language,
        save_dir=save_dir,
    )
    for data in data_list:
        source = data[0]
        extraction: podcast.Podcast = data[1]
        audio_output_file = data[2]

        speakers = podcast.speakers(extraction, role_tts_voices)
        logging.info(f"speakers:{speakers}")
        audio_content = podcast.content(extraction, "text")
        logging.info(f"audio_content:{audio_content}")
        insert_podcast_to_d1(
            audio_output_file,
            extraction.title,
            "weedge",
            ",".join(speakers),
            description=extraction.description,
            audio_content=audio_content,
            is_published=is_published,
            category=category,
            # source=get_source_type(source),
            source=source,
            # pid="",
            language=language,
        )


r"""
python -m demo.gen_podcast run \
    "https://en.wikipedia.org/wiki/Large_language_model"

python -m demo.gen_podcast run \
    --role-tts-voices zh-CN-XiaoxiaoNeural \
    --role-tts-voices zh-CN-YunjianNeural \
    --category 1 \
    --language zh \
    --is-published \
    "https://www.youtube.com/watch?v=aR6CzM0x-g0" \
    "https://en.wikipedia.org/wiki/Large_language_model" \
    "/Users/wuyong/Desktop/iOS_18_All_New_Features_Sept_2024.pdf"

"""
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(funcName)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    app()
