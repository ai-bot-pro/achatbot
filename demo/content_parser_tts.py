import os
import re
import logging
import asyncio
from datetime import datetime
import shutil
from typing import Generator, List, Union

from pydub import AudioSegment
import edge_tts
import typer

from .content_parser.content_extractor_instructor import ContentExtractor
from .content_parser.table import podcast

app = typer.Typer()


async def edge_tts_conversion(text_chunk: str, output_file: str, voice: str):
    webvtt_file = ".".join(output_file.split(".")[:-1]) + ".vtt"
    communicate = edge_tts.Communicate(text_chunk, voice, rate="+15%")
    submaker = edge_tts.SubMaker()
    with open(output_file, "wb") as file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                # print(chunk)
                submaker.create_sub((chunk["offset"], chunk["duration"]), chunk["text"])

    with open(webvtt_file, "w", encoding="utf-8") as file:
        file.write(submaker.generate_subs())


async def gen_role_tts_audios(
    data_models: Generator[podcast.Role, None, None],
    save_dir: str,
    role_tts_voices: List[str],
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    i = 0
    for role in data_models:
        if not role.content:
            continue
        output_file = os.path.join(save_dir, f"{i}_{role.name}.mp3")
        voice = role_tts_voices[i % len(role_tts_voices)]
        print(f"{i}. {role.name}: {role.content} speaker:{voice} \n")
        i += 1
        await edge_tts_conversion(
            role.content,
            output_file,
            voice,
        )
    return 1 if i > 0 else 0


async def gen_podcast_tts_audios(
    data_models: Generator[podcast.Podcast, None, None],
    save_dir: str,
    role_tts_voices: List[str],
):
    podcast_index, role_index = (0, 0)
    pre_role = ""
    pre_cn, cur_cn = (0, 0)
    for extraction in data_models:
        if not extraction.roles:
            continue
        p_save_dir = os.path.join(save_dir, str(podcast_index))
        if not os.path.exists(p_save_dir):
            os.makedirs(p_save_dir)
        # print(f"----------podcast {podcast_index}----{len(extraction.roles)}---------")

        pre_cn = cur_cn
        cur_cn = len(extraction.roles)
        if pre_cn == cur_cn:
            # print(f"pre_cn == cur_cn :{pre_cn} continue")
            continue
        if pre_cn > cur_cn:
            # just use the first podcast, break
            break
            podcast_index += 1
            pre_cn = 1
            role_index = 0

        # print(pre_cn, extraction.roles, extraction.roles[pre_cn - 1:])
        for role in extraction.roles[pre_cn - 1 :]:
            if not role.content:
                continue
            if pre_role == role.name:
                logging.warning(f"duplicate {role.name}: {role.content}")
                # remove pre tts audio content
                pre_audio_file = os.path.join(p_save_dir, f"{role_index-1}_{role.name}.mp3")
                if os.path.exists(pre_audio_file):
                    os.remove(pre_audio_file)
                    # logging.warning(f"remove {pre_audio_file}")
                pre_vtt_file = os.path.join(p_save_dir, f"{role_index-1}_{role.name}.vtt")
                if os.path.exists(pre_vtt_file):
                    os.remove(pre_vtt_file)
                    # logging.warning(f"remove {pre_vtt_file}")
                role_index -= 1

            output_file = os.path.join(p_save_dir, f"{role_index}_{role.name}.mp3")
            voice = role_tts_voices[role_index % len(role_tts_voices)]
            print(f"{role_index}. {role.name}: {role.content} speaker:{voice} \n")
            pre_role = role.name
            role_index += 1
            await edge_tts_conversion(role.content, output_file, voice)

    return extraction


@app.command("merge_audio_files")
def merge_audio_files(input_dir: str, output_file: str) -> None:
    try:
        # Function to sort filenames naturally
        def natural_sort_key(filename: str) -> List[Union[int, str]]:
            return [int(text) if text.isdigit() else text for text in re.split(r"(\d+)", filename)]

        combined = AudioSegment.empty()
        audio_files = sorted(
            [f for f in os.listdir(input_dir) if f.endswith(".mp3")], key=natural_sort_key
        )
        logging.info(f"sorted audio_files: {audio_files}")
        for file in audio_files:
            file_path = os.path.join(input_dir, file)
            combined += AudioSegment.from_file(file_path, format="mp3")

        combined.export(output_file, format="mp3")
        logging.info(f"Merged audio saved to {output_file}")
    except Exception as e:
        logging.error(f"Error merging audio files: {str(e)}")
        raise


@app.command()
def instruct_role_tts(
    content: str,
    tmp_dir: str,
    role_tts_voices: List[str] = ["en-US-JennyNeural", "en-US-EricNeural"],
    language: str = "en",
):
    data_models = podcast.extract_role_models_iterable(content, language=language)
    return asyncio.run(gen_role_tts_audios(data_models, tmp_dir, role_tts_voices))


@app.command()
def instruct_podcast_tts(
    content: str,
    tmp_dir: str,
    role_tts_voices: List[str] = ["en-US-JennyNeural", "en-US-EricNeural"],
    language: str = "en",
):
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    data_models = podcast.extract_models(content, language=language)
    return asyncio.run(gen_podcast_tts_audios(data_models, tmp_dir, role_tts_voices))


@app.command()
def instruct_content_tts(
    sources: List[str],
    role_tts_voices: List[str] = ["en-US-JennyNeural", "en-US-EricNeural"],
    language: str = "en",
    save_dir: str = "./audios/podcast",
) -> list:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    extractor = ContentExtractor()
    res = []
    for source in sources:
        try:
            content = extractor.extract_content(source)
            logging.info(f"{source} extracted content done")
            now = datetime.now()
            formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
            output_file = os.path.join(save_dir, f"{extractor.file_name}_{formatted_time}.mp3")
            tmp_dir = os.path.join(save_dir, extractor.file_name)
            extraction = instruct_podcast_tts(content, tmp_dir, role_tts_voices, language)
            p_tmp_dir = os.path.join(tmp_dir, "0")
            merge_audio_files(input_dir=p_tmp_dir, output_file=output_file)
            res.append((source, extraction, output_file))
        except Exception as e:
            logging.error(f"An error occurred while processing {source}: {str(e)}", exc_info=True)

    return res


r"""
python -m demo.content_parser_tts instruct-content-tts \
    "https://en.wikipedia.org/wiki/Large_language_model"

python -m demo.content_parser_tts instruct-content-tts \
    --role-tts-voices zh-CN-YunjianNeural \
    --role-tts-voices zh-CN-XiaoxiaoNeural \
    --language zh \
    "https://en.wikipedia.org/wiki/Large_language_model" \
    "https://www.youtube.com/watch?v=aR6CzM0x-g0" \
    "/Users/wuyong/Desktop/iOS_18_All_New_Features_Sept_2024.pdf"

python -m demo.content_parser_tts merge_audio_files \
    audios/podcast/2401.02669/0  audios/podcast/2401.02669.mp3
"""
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(funcName)s - %(message)s",
        handlers=[
            # logging.FileHandler("content_parser_tts.log"),
            logging.StreamHandler()
        ],
    )
    app()
