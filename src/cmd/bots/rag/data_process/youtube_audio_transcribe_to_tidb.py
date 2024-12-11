from datetime import datetime
import math
import re
import logging
import os
import time
import subprocess
from typing import List

from deep_translator import GoogleTranslator
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
import httpx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import TiDBVectorStore
from langchain_community.embeddings import JinaEmbeddings

from .pytube_monkey_fix import YouTube
from src.common.types import VIDEOS_DIR
from src.common.logger import Logger
from src.cmd.bots.rag.helper import get_tidb_url


# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv(override=True)


def time_str_to_seconds(time_str):
    time_format = "%H:%M:%S.%f"
    dt = datetime.strptime(time_str, time_format)
    total_seconds = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6

    return total_seconds


def copy_video_to_filter(video_file):
    ffmpeg_command = ["ffmpeg", "-i", video_file, "-c", "copy", f"{video_file}_tmp.mp4"]
    ffmpeg_process = subprocess.Popen(ffmpeg_command)
    ffmpeg_process.wait()
    mv_command = ["mv", f"{video_file}_tmp.mp4", video_file]
    mv_process = subprocess.Popen(mv_command)
    mv_process.wait()


def get_video_duration(video_file):
    ffmpeg_command = ["ffmpeg", "-i", video_file]
    grep_command = ["grep", "Duration"]
    cut_command = ["cut", "-d", " ", "-f", "4"]
    sed_command = ["sed", "s/,//"]

    ffmpeg_process = subprocess.Popen(
        ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    grep_process = subprocess.Popen(
        grep_command, stdin=ffmpeg_process.stdout, stdout=subprocess.PIPE
    )
    ffmpeg_process.stdout.close()
    cut_process = subprocess.Popen(cut_command, stdin=grep_process.stdout, stdout=subprocess.PIPE)
    grep_process.stdout.close()
    sed_process = subprocess.Popen(sed_command, stdin=cut_process.stdout, stdout=subprocess.PIPE)
    cut_process.stdout.close()

    duration = sed_process.communicate()[0].decode("utf-8").strip()

    return duration


def download_videos(link: str, download_path: str):
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    yt = YouTube(link)

    download_file_path = f"{download_path}/{yt.title}.mp4"
    if os.path.exists(download_file_path):
        logging.info(f"{download_file_path} is exists, skip download")
        return (
            yt.title,
            download_file_path,
        )

    logging.info(f"{download_file_path} Downloading Audio...")
    audio_download = yt.streams.get_audio_only()
    audio_download.download(
        filename=f"{yt.title}.mp4",
        output_path=download_path,
    )

    return (
        yt.title,
        download_file_path,
    )


def splite_chunk_videos(video_file: str, ss: str, duration: float):
    duration = get_video_duration(video_file)
    d_time = time_str_to_seconds(duration)
    if d_time / 3600 > 0:
        #!TODO
        pass


def transcribe_file(audio_file: str, text_file_name: str = "", path: str = VIDEOS_DIR):
    """
    see: https://developers.deepgram.com/docs/getting-started-with-pre-recorded-audio
    """

    text_file_path = f"{path}/{text_file_name}.txt"
    transcribe_respone_file_path = f"{path}/{text_file_name}.json"
    text_done_file_path = f"{path}/{text_file_name}_done.txt"
    if os.path.exists(text_done_file_path) and os.path.exists(text_file_path):
        logging.info(f"{text_file_path} is exists, skip transcribe, read from file")
        with open(text_file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return content

    try:
        with open(audio_file, "rb") as file:
            buffer_data = file.read()
        payload: FileSource = {
            "buffer": buffer_data,
        }

        # Create a Deepgram client using the API key
        deepgram = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))
        logging.info(f"transcribing {audio_file} save to text file {text_file_path}")
        # Configure Deepgram options for audio analysis
        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
            language="en",
        )
        # Call the transcribe_file method with the text payload and options
        response = deepgram.listen.rest.v("1").transcribe_file(
            payload,
            options,
            timeout=httpx.Timeout(1800.0, connect=10.0),
        )
        transcript = response.results.channels[0].alternatives[0].transcript
        logging.info(f"transcribing {audio_file} transcript text len {len(transcript)}")

        if len(text_file_name) > 0:
            with open(text_file_path, "w", encoding="utf-8") as file:
                file.write(transcript)
            with open(transcribe_respone_file_path, "w", encoding="utf-8") as file:
                file.write(response.to_json())
            with open(text_done_file_path, "a"):
                os.utime(text_done_file_path, None)

        return transcript

    except Exception as e:
        logging.error(f"Exception: {e}")
        raise e


def translate_text(
    text: str,
    src: str = "en",
    target: str = "zh-CN",
    text_file_name: str = "",
    path: str = VIDEOS_DIR,
):
    text_file_path = f"{path}/{text_file_name}_{target}.txt"
    if os.path.exists(text_file_path):
        logging.info(f"{text_file_path} is exists, skip translate, read from file")
        with open(text_file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return content

    try_cn = 10
    translated_texts = []
    pattern = "|".join(map(re.escape, ["!", ".", "！", "。"]))
    result = list(filter(None, re.split("(" + pattern + ")", text, maxsplit=0, flags=re.UNICODE)))
    sentences = []
    for i in range(0, len(result) - 1, 2):
        sentences.append(result[i] + result[i + 1])
    if len(result) % 2 != 0:
        sentences.append(result[-1])
    logging.info(f"translate sentences len: {len(sentences)}")
    while try_cn > 0:
        try:
            translated_texts = GoogleTranslator(source=src, target=target).translate_batch(
                sentences
            )
            break
        except Exception as e:
            logging.error("An error occurred:", e)
            time.sleep(1)
            try_cn -= 1

    res = "".join(translated_texts)
    if len(text_file_name) > 0 and len(translated_texts) > 0:
        with open(text_file_path, "w", encoding="utf-8") as file:
            file.write(res)

    return res


def split_to_chunk_texts(text: str, lang: str = "en"):
    # !NOTE: maybe don't to check split to chunk doc task done
    # just simple split by len
    # see: https://chunkviz.up.railway.app/
    logging.info(f"Spliting text len:{len(text)}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    split_text_list = text_splitter.split_text(text)
    logging.debug(f"splited chunk text list len {len(split_text_list)}")
    return split_text_list


def save_embeddings_to_db(
    table_name: str,
    title: str,
    texts: list[str],
    path: str = VIDEOS_DIR,
    lang: str = "en",
    link: str = "",
):
    # just a simple done, need use task meta info to manage
    save_done_file_path = f"{path}/{title}_{table_name}_save_done.txt"
    if os.path.exists(save_done_file_path):
        logging.info(f"{save_done_file_path} is exists, skip save embeddings to table {table_name}")
        return

    logging.info(f"saving embeddings to db table_name:{table_name} texts len:{len(texts)}")
    # https://jina.ai/embeddings/
    # !TODO: use hf embeddings model to gen embeddings or other api model
    model_name = "jina-embeddings-v2-base-en"
    if lang == "zh":
        model_name = "jina-embeddings-v2-base-zh"
    embeddings = JinaEmbeddings(
        jina_api_key=os.getenv("JINA_API_KEY"),
        model_name=model_name,
    )

    # https://docs.pingcap.com/tidbcloud/vector-search-integrate-with-langchain
    # Connect to tidb table vector index and insert the chunked docs as contents
    # !NOTE limitations: https://docs.pingcap.com/tidb/stable/tidb-limitations
    url = get_tidb_url()
    metadatas: List[dict] = []
    for text in texts:
        metadatas.append(
            {
                "text_len": len(text),
                "lang": lang,
                "embeddings_model": model_name,
                "text_title": title,
                "link": link,
            }
        )
    _ = TiDBVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        table_name=table_name,
        connection_string=url,
        metadatas=metadatas,
        # default, another option is "l2"
        distance_strategy=os.getenv("TIDB_VSS_DISTANCE_STRATEGY", "cosine"),
    )

    with open(save_done_file_path, "a"):
        os.utime(save_done_file_path, None)


if __name__ == "__main__":
    """
    python -m src.cmd.bots.rag.data_process.youtube_audio_transcribe_to_tidb
    !TODO:
    - define pipelie task meta info save to tidb table rag_task_info
    - check save embedding to db task meta info
    have a good day! :)
    """

    Logger.init(level=logging.DEBUG, is_file=False)

    video_links = {
        "ThreeBlueOneBrown": [
            # NN
            "https://www.youtube.com/watch?v=aircAruvnKk",
            "https://www.youtube.com/watch?v=IHZwWFHWa-w",
            "https://www.youtube.com/watch?v=Ilg3gGewQ5U",
            "https://www.youtube.com/watch?v=tIeHLnjs5U8",
            "https://www.youtube.com/watch?v=wjZofJX0v4M",
            "https://www.youtube.com/watch?v=eMlx5fFNoYc",
        ],
        "AndrejKarpathy": [
            "https://www.youtube.com/watch?v=zjkBMFhNj_g",
            # zero2hero
            "https://www.youtube.com/watch?v=VMj-3S1tku0",
            "https://www.youtube.com/watch?v=PaCmpygFfXo",
            "https://www.youtube.com/watch?v=TCH_1BHY58I",
            "https://www.youtube.com/watch?v=P6sfmUTpUmc",
            "https://www.youtube.com/watch?v=q8SA3rM6ckI",
            "https://www.youtube.com/watch?v=t3YJ5hKiMQ0",
            "https://www.youtube.com/watch?v=kCc8FmEb1nY",
            "https://www.youtube.com/watch?v=bZQun8Y4L2A",
            "https://www.youtube.com/watch?v=zduSFxRajkE",
            "https://www.youtube.com/watch?v=l8pRSuU81PU",
        ],
    }

    # !TODO: async currency task run
    for name, links in video_links.items():
        for link in links:
            try:
                dir_path = f"{VIDEOS_DIR}/{name}"
                title, download_file_path = download_videos(link, dir_path)
                copy_video_to_filter(download_file_path)
                transcribed_text = transcribe_file(download_file_path, title, dir_path)
                chunk_texts = split_to_chunk_texts(transcribed_text, lang="en")
                save_embeddings_to_db(
                    name,
                    title=title,
                    texts=chunk_texts,
                    path=dir_path,
                    lang="en",
                    link=link,
                )

                translated_text = translate_text(
                    transcribed_text, target="zh-CN", text_file_name=title, path=dir_path
                )
                chunk_texts = split_to_chunk_texts(translated_text, lang="zh")
                save_embeddings_to_db(
                    name,
                    title=title + " zh-CN",
                    texts=chunk_texts,
                    path=dir_path,
                    lang="zh",
                    link=link,
                )
            except Exception as e:
                logging.exception(f"name:{name}, link:{link}, Exception: {e}")
                continue
