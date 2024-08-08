import re
import logging
import os

from pytube import YouTube, cipher
from pytube.innertube import _default_clients

from deepgram import DeepgramClient, PrerecordedOptions, FileSource
import httpx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import TiDBVectorStore
from langchain_community.embeddings import JinaEmbeddings

from src.common.types import VIDEOS_DIR
from src.common.logger import Logger


# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(override=True)


# issue: https://github.com/pytube/pytube/issues/1973
_default_clients["ANDROID"]["context"]["client"]["clientVersion"] = "19.08.35"
_default_clients["IOS"]["context"]["client"]["clientVersion"] = "19.08.35"
_default_clients["ANDROID_EMBED"]["context"]["client"]["clientVersion"] = "19.08.35"
_default_clients["IOS_EMBED"]["context"]["client"]["clientVersion"] = "19.08.35"
_default_clients["IOS_MUSIC"]["context"]["client"]["clientVersion"] = "6.41"
_default_clients["ANDROID_MUSIC"] = _default_clients["ANDROID_CREATOR"]


class RegexMatchError(Exception):
    pass


def get_throttling_function_name(js: str) -> str:
    """Extract the name of the function that computes the throttling parameter.

    :param str js:
        The contents of the base.js asset file.
    :rtype: str
    :returns:
        The name of the function used to compute the throttling parameter.
    """
    function_patterns = [
        r'a\.[a-zA-Z]\s*&&\s*\([a-z]\s*=\s*a\.get\("n"\)\)\s*&&\s*'
        r'\([a-z]\s*=\s*([a-zA-Z0-9$]+)(\[\d+\])?\([a-z]\)',
        r'\([a-z]\s*=\s*([a-zA-Z0-9$]+)(\[\d+\])\([a-z]\)',
    ]
    # logger.debug('Finding throttling function name')
    for pattern in function_patterns:
        regex = re.compile(pattern)
        function_match = regex.search(js)
        if function_match:
            # logger.debug("finished regex search, matched: %s", pattern)
            if len(function_match.groups()) == 1:
                return function_match.group(1)
            idx = function_match.group(2)
            if idx:
                idx = idx.strip("[]")
                array = re.search(
                    r'var {nfunc}\s*=\s*(\[.+?\]);'.format(
                        nfunc=re.escape(function_match.group(1))),
                    js
                )
                if array:
                    array = array.group(1).strip("[]").split(",")
                    array = [x.strip() for x in array]
                    return array[int(idx)]

    raise RegexMatchError(
        caller="get_throttling_function_name", pattern="multiple"
    )


cipher.get_throttling_function_name = get_throttling_function_name


def download_videos(link: str, download_path: str):

    if not os.path.exists(download_path):
        os.makedirs(download_path)

    yt = YouTube(link)

    download_file_path = f"{download_path}/{yt.title}.mp4"
    if os.path.exists(download_file_path):
        logging.info(f"{download_file_path} is exists, skip download")
        return yt.title, download_file_path,

    logging.info(f"{download_file_path} Downloading Audio...")
    audio_download = yt.streams.get_audio_only()
    audio_download.download(
        filename=f"{yt.title}.mp4",
        output_path=download_path,
    )

    return yt.title, download_file_path,


def transcribe_file(audio_file: str, text_file_name: str = ""):
    """
    see: https://developers.deepgram.com/docs/getting-started-with-pre-recorded-audio
    """

    text_file_path = f'{VIDEOS_DIR}/{text_file_name}.txt'
    text_done_file_path = f'{VIDEOS_DIR}/{text_file_name}_done.txt'
    if os.path.exists(text_done_file_path) and os.path.exists(text_file_path):
        logging.info(f"{text_file_path} is exists, skip transcribe, read from file")
        with open(text_file_path, 'r', encoding='utf-8') as file:
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
        logging.info(f'transcribing {audio_file} save to text file {text_file_path}')
        # Configure Deepgram options for audio analysis
        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
            language="en",
        )
        # Call the transcribe_file method with the text payload and options
        response = deepgram.listen.prerecorded.v("1").transcribe_file(
            payload, options,
            timeout=httpx.Timeout(300.0, connect=10.0),
        )
        logging.info(f'transcribing {audio_file} response {response}')
        transcript = response.results.channels[0].alternatives[0].transcript

        if len(text_file_name) > 0:
            with open(text_file_path, 'w', encoding='utf-8') as file:
                file.write(transcript)
            with open(text_done_file_path, 'a'):
                os.utime(text_done_file_path, None)

        return transcript

    except Exception as e:
        logging.exception(f"Exception: {e}")


def split_to_chunk_texts(text: str):
    # !NOTE: maybe don't to check split to chunk doc task done
    logging.info(f'Spliting text len:{len(text)}')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    split_text_list = text_splitter.split_text(text)
    logging.debug(f'splited chunk text list len {len(split_text_list)}')
    return split_text_list


def get_tidb_url():
    url = f"mysql+pymysql://{os.getenv('TIDB_USERNAME')}:{os.getenv('TIDB_PASSWORD')}@{os.getenv('TIDB_HOST')}:{os.getenv('TIDB_PORT')}/{os.getenv('TIDB_DATABASE')}?ssl_ca={os.getenv('TIDB_SSL_CA')}&ssl_verify_cert=true&ssl_verify_identity=true"
    logging.debug(f"tidb url: {url}")
    return url


def save_embeddings_to_db(table_name: str, title: str, texts: list[str]):
    # just a simple done, need use task meta info to manage
    save_done_file_path = f'{VIDEOS_DIR}/{title}_{table_name}_save_done.txt'
    if os.path.exists(save_done_file_path):
        logging.info(f"{save_done_file_path} is exists, skip save embeddings to table {table_name}")
        return

    logging.info(f'saving embeddings to db table_name:{table_name} texts len:{len(texts)}')
    # https://jina.ai/embeddings/
    embeddings = JinaEmbeddings(
        jina_api_key=os.getenv('JINA_API_KEY'),
        model_name="jina-embeddings-v2-base-en",
    )

    # https://docs.pingcap.com/tidbcloud/vector-search-integrate-with-langchain
    # Connect to tidb table vector index and insert the chunked docs as contents
    # !NOTE limitations: https://docs.pingcap.com/tidb/stable/tidb-limitations
    url = get_tidb_url()
    vector_store = TiDBVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        table_name=table_name,
        connection_string=url,
        # default, another option is "l2"
        distance_strategy=os.getenv('TIDB_VSS_DISTANCE_STRATEGY', 'cosine'),
    )

    with open(save_done_file_path, 'a'):
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
            "https://www.youtube.com/watch?v=eMlx5fFNoYc&t=786s",
        ],
        "AndrejKarpathy": [
            "https://www.youtube.com/watch?v=zjkBMFhNj_g",
            "https://www.youtube.com/watch?v=l8pRSuU81PU",
            "https://www.youtube.com/watch?v=zduSFxRajkE",
            "https://www.youtube.com/watch?v=kCc8FmEb1nY"
        ],
    }

    for name, links in video_links.items():
        for link in links:
            title, download_path = download_videos(link, VIDEOS_DIR)
            transcribe_text = transcribe_file(download_path, title)
            chunk_texts = split_to_chunk_texts(transcribe_text)
            save_embeddings_to_db(name, title=title, texts=chunk_texts)
