from datetime import datetime
import os
import logging
import uuid

from deep_translator import GoogleTranslator
from pydub import AudioSegment
from pydantic import BaseModel
import typer
from dotenv import load_dotenv

from demo.aws.upload import r2_upload
from demo.cloudflare.rest_api import d1_table_query
from demo.together_ai import save_gen_image


# Load environment variables from .env file
load_dotenv(override=True)

app = typer.Typer()


r"""
DROP TABLE IF EXISTS podcast;
CREATE TABLE IF NOT EXISTS podcast (
  id INTEGER PRIMARY KEY,
  pid text NOT NULL,
  title text NOT NULL,
  author text NOT NULL,
  /*speakker: use ',' split*/
  speakers text NOT NULL,
  /*source: video_youtube | pdf | text(txt,md) | img(jpeg,png) | audio(mp3) */
  source text DEFAULT "",
  audio_url text NOT NULL,
  description text DEFAULT "",
  audio_content text DEFAULT "",
  cover_img_url text DEFAULT "",
  duration int DEFAULT 0,
  tags text DEFAULT "",
  /*category: 0: unknow 1:tech 2:education 3:food 4:travel 5:code 6:life 7:sport 8:music */
  category int DEFAULT 0,
  /*status: 0:init 1:edited 2:checking 3:passed 4:rejected 5:deleted */
  status int DEFAULT 0,
  is_published boolean DEFAULT false,
  create_time text NOT NULL,
  update_time text NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_podcast_pid ON podcast(pid);
CREATE INDEX IF NOT EXISTS idx_podcast_ctime ON podcast(create_time);
CREATE INDEX IF NOT EXISTS idx_podcast_status ON podcast(is_published,category,status) where status!=5;
"""


class Podcast(BaseModel):
    pid: str
    title: str = ""
    author: str = ""
    speakers: str = ""
    audio_url: str = ""
    source: str = ""
    audio_content: str = ""
    cover_img_url: str = ""
    duration: int = 0
    description: str = ""
    tags: str = ""
    status: int = 0
    category: int = 0
    is_published: bool = False


@app.command("get_audio_duration")
def get_audio_duration(file_path, format="mp3"):
    audio = None
    match format:
        case "flv":
            audio = AudioSegment.from_flv(file_path)
        case "ogg":
            audio = AudioSegment.from_ogg(file_path)
        case "wav":
            audio = AudioSegment.from_wav(file_path)
        case _:
            audio = AudioSegment.from_mp3(file_path)
    duration = len(audio) // 1000  # s
    logging.info(f"{file_path} duration: {duration}s")
    return duration


@app.command("get_podcast")
def get_podcast(
    audio_file: str,
    title: str,
    author: str,
    speakers: str,
    audio_content: str = "",
    pid: str = "",
    language: str = "en",
) -> Podcast:
    cover_img_url = ""
    if title:
        en_title = title
        if language != "en":
            language = "zh-CN" if language == "zh" else language
            en_title = GoogleTranslator(
                source=language,
                target="en",
            ).translate(title)
        gen_img_prompt = f"podcast cover image which content is about {en_title}"
        img_file = save_gen_image(gen_img_prompt, uuid.uuid4().hex)
        cover_img_url = r2_upload("podcast", img_file)

    audio_url = ""
    duration = 0
    if audio_file:
        audio_url = r2_upload("podcast", audio_file)
        duration = get_audio_duration(audio_file, format=audio_file.split(".")[-1])

    podcast = Podcast(
        pid=uuid.uuid4().hex if not pid else pid,
        title=title,
        author=author,
        speakers=speakers,
        audio_content=audio_content,
        audio_url=audio_url,
        cover_img_url=cover_img_url,
        duration=duration,
    )
    logging.info(f"podcast:{podcast}")
    return podcast


@app.command("insert_podcast_to_d1")
def insert_podcast_to_d1(
    audio_file: str,
    title: str,
    author: str,
    speakers: str,
    description: str = "",
    audio_content: str = "",
    is_published: bool = False,
    status: int = 0,
    category: int = 0,
    source: str = "",
    pid: str = "",
    language: str = "en",
) -> Podcast:
    podcast = get_podcast(
        audio_file=audio_file,
        title=title,
        author=author,
        speakers=speakers,
        audio_content=audio_content,
        pid=pid,
        language=language,
    )

    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    db_id = os.getenv("PODCAST_D1_DB_ID")
    sql = "replace into podcast(pid,title,description,author,speakers,source,audio_url,audio_content,cover_img_url,duration,is_published,status,category,create_time,update_time) values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);"
    sql_params = [
        podcast.pid,
        podcast.title,
        description,
        podcast.author,
        podcast.speakers,
        source,
        podcast.audio_url,
        podcast.audio_content,
        podcast.cover_img_url,
        podcast.duration,
        1 if is_published else 0,
        status,
        category,
        formatted_time,
        formatted_time,
    ]
    res = d1_table_query(db_id, sql, sql_params)
    return res["success"]


@app.command("update_podcast_cover_to_d1")
def update_podcast_cover_to_d1(
    pid: str,
    cover_img_url: str,
) -> Podcast:
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    db_id = os.getenv("PODCAST_D1_DB_ID")
    sql = "update podcast set cover_img_url=?, update_time=? where pid=?;"
    sql_params = [
        cover_img_url,
        formatted_time,
        pid,
    ]
    res = d1_table_query(db_id, sql, sql_params)
    return res["success"]


r"""
python -m demo.insert_podcast get_podcast \
    ./audios/podcast/LLM.mp3 \
    "large language model" \
    "weedge" \
    "zh-CN-YunjianNeural,zh-CN-XiaoxiaoNeural"

python -m demo.insert_podcast insert_podcast_to_d1 \
    ./audios/podcast/LLM.mp3 \
    "large language model" \
    "weedge" \
    "zh-CN-YunjianNeural,zh-CN-XiaoxiaoNeural"

python -m demo.insert_podcast update_podcast_cover_to_d1 \
    195b08f114a94819bbb90ea2deac3220 \
    "https://pcsolutions2001.com/wp-content/uploads/2024/09/IOS_18_logo.png"
"""
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(funcName)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    app()
