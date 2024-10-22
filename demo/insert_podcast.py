from datetime import datetime
import os
import logging
import uuid

from pydantic import BaseModel
import typer
from dotenv import load_dotenv

from demo.aws.upload import r2_upload
from demo.cloudflare.rest_api import d1_table_query
from demo.image_together_flux import save_gen_image


# Load environment variables from .env file
load_dotenv(override=True)

app = typer.Typer()


r"""
DROP TABLE IF EXISTS podcast;
CREATE TABLE IF NOT EXISTS podcast (
  pid text NOT NULL,
  title text NOT NULL,
  author text NOT NULL,
  speakers text NOT NULL,
  audio_url text NOT NULL,
  audio_content text DEFAULT "",
  cover_img_url text DEFAULT "",
  create_time text NOT NULL,
  update_time text NOT NULL
);
"""


class Podcast(BaseModel):
    pid: str
    title: str = ""
    author: str = ""
    speakers: str = ""
    audio_url: str = ""
    audio_content: str = ""
    cover_img_url: str = ""


@app.command('get_podcast')
def get_podcast(
    audio_file: str,
    title: str,
    author: str,
    speakers: str,
    audio_content: str = "",
) -> Podcast:
    gen_img_prompt = f"podcast cover image which content is about {title}, image no words."
    img_file = save_gen_image(gen_img_prompt, uuid.uuid4().hex)
    cover_img_url = r2_upload("podcast", img_file)

    audio_url = r2_upload("podcast", audio_file)

    podcast = Podcast(
        pid=uuid.uuid4().hex,
        title=title,
        author=author,
        speakers=speakers,
        audio_content=audio_content,
        audio_url=audio_url,
        cover_img_url=cover_img_url,
    )
    logging.info(f"podcast:{podcast}")
    return podcast


@app.command('insert_podcast_to_d1')
def insert_podcast_to_d1(
    audio_file: str,
    title: str,
    author: str,
    speakers: str,
    audio_content: str = "",
) -> Podcast:
    podcast = get_podcast(
        audio_file=audio_file, title=title,
        author=author, speakers=speakers,
        audio_content=audio_content,
    )

    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    db_id = os.getenv("PODCAST_D1_DB_ID")
    sql = "insert into podcast(pid,title,author,speakers,audio_url,audio_content,cover_img_url,create_time,update_time) values(?,?,?,?,?,?,?,?,?);"
    sql_params = [
        podcast.pid,
        podcast.title,
        podcast.author,
        podcast.speakers,
        podcast.audio_url,
        podcast.audio_content,
        podcast.cover_img_url,
        formatted_time,
        formatted_time,
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
"""
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(funcName)s - %(message)s',
        handlers=[
            # logging.FileHandler("content_parser_tts.log"),
            logging.StreamHandler()
        ],
    )
    app()
