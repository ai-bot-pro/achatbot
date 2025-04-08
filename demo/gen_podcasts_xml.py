import logging
import os
from datetime import datetime
import html
from typing import List, Dict

import typer
from dotenv import load_dotenv
import xml.etree.ElementTree as ET

from demo.cloudflare.rest_api import d1_table_query
from demo.aws.upload import r2_upload

# Load environment variables from .env file
load_dotenv(override=True)
app = typer.Typer()


def escape_xml(text: str) -> str:
    """Escapes XML special characters in a string."""
    return html.escape(text, quote=True)


def format_date_to_rfc822(date_string: str) -> str:
    """
    Formats a date string (YYYY-MM-DD HH:MM:SS) to RFC 822 format.
    """
    try:
        # Assuming the input date string includes time as well
        date_obj = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        try:
            # Try parsing without time
            date_obj = datetime.strptime(date_string, "%Y-%m-%d")
        except ValueError:
            # If both formats fail, raise an error
            raise ValueError("Incorrect date format, should be YYYY-MM-DD or YYYY-MM-DD HH:MM:SS")

    return date_obj.strftime("%a, %d %b %Y %H:%M:%S +0000")


def generate_podcast_feed_xml(podcasts: List[Dict]) -> str:
    """Generates an RSS feed XML string for a list of podcasts.

    https://help.apple.com/itc/podcasts_connect/#/itcbaf351599

    Args:
        podcasts: A list of dictionaries, where each dictionary represents a podcast
                  and contains the following keys:
                  - 'title': str, required
                  - 'description': str, required
                  - 'pid': str, required, unique identifier
                  - 'create_time': str, required, in YYYY-MM-DD HH:MM:SS format
                  - 'audio_url': str, required
                  - 'audio_size': int, required
                  - 'duration': int, required, in second format
                  - 'cover_img_url': str, required, the url of image
                  - 'source': str, optional.
    Returns:
        A string containing the generated RSS feed XML.
    """

    channel_title = "AI Podcast"  # Replace with your podcast name
    channel_description = "Latest podcasts about AI Technology and Papers."  # Replace with your podcast description  # Replace with your podcast description
    channel_link = "https://podcast-997.pages.dev"  # Replace with your website URL
    channel_language = "en-us"
    channel_image_url = (
        "https://podcast-997.pages.dev/an_AI_podcast_1400.jpg"  # Replace with your podcast image
    )

    # <rss xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd" xmlns:podcast="https://podcastindex.org/namespace/1.0" version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
    rss = ET.Element(
        "rss",
        version="2.0",
        attrib={
            "xmlns:itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd",
            "xmlns:atom": "http://www.w3.org/2005/Atom",
            "xmlns:podcast": "https://podcastindex.org/namespace/1.0",
            "xmlns:content": "http://purl.org/rss/1.0/modules/content/",
        },
    )
    channel = ET.SubElement(rss, "channel")
    ET.SubElement(channel, "title").text = channel_title
    ET.SubElement(channel, "itunes:author").text = "weedge"
    ET.SubElement(channel, "itunes:explicit").text = "false"
    ET.SubElement(channel, "itunes:image", href=channel_image_url)
    ET.SubElement(channel, "itunes:category", text="Technology")
    ET.SubElement(channel, "link").text = channel_link
    ET.SubElement(channel, "description").text = channel_description
    ET.SubElement(channel, "language").text = channel_language
    ET.SubElement(channel, "copyright").text = "&#169; 2024-2025 weedge"
    ET.SubElement(
        channel,
        "atom:link",
        href="https://pub-f8da0a7ab3e74cc8a8081b2d4b8be851.r2.dev/rss.xml",
        rel="self",
        type="application/rss+xml",
    )

    image = ET.SubElement(channel, "image")
    ET.SubElement(image, "url").text = channel_image_url
    ET.SubElement(image, "title").text = channel_title
    ET.SubElement(image, "link").text = channel_link

    for podcast in podcasts:
        item = ET.SubElement(channel, "item")

        item_title = escape_xml(podcast["title"])
        item_author = escape_xml(podcast["author"])
        item_description = escape_xml(podcast["description"])

        item_link = f"https://podcast-997.pages.dev/podcast/{podcast['pid']}"

        item_pub_date = format_date_to_rfc822(podcast["create_time"])
        item_audio_url = escape_xml(podcast["audio_url"])
        item_audio_length = str(podcast["audio_size"])

        duration_seconds = podcast["duration"]
        minutes, seconds = divmod(duration_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        item_duration = (
            "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))
            if hours
            else "{:02d}:{:02d}".format(int(minutes), int(seconds))
        )

        ET.SubElement(item, "title").text = item_title
        ET.SubElement(item, "author").text = item_author
        ET.SubElement(item, "description").text = item_description
        ET.SubElement(item, "link").text = item_link
        ET.SubElement(item, "guid").text = item_link
        ET.SubElement(item, "pubDate").text = item_pub_date

        ET.SubElement(
            item, "enclosure", url=item_audio_url, length=item_audio_length, type="audio/mpeg"
        )

        ET.SubElement(item, "itunes:duration").text = item_duration
        ET.SubElement(item, "itunes:explicit").text = "false"

    xml_string = ET.tostring(rss, encoding="UTF-8", xml_declaration=True).decode("utf-8")
    return xml_string


@app.command("gen_xml_from_d1_podcast")
def gen_xml_from_d1_podcast(is_upload: bool = False):
    db_id = os.getenv("PODCAST_D1_DB_ID")
    select_sql = "select * from podcast where audio_size>0 and is_published is True order by create_time desc;"
    select_res = d1_table_query(db_id, select_sql)
    podcasts_data = select_res["result"][0]["results"]

    rss_xml = generate_podcast_feed_xml(podcasts_data)
    # print(rss_xml)

    # Optionally, save to a file:
    with open("rss.xml", "w", encoding="utf-8") as f:
        f.write(rss_xml)

    if is_upload:
        r2_upload("podcast", "rss.xml")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(funcName)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    app()
