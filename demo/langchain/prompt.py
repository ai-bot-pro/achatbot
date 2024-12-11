import logging
from typing import List

from dotenv import load_dotenv
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
import typer

# Load environment variables from .env file
load_dotenv(override=True)


app = typer.Typer()


def compose_prompt(num_images: int):
    # use LANGCHAIN_API_KEY
    prompt_template = hub.pull("weedge/podcastfy")

    image_path_keys = []
    messages = []
    text_content = {"type": "text", "text": "{input_text}"}
    messages.append(text_content)
    for i in range(num_images):
        key = f"image_path_{i}"
        image_content = {
            "image_url": {"path": f"{{{key}}}", "detail": "high"},
            "type": "image_url",
        }
        image_path_keys.append(key)
        messages.append(image_content)
    user_prompt_template = ChatPromptTemplate.from_messages(
        messages=[HumanMessagePromptTemplate.from_template(messages)]
    )

    # Compose messages from podcastfy_prompt_template and user_prompt_template
    combined_messages = prompt_template.messages + user_prompt_template.messages

    # Create a new ChatPromptTemplate object with the combined messages
    composed_prompt_template = ChatPromptTemplate.from_messages(combined_messages)

    return composed_prompt_template, image_path_keys


def compose_prompt_params(
    image_file_paths: List[str], image_path_keys: List[str], input_texts: str
):
    prompt_params = {
        "input_text": input_texts,
        "word_count": "2000",
        "podcast_name": "podcast_name",
        "podcast_tagline": "podcast_tagline",
        "output_language": "English",
        "conversation_style": ", ".join(["happy"]),
        "str_roles": ", ".join(["roles_person1", "roles_person2"]),
        "dialogue_structure": ", ".join(["clear"]),
        "engagement_techniques": ", ".join(["ok"]),
        "speech_synthesis_markup_language_shots": "",
    }

    # for each image_path_key, add the corresponding image_file_path to the prompt_params
    for key, path in zip(image_path_keys, image_file_paths):
        prompt_params[key] = path

    return prompt_params


@app.command()
def main(imgs: List[str] = [], input_text="hello"):
    composed_prompt_template, image_path_keys = compose_prompt(len(imgs))
    print(composed_prompt_template, type(composed_prompt_template))
    print("-" * 70)
    prompt_params = compose_prompt_params(imgs, image_path_keys, input_text)
    print(prompt_params)
    print("-" * 70)
    prompt = composed_prompt_template.invoke(prompt_params)
    print(prompt.model_dump_json(indent=4))


r"""
python -m demo.langchain.prompt

python -m demo.langchain.prompt --imgs "./images/cv_capture0.jpg"
"""
if __name__ == "__main__":
    app()
