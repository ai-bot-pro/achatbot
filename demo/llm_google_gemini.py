import os
from typing import Iterable

import google.generativeai as genai
from google.protobuf.struct_pb2 import Struct
from google.generativeai.types import content_types
import typer

from dotenv import load_dotenv

load_dotenv()

app = typer.Typer()

api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)


def set_light_values(brightness, color_temp):
    """Set the brightness and color temperature of a room light. (mock API).

    Args:
        brightness: Light level from 0 to 100. Zero is off and 100 is full brightness
        color_temp: Color temperature of the light fixture, which can be `daylight`, `cool` or `warm`.

    Returns:
        A dictionary containing the set brightness and color temperature.
    """
    return {"brightness": brightness, "colorTemperature": color_temp}


def add(a: float, b: float):
    """returns a + b."""
    return a + b


def subtract(a: float, b: float):
    """returns a - b."""
    return a - b


def multiply(a: float, b: float):
    """returns a * b."""
    return a * b


def divide(a: float, b: float):
    """returns a / b."""
    return a / b


@app.command()
def run_auto_function_calling():
    """
    Function calls naturally fit in to [multi-turn chats](https://ai.google.dev/api/python/google/generativeai/GenerativeModel#multi-turn) as they capture a back and forth interaction between the user and model. The Python SDK's [`ChatSession`](https://ai.google.dev/api/python/google/generativeai/ChatSession) is a great interface for chats because handles the conversation history for you, and using the parameter `enable_automatic_function_calling` simplifies function calling even further
    """
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-latest",
        tools=[add, subtract, multiply, divide],
        system_instruction="You are a helpful assistant who converses with a user and answers questions. Respond concisely to general questions. ",
    )
    print(model._generation_config)
    chat = model.start_chat(enable_automatic_function_calling=True)
    response = chat.send_message(
        [
            # "what's your name?",
            "I have 57 cats, each owns 44 mittens, how many mittens is that in total?",
        ],
        # stream=True, # enable_automatic_function_calling=True, unsupport stream
    )
    print(f"run_auto_function_calling response: {response}")
    for content in chat.history:
        print(content.role, "->", [type(part).to_dict(part) for part in content.parts])
        print("-" * 80)


def find_movies(description: str, location: str = ""):
    """find movie titles currently playing in theaters based on any description, genre, title words, etc.

    Args:
        description: Any kind of description including category or genre, title words, attributes, etc.
        location: The city and state, e.g. San Francisco, CA or a zip code e.g. 95616
    """
    return ["Barbie", "Oppenheimer"]


def find_theaters(location: str, movie: str = ""):
    """Find theaters based on location and optionally movie title which are is currently playing in theaters.

    Args:
        location: The city and state, e.g. San Francisco, CA or a zip code e.g. 95616
        movie: Any movie title
    """
    return ["Googleplex 16", "Android Theatre"]


def get_showtimes(location: str, movie: str, theater: str, date: str):
    """
    Find the start times for movies playing in a specific theater.

    Args:
      location: The city and state, e.g. San Francisco, CA or a zip code e.g. 95616
      movie: Any movie title
      thearer: Name of the theater
      date: Date for requested showtime
    """
    return ["10:00", "11:00"]


def call_function(function_call, functions):
    function_name = function_call.name
    function_args = function_call.args
    return functions[function_name](**function_args)


@app.command()
def run_manual_function_calling(stream: bool = True):
    """
        run manual function calling, no chat session, no history message
        For more control, you can process [`genai.protos.FunctionCall`](https://ai.google.dev/api/python/google/generativeai/protos/FunctionCall) requests from the model yourself. This would be the case if:
    - You use a `ChatSession` with the default `enable_automatic_function_calling=False`.
    - You use `GenerativeModel.generate_content` (and manage the chat history yourself).
    """
    functions = {
        "find_movies": find_movies,
        "find_theaters": find_theaters,
        "get_showtimes": get_showtimes,
    }
    model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest", tools=functions.values())
    # 1. llm --gen--> function call
    response = model.generate_content(
        "Which theaters in Mountain View show the Barbie movie?",
        stream=stream,
    )
    print(f"run_manual_function_calling response:{response}")
    for chunk in response:
        print("function call chunk", chunk)

    # 2. function call --do--> result
    part = response.candidates[0].content.parts[0]
    # Check if it's a function call; in real use you'd need to also handle text
    # responses as you won't know what the model will respond with.
    if part.function_call:
        result = call_function(part.function_call, functions)
    print(f"function_call:{part.function_call} result: {result}")

    # 3. llm with build messages --gen--> text response
    # Put the result in a protobuf Struct
    s = Struct()
    s.update({"result": result})
    # Update this after https://github.com/google/generative-ai-python/issues/243
    function_response = genai.protos.Part(
        function_response=genai.protos.FunctionResponse(name="find_theaters", response=s)
    )
    # Build the message history
    messages = [
        # fmt: off
        {"role": "user", "parts": ["Which theaters in Mountain View show the Barbie movie?."]},
        {"role": "model", "parts": response.candidates[0].content.parts},
        {"role": "user", "parts": [function_response]},
        # fmt: on
    ]
    print("messages-->", messages)
    # Generate the next response
    response = model.generate_content(messages, stream=stream)
    print(f"run_manual_function_calling Generate the next response:{response}")
    for chunk in response:
        print("llm generate chunk", chunk)


@app.command()
def run_function_calling_chain():
    """
    The model is not limited to one function call,
    it can chain them until it finds the right answer.
    """
    functions = {
        "find_movies": find_movies,
        "find_theaters": find_theaters,
        "get_showtimes": get_showtimes,
    }
    model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest", tools=functions.values())
    chat = model.start_chat(enable_automatic_function_calling=True)
    response = chat.send_message(
        "Which comedy movies are shown tonight in Mountain view and at what time?",
        # NotImplementedError: Unsupported configuration: The
        # `google.generativeai` SDK currently does not support the combination of
        # `stream=True` and `enable_automatic_function_calling=True`.
        # stream=True,
    )
    print(f"run_function_calling_chain response:{response}")
    for content in chat.history:
        print(content.role, "->", [type(part).to_dict(part) for part in content.parts])
        print("-" * 80)


def power_disco_ball(power: bool) -> bool:
    """Powers the spinning disco ball."""
    print(f"Disco ball is {'spinning!' if power else 'stopped.'}")
    return True


def start_music(energetic: bool, loud: bool, bpm: int) -> str:
    """Play some music matching the specified parameters.

    Args:
      energetic: Whether the music is energetic or not.
      loud: Whether the music is loud or not.
      bpm: The beats per minute of the music.

    Returns: The name of the song being played.
    """
    print(f"Starting music! {energetic=} {loud=}, {bpm=}")
    return "Never gonna give you up."


def dim_lights(brightness: float) -> bool:
    """Dim the lights.

    Args:
      brightness: The brightness of the lights, 0.0 is off, 1.0 is full.
    """
    print(f"Lights are now set to {brightness:.0%}")
    return True


@app.command()
def run_parallel_function_calls():
    """
    The Gemini API can call multiple functions in a single turn.
    This caters for scenarios where there are multiple function calls
    that can take place independently to complete a task.
    """

    # Set the model up with tools.
    house_fns = [power_disco_ball, start_music, dim_lights]
    # Try this out with Pro and Flash...
    model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest", tools=house_fns)

    # Call the API.
    # chat = model.start_chat(enable_automatic_function_calling=True) # chat complete
    chat = model.start_chat()  # function call
    response = chat.send_message(
        "Turn this place into a party!",
        # NotImplementedError: Unsupported configuration: The
        # `google.generativeai` SDK currently does not support the combination of
        # `stream=True` and `enable_automatic_function_calling=True`.
        # stream=True,
    )
    print(f"run_parallel_function_calls response:{response}")

    # Print out each of the function calls requested from this single call.
    for part in response.parts:
        if fn := part.function_call:
            args = ", ".join(f"{key}={val}" for key, val in fn.args.items())
            print(f"{fn.name}({args})")

    for content in chat.history:
        print(content.role, "->", [type(part).to_dict(part) for part in content.parts])
        print("-" * 80)


def enable_lights():
    """Turn on the lighting system."""
    print("LIGHTBOT: Lights enabled.")


def set_light_color(rgb_hex: str):
    """Set the light color. Lights must be enabled for this to work."""
    print(f"LIGHTBOT: Lights set to {rgb_hex}.")


def stop_lights():
    """Stop flashing lights."""
    print("LIGHTBOT: Lights turned off.")


def tool_config_from_mode(mode: str, fns: Iterable[str] = ()):
    """Create a tool config with the specified function calling mode."""
    return content_types.to_tool_config(
        {"function_calling_config": {"mode": mode, "allowed_function_names": fns}}
    )


@app.command()
def run_auto_function_calling_with_config():
    r"""
    Specifying a `function_calling_config` allows you to control how the Gemini API acts when `tools` have been specified. For example, you can choose to only allow free-text output (disabling function calling), force it to choose from a subset of the functions provided in `tools`, or let it act automatically.
    This guide assumes you are already familiar with function calling. For an introduction, check out the [docs](https://ai.google.dev/docs/function_calling).
    """
    light_controls = [enable_lights, set_light_color, stop_lights]
    instruction = "You are a helpful lighting system bot. You can turn lights on and off, and you can set the color. Do not perform any other tasks."

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-latest", tools=light_controls, system_instruction=instruction
    )

    chat = model.start_chat()

    print("-------none function call-----------")
    tool_config = tool_config_from_mode("none")
    response = chat.send_message("Hello light-bot, what can you do?", tool_config=tool_config)
    print(response.text)
    for content in chat.history:
        print(content.role, "->", [type(part).to_dict(part) for part in content.parts])
        print("-" * 80)

    print("-------auto function call-----------")
    tool_config = tool_config_from_mode("auto")
    response = chat.send_message("Light this place up!", tool_config=tool_config)
    print(response.parts[0])
    for content in chat.history:
        print(content.role, "->", [type(part).to_dict(part) for part in content.parts])
        print("-" * 80)
    # You are not actually calling the function, so remove this from the history.
    chat.rewind()
    print("-------rewind chat history-----------")
    for content in chat.history:
        print(content.role, "->", [type(part).to_dict(part) for part in content.parts])
        print("-" * 80)

    print("-------any function call-----------")
    available_fns = ["set_light_color", "stop_lights"]
    tool_config = tool_config_from_mode("any", available_fns)
    response = chat.send_message("Make this place PURPLE!", tool_config=tool_config)
    print(response.parts[0])
    for content in chat.history:
        print(content.role, "->", [type(part).to_dict(part) for part in content.parts])
        print("-" * 80)

    print("-------any function call with chat complete-----------")
    available_fns = ["enable_lights"]
    tool_config = tool_config_from_mode("any", available_fns)
    auto_chat = model.start_chat(enable_automatic_function_calling=True)
    response = auto_chat.send_message("It's awful dark in here...", tool_config=tool_config)
    print(response.parts[0])
    for content in chat.history:
        print(content.role, "->", [type(part).to_dict(part) for part in content.parts])
        print("-" * 80)


if __name__ == "__main__":
    app()
