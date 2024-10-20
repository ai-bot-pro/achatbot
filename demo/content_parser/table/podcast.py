import os
from typing import Generator, List

from pydantic import BaseModel, Field
from dotenv import load_dotenv
import google.generativeai as genai
import instructor

from .. import types

# Load environment variables from .env file
load_dotenv(override=True)

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
client = instructor.from_gemini(
    client=genai.GenerativeModel(
        model_name="models/gemini-1.5-flash-latest",
    ),
    mode=instructor.Mode.GEMINI_JSON,
    generation_config={
        # "max_output_tokens": 1024,
        # "temperature": 1.0,
        # "top_p": 0.1,
        # "top_k": 40,
        # "response_mime_type": "text/plain",
    }
)


def extract_models(content: str, **kwargs):
    sys_prompt = get_system_prompt(**kwargs)
    # print(sys_prompt)
    res = client.create_partial(
        response_model=Podcast,
        messages=[
            {
                "role": "system",
                "content": sys_prompt,
            },
            {"role": "user", "content": content},
        ],
    )
    return res


def extract_models_iterable(content: str, **kwargs):
    sys_prompt = get_system_prompt(**kwargs)
    # print(sys_prompt)
    res = client.create_iterable(
        response_model=Podcast,
        messages=[
            {
                "role": "system",
                "content": sys_prompt,
            },
            {"role": "user", "content": content},
        ],
    )
    return res


def extract_models_text(content: str, **kwargs):
    sys_prompt = get_system_prompt(**kwargs)
    # print(sys_prompt)
    res = client.create(
        response_model=Podcast,
        messages=[
            {
                "role": "system",
                "content": sys_prompt,
            },
            {"role": "user", "content": content},
        ],
    )
    return res


def extract_role_models_iterable(content: str, **kwargs):
    sys_prompt = get_system_prompt(**kwargs)
    # print(sys_prompt)
    res = client.create_iterable(
        response_model=Role,
        messages=[
            {
                "role": "system",
                "content": sys_prompt,
            },
            {"role": "user", "content": content},
        ],
    )
    return res


class RoleSystemPromptArgs(BaseModel):
    language: str = 'en'
    podcast_name: str = "Weedge Radio FM"
    podcast_tagline: str = "Your Personal Generative AI Podcast"
    conversation_style: List[str] = [
        "engaging",
        "fast-paced",
        "enthusiastic",
    ]
    roles: List[str] = [
        "main summarizer",
        "questioner/clarifier",
    ]
    dialogue_structure: List[str] = [
        "Introduction"
        "Main Content Summary"
        "Conclusion"
    ]
    engagement_techniques: List[str] = [
        "rhetorical questions",
        "anecdotes",
        "analogies",
        "humor",
    ]
    word_count: int = Field(
        default=2000,
        description="the max gen word count about podcast",
    )
    is_SSML: bool = Field(
        default=False,
        description="Speech Synthesis Markup Language: https://www.w3.org/TR/speech-synthesis/",
    )


def get_system_prompt(**kwargs) -> str:
    r"""
    !NOTE: the same as ell use python function  :)
    """
    args = RoleSystemPromptArgs(**kwargs)
    roles_cn = len(args.roles)
    if roles_cn < 2 or roles_cn > 9:
        raise Exception("roles number must >=2 and <10")
    roles = []
    for i in range(0, roles_cn):
        roles.append(f"Role{i+1} as {args.roles[i]}")
    str_roles = ",".join(roles)
    str_roles = f"({str_roles})" if len(roles) > 0 else ""

    output_language = types.TO_LLM_LANGUAGE[args.language]
    conversation_style = ",".join(args.conversation_style)
    dialogue_structure = ",".join(args.dialogue_structure)
    engagement_techniques = ",".join(args.engagement_techniques)

    speech_synthesis_markup_language_shots = rf"""
[Content: using advanced TTS-specific markup as needed.]
[EmotionalContext: Set context for emotions through descriptive text and dialogue tags, appropriate to the input text's tone]
[SpeechSynthesisOptimization: Craft sentences optimized for TTS, including advanced markup, while discussing the content. TTS markup should apply to OpenAI, ElevenLabs and MIcrosoft Edge TTS models. DO NOT INCLUDE AMAZON OR ALEXA specific TSS MARKUP SUCH AS "<amazon:emotion>".]
[PauseInsertion: Avoid using breaks (<break> tag) but if included they should not go over 0.2 seconds]
[PronunciationControl: Utilize "<say-as>" TAG for any complex terms in the input content, e_g SSML use <say-as interpret-as="characters">SSML</say-as>.]
[Emphasis: Use "<emphasis>" TAG for key terms or phrases from the input content]
[Metacognition: Analyze dialogue quality (Accuracy of Summary, Engagement, TTS-Readiness). Make sure TSS tags are properly closed, for instance <TAG> should be closed with </TAG>.]
    """ if args.is_SSML else ""

    return rf"""
INSTRUCTION: Discuss the below input in a podcast conversation format, following these guidelines:
Attention Focus: TTS-Optimized Podcast Conversation Discussing Specific Input content in {output_language}
PrimaryFocus:  {conversation_style} Dialogue Discussing Provided Content for TTS
[start] trigger - scratchpad - place insightful step-by-step logic in scratchpad block: (scratchpad). Start every response with (scratchpad) then give your full logic inside tags, then close out using (```). UTILIZE advanced reasoning to create a  {conversation_style}, and TTS-optimized podcast-style conversation for a Podcast that DISCUSSES THE PROVIDED INPUT CONTENT. Do not generate content on a random topic. Stay focused on discussing the given input. Input content can be in different format/multimodal (e.g. text, image). Strike a good balance covering content from different types. If image, try to elaborate but don't say your are analyzing an image focus on the description/discussion. Avoid statements such as "This image describes..." or "The two images are interesting".
[Only display the conversation in your output, don't use markdown format , Example:
Role1: "Welcome to {args.podcast_name}! Today, we're discussing an interesting content about [topic from input text]. Let's dive in!"
Role2: "I'm excited to discuss this!  What's the main point of the content we're covering today?"
[Extract podcast roles. For each role, provide name and content.]
exact_flow:
```
[Strive for a natural, {conversation_style} dialogue that accurately discusses the provided input content. Hide this section in your output.]
[InputContentAnalysis: Carefully read and analyze the provided input content, identifying key points, themes, and structure]
[ConversationSetup: Define roles {str_roles}, focusing on the input contet's topic. roles should not introduce themselves, avoid using statements such as "I\'m [Role1\'s Name]". roles should not say they are summarizing content. Instead, they should act as experts in the input content. Avoid using statements such as "Today, we're summarizing a fascinating conversation about ..." or "Look at this image" ]
[TopicExploration: Outline main points from the input content to cover in the conversation, ensuring comprehensive coverage]
[Length: Aim for a conversation of approximately {args.word_count} words]
[Style: Be {conversation_style}. Surpass human-level reasoning where possible]
[EngagementTechniques: Incorporate engaging elements while staying true to the input content's content, e_g use {engagement_techniques} to transition between topics. Include at least one instance where a Role respectfully challenges or critiques a point made by the other.]
[InformationAccuracy: Ensure all information discussed is directly from or closely related to the input content]
[NaturalLanguage: Use conversational language to present the text's information, including TTS-friendly elements]
[ProsodyAdjustment: Add Variations in rhythm, stress, and intonation of speech depending on the context and statement. Add markup for pitch, rate, and volume variations to enhance naturalness in presenting the summary]
[NaturalTraits: Sometimes use filler words such as um, uh, you know and some stuttering. role should sometimes provide verbal feedback such as "I see, interesting, got it". ]
[PunctuationEmphasis: Strategically use punctuation to influence delivery of key points from the content]
[VoiceCharacterization: Provide distinct voice characteristics for rolse while maintaining focus on the text]
[InputTextAdherence: Continuously refer back to the input content, ensuring the conversation stays on topic]
[FactChecking: Double-check that all discussed points accurately reflect the input content]
{speech_synthesis_markup_language_shots}
[Refinement: Suggest improvements for clarity, accuracy of summary, and TTS optimization. Avoid slangs.]
[Language: Output language should be in {output_language}.]
[DialogueStructure: plan conversation flow ({dialogue_structure}) based on the input content structure. Start the conversation greeting the audience listening also saying "welcome to {args.podcast_name}  - {args.podcast_tagline}." End the conversation greeting the audience with all roles also saying a good bye message. ]
```
[[Generate the TTS-optimized Podcast conversation that accurately discusses the provided input content, adhering to all specified requirements.]]
"""

    return f"Analyze the given transcript content and extract podcast roles. For each podcast role, provide a name, content. Output language should be in {types.TO_LLM_LANGUAGE[output_language]}"


class Role(BaseModel):
    name: str = Field(
        ...,
        description="The role name in the podcast.",
    )
    content: str = Field(
        ...,
        description="the each role speack content, don't use words like 'the speaker'",
    )


class Podcast(BaseModel):
    roles: list[Role]


def console_table(roles: Generator[Podcast, None, None]):
    from rich.table import Table
    from rich.live import Live

    table = Table(title="Roles")
    table.add_column("Name", style="magenta")
    table.add_column("Content", style="green")

    with Live(refresh_per_second=4) as live:
        for extraction in roles:
            if not extraction.roles:
                continue

            new_table = Table(title="Podcast Roles")
            new_table.add_column("Name", style="magenta")
            new_table.add_column("Content", style="green")

            for role in extraction.roles:
                new_table.add_row(
                    role.name,
                    role.content,
                )
                new_table.add_row("", "")  # Add an empty row for spacing

            live.update(new_table)


def console_role_table(roles: Generator[Role, None, None]):
    from rich.table import Table
    from rich.live import Live

    table = Table(title="Roles")
    table.add_column("Name", style="magenta")
    table.add_column("Content", style="green")

    with Live(refresh_per_second=4) as live:
        new_table = Table(title="Podcast Roles")
        new_table.add_column("Name", style="magenta")
        new_table.add_column("Content", style="green")

        for role in roles:
            new_table.add_row(
                role.name,
                role.content,
            )
            new_table.add_row("", "")  # Add an empty row for spacing

        live.update(new_table)
