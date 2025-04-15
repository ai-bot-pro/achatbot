import modal
import os
import io

app = modal.App("qwen2_5_omni_web_demo")
omni_img = (
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "git-lfs", "ffmpeg", "cmake")
    .pip_install("wheel", "openai", "qwen-omni-utils[decord]")
    .run_commands(
        f"pip install git+https://github.com/huggingface/transformers",
    )
    .pip_install("accelerate", "torch", "torchvision", "torchaudio")
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .pip_install(
        "gradio==5.23.1",
        "gradio_client==1.8.0",
        "ffmpeg==1.4",
        "ffmpeg-python==0.2.0",
        "soundfile==0.13.0",
        "librosa==0.11.0",
        "modelscope_studio==1.2.2",
        "av",
    )
)

HF_MODEL_DIR = "/root/models"
hf_model_vol = modal.Volume.from_name("models", create_if_missing=True)
ASSETS_DIR = "/root/assets"
assets_dir = modal.Volume.from_name("assets", create_if_missing=True)

# NOTE: if want to generate speech, need use this system prompt to generate speech
SPEECH_SYS_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

with omni_img.imports():
    import subprocess
    import ffmpeg
    import numpy as np
    import soundfile as sf
    import torch, torchaudio

    import modelscope_studio.components.base as ms
    import modelscope_studio.components.antd as antd

    import gradio as gr
    import gradio.processing_utils as processing_utils
    from gradio_client import utils as client_utils

    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from qwen_omni_utils import process_mm_info

    subprocess.run("nvidia-smi --version", shell=True)
    subprocess.run("nvcc --version", shell=True)
    gpu_prop = torch.cuda.get_device_properties("cuda")
    print(gpu_prop)

    model_path = os.path.join(HF_MODEL_DIR, "Qwen/Qwen2.5-Omni-7B")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    ).eval()
    model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
    # print(model)
    print(f"{model_million_params} M parameters")

    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

    subprocess.run("nvidia-smi", shell=True)


def _launch_demo(model, processor, ui_language, share, inbrowser, server_port, server_name):
    # Voice settings
    VOICE_LIST = ["Chelsie", "Ethan"]
    DEFAULT_VOICE = "Chelsie"

    default_system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

    language = ui_language

    def get_text(text: str, cn_text: str):
        if language == "en":
            return text
        if language == "zh":
            return cn_text
        return text

    def convert_webm_to_mp4(input_file, output_file):
        try:
            (
                ffmpeg.input(input_file)
                .output(output_file, acodec="aac", ar="16000", audio_bitrate="192k")
                .run(quiet=True, overwrite_output=True)
            )
            print(f"Conversion successful: {output_file}")
        except ffmpeg.Error as e:
            print("An error occurred during conversion.")
            print(e.stderr.decode("utf-8"))

    def format_history(history: list, system_prompt: str):
        messages = []
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        for item in history:
            if isinstance(item["content"], str):
                messages.append({"role": item["role"], "content": item["content"]})
            elif item["role"] == "user" and (
                isinstance(item["content"], list) or isinstance(item["content"], tuple)
            ):
                file_path = item["content"][0]

                mime_type = client_utils.get_mimetype(file_path)
                if mime_type.startswith("image"):
                    messages.append(
                        {"role": item["role"], "content": [{"type": "image", "image": file_path}]}
                    )
                elif mime_type.startswith("video"):
                    messages.append(
                        {"role": item["role"], "content": [{"type": "video", "video": file_path}]}
                    )
                elif mime_type.startswith("audio"):
                    messages.append(
                        {
                            "role": item["role"],
                            "content": [
                                {
                                    "type": "audio",
                                    "audio": file_path,
                                }
                            ],
                        }
                    )
        return messages

    def predict(messages, voice=DEFAULT_VOICE):
        print("predict history: ", messages)

        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)

        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True,
        )
        inputs = inputs.to(model.device).to(model.dtype)

        text_ids, audio = model.generate(
            **inputs, speaker=voice, use_audio_in_video=True, return_auidio=True
        )

        response = processor.batch_decode(
            text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = response[0].split("\n")[-1]
        yield {"type": "text", "data": response}

        audio = np.array(audio * 32767).astype(np.int16)
        wav_io = io.BytesIO()
        sf.write(wav_io, audio, samplerate=24000, format="WAV")
        wav_io.seek(0)
        wav_bytes = wav_io.getvalue()
        audio_path = processing_utils.save_bytes_to_cache(
            wav_bytes, "audio.wav", cache_dir=demo.GRADIO_CACHE
        )
        yield {"type": "audio", "data": audio_path}

    def media_predict(audio, video, history, system_prompt, voice_choice):
        # First yield
        yield (
            None,  # microphone
            None,  # webcam
            history,  # media_chatbot
            gr.update(visible=False),  # submit_btn
            gr.update(visible=True),  # stop_btn
        )

        if video is not None:
            convert_webm_to_mp4(video, video.replace(".webm", ".mp4"))
            video = video.replace(".webm", ".mp4")
        files = [audio, video]

        for f in files:
            if f:
                history.append({"role": "user", "content": (f,)})

        formatted_history = format_history(
            history=history,
            system_prompt=system_prompt,
        )

        history.append({"role": "assistant", "content": ""})

        for chunk in predict(formatted_history, voice_choice):
            if chunk["type"] == "text":
                history[-1]["content"] = chunk["data"]
                yield (
                    None,  # microphone
                    None,  # webcam
                    history,  # media_chatbot
                    gr.update(visible=False),  # submit_btn
                    gr.update(visible=True),  # stop_btn
                )
            if chunk["type"] == "audio":
                history.append({"role": "assistant", "content": gr.Audio(chunk["data"])})

        # Final yield
        yield (
            None,  # microphone
            None,  # webcam
            history,  # media_chatbot
            gr.update(visible=True),  # submit_btn
            gr.update(visible=False),  # stop_btn
        )

    def chat_predict(text, audio, image, video, history, system_prompt, voice_choice):
        # Process text input
        if text:
            history.append({"role": "user", "content": text})

        # Process audio input
        if audio:
            history.append({"role": "user", "content": (audio,)})

        # Process image input
        if image:
            history.append({"role": "user", "content": (image,)})

        # Process video input
        if video:
            history.append({"role": "user", "content": (video,)})

        formatted_history = format_history(history=history, system_prompt=system_prompt)

        yield None, None, None, None, history

        history.append({"role": "assistant", "content": ""})
        for chunk in predict(formatted_history, voice_choice):
            if chunk["type"] == "text":
                history[-1]["content"] = chunk["data"]
                yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), history
            if chunk["type"] == "audio":
                history.append({"role": "assistant", "content": gr.Audio(chunk["data"])})
        yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), history

    with gr.Blocks() as demo, ms.Application(), antd.ConfigProvider():
        with gr.Sidebar(open=False):
            system_prompt_textbox = gr.Textbox(label="System Prompt", value=default_system_prompt)
        with antd.Flex(gap="small", justify="center", align="center"):
            with antd.Flex(vertical=True, gap="small", align="center"):
                antd.Typography.Title(
                    "Qwen2.5-Omni Demo", level=1, elem_style=dict(margin=0, fontSize=28)
                )
                with antd.Flex(vertical=True, gap="small"):
                    antd.Typography.Text(
                        get_text("🎯 Instructions for use:", "🎯 使用说明："), strong=True
                    )
                    antd.Typography.Text(
                        get_text(
                            "1️⃣ Click the Audio Record button or the Camera Record button.",
                            "1️⃣ 点击音频录制按钮，或摄像头-录制按钮",
                        )
                    )
                    antd.Typography.Text(get_text("2️⃣ Input audio or video.", "2️⃣ 输入音频或者视频"))
                    antd.Typography.Text(
                        get_text(
                            "3️⃣ Click the submit button and wait for the model's response.",
                            "3️⃣ 点击提交并等待模型的回答",
                        )
                    )
        voice_choice = gr.Dropdown(label="Voice Choice", choices=VOICE_LIST, value=DEFAULT_VOICE)
        with gr.Tabs():
            with gr.Tab("Online"):
                with gr.Row():
                    with gr.Column(scale=1):
                        microphone = gr.Audio(sources=["microphone"], type="filepath")
                        webcam = gr.Video(sources=["webcam"], height=400, include_audio=True)
                        submit_btn = gr.Button(get_text("Submit", "提交"), variant="primary")
                        stop_btn = gr.Button(get_text("Stop", "停止"), visible=False)
                        clear_btn = gr.Button(get_text("Clear History", "清除历史"))
                    with gr.Column(scale=2):
                        media_chatbot = gr.Chatbot(height=650, type="messages")

                    def clear_history():
                        return [], gr.update(value=None), gr.update(value=None)

                    submit_event = submit_btn.click(
                        fn=media_predict,
                        inputs=[
                            microphone,
                            webcam,
                            media_chatbot,
                            system_prompt_textbox,
                            voice_choice,
                        ],
                        outputs=[microphone, webcam, media_chatbot, submit_btn, stop_btn],
                    )
                    stop_btn.click(
                        fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
                        inputs=None,
                        outputs=[submit_btn, stop_btn],
                        cancels=[submit_event],
                        queue=False,
                    )
                    clear_btn.click(
                        fn=clear_history, inputs=None, outputs=[media_chatbot, microphone, webcam]
                    )

            with gr.Tab("Offline"):
                chatbot = gr.Chatbot(type="messages", height=650)

                # Media upload section in one row
                with gr.Row(equal_height=True):
                    audio_input = gr.Audio(
                        sources=["upload"],
                        type="filepath",
                        label="Upload Audio",
                        elem_classes="media-upload",
                        scale=1,
                    )
                    image_input = gr.Image(
                        sources=["upload"],
                        type="filepath",
                        label="Upload Image",
                        elem_classes="media-upload",
                        scale=1,
                    )
                    video_input = gr.Video(
                        sources=["upload"],
                        label="Upload Video",
                        elem_classes="media-upload",
                        scale=1,
                    )

                # Text input section
                text_input = gr.Textbox(show_label=False, placeholder="Enter text here...")

                # Control buttons
                with gr.Row():
                    submit_btn = gr.Button(get_text("Submit", "提交"), variant="primary", size="lg")
                    stop_btn = gr.Button(get_text("Stop", "停止"), visible=False, size="lg")
                    clear_btn = gr.Button(get_text("Clear History", "清除历史"), size="lg")

                def clear_chat_history():
                    return (
                        [],
                        gr.update(value=None),
                        gr.update(value=None),
                        gr.update(value=None),
                        gr.update(value=None),
                    )

                submit_event = gr.on(
                    triggers=[submit_btn.click, text_input.submit],
                    fn=chat_predict,
                    inputs=[
                        text_input,
                        audio_input,
                        image_input,
                        video_input,
                        chatbot,
                        system_prompt_textbox,
                        voice_choice,
                    ],
                    outputs=[text_input, audio_input, image_input, video_input, chatbot],
                )

                stop_btn.click(
                    fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
                    inputs=None,
                    outputs=[submit_btn, stop_btn],
                    cancels=[submit_event],
                    queue=False,
                )

                clear_btn.click(
                    fn=clear_chat_history,
                    inputs=None,
                    outputs=[chatbot, text_input, audio_input, image_input, video_input],
                )

                # Add some custom CSS to improve the layout
                gr.HTML("""
                    <style>
                        .media-upload {
                            margin: 10px;
                            min-height: 160px;
                        }
                        .media-upload > .wrap {
                            border: 2px dashed #ccc;
                            border-radius: 8px;
                            padding: 10px;
                            height: 100%;
                        }
                        .media-upload:hover > .wrap {
                            border-color: #666;
                        }
                        /* Make upload areas equal width */
                        .media-upload {
                            flex: 1;
                            min-width: 0;
                        }
                    </style>
                """)

    demo.queue(default_concurrency_limit=100, max_size=100).launch(
        max_threads=100,
        ssr_mode=False,
        share=share,
        inbrowser=inbrowser,
        server_port=server_port,
        server_name=server_name,
    )


@app.function(
    gpu=os.getenv("IMAGE_GPU", "L40s"),
    cpu=2.0,
    image=omni_img,
    volumes={
        HF_MODEL_DIR: hf_model_vol,
        ASSETS_DIR: assets_dir,
    },
    timeout=1200,  # default 300s
    scaledown_window=1200,
    max_containers=100,
)
def server(
    ui_language="en",
    server_port="7860",
    server_name="127.0.0.1",
):
    _launch_demo(
        model,
        processor,
        ui_language,
        True,
        False,
        int(server_port),
        server_name,
    )


"""
modal run src/llm/transformers/qwen2_5omni_web_demo.py
"""
