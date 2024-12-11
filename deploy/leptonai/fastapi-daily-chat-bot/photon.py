from leptonai.photon import Photon
from achatbot.cmd.http.server import fastapi_daily_bot_serve as serve


class FastapiDailyChatBot(Photon):
    r"""
    use cmd to run server, need metrics method, but not use it(for qps, latency, etc). need to fix (@leptonai)
    """

    cmd = [
        "bash",
        "-c",
        (
            "apt update && apt install -y ffmpeg net-tools "
            "&& pip install -U 'achatbot[fastapi_bot_server, daily_rtvi_bot, daily_langchain_rag_bot, speech_vad_analyzer, asr_processor, tts_processor]' "
            "&& python -m achatbot.cmd.http.server.fastapi_daily_bot_serve "
            "--host 0.0.0.0 --port 8080"
        ),
    ]
    deployment_template = {
        "resource_shape": "cpu.small",  # use cpu for api model
        # "resource_shape": "gpu.t4",  # use gpu for local model
    }

    requirement_dependency = [
        # "achatbot[fastapi_bot_server, daily_rtvi_bot, daily_langchain_rag_bot, speech_vad_analyzer, asr_processor, tts_processor]",
        # "git+https://github.com/facebookresearch/nougat.git@84b3ae1",
    ]

    system_dependency = [
        # "ffmpeg",
        # "net-tools",
    ]
