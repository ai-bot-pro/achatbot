r"""
TQDM_DISABLE=True python -m src.cmd.local-terminal-chat.generate_audio2audio > ./log/std_out.log

TQDM_DISABLE=True RECORDER_TAG=wakeword_rms_recorder python -m src.cmd.local-terminal-chat.generate_audio2audio > ./log/std_out.log

TQDM_DISABLE=True \
    RECORDER_TAG=wakeword_rms_recorder \
    LLM_MODEL_NAME=qwen-2 \
    LLM_MODEL_PATH=./models/qwen2-1_5b-instruct-q8_0.gguf \
    python -m src.cmd.local-terminal-chat.generate_audio2audio > ./log/std_out.log

TQDM_DISABLE=True \
    TTS_TAG=tts_coqui \
    RECORDER_TAG=wakeword_rms_recorder \
    LLM_MODEL_NAME=qwen \
    LLM_MODEL_PATH=./models/qwen2-1_5b-instruct-q8_0.gguf \
    python -m src.cmd.local-terminal-chat.generate_audio2audio > ./log/std_out.log

TQDM_DISABLE=True \
    TTS_TAG=tts_edge \
    RECORDER_TAG=wakeword_rms_recorder \
    LLM_MODEL_NAME=qwen \
    LLM_MODEL_PATH=./models/qwen2-1_5b-instruct-q8_0.gguf \
    python -m src.cmd.local-terminal-chat.generate_audio2audio > ./log/std_out.log

TQDM_DISABLE=True \
    RECORDER_TAG=wakeword_rms_recorder \
    ASR_TAG=whisper_faster_asr \
    ASR_MODEL_NAME_OR_PATH=./models/Systran/faster-whisper-base \
    python -m src.cmd.local-terminal-chat.generate_audio2audio > ./log/std_out.log

TQDM_DISABLE=True \
    TTS_TAG=tts_edge \
    RECORDER_TAG=rms_recorder \
    ASR_TAG=sense_voice_asr \
    ASR_LANG=zn \
    ASR_MODEL_NAME_OR_PATH=./models/FunAudioLLM/SenseVoiceSmall \
    python -m src.cmd.local-terminal-chat.generate_audio2audio > ./log/std_out.log

TQDM_DISABLE=True \
    TTS_TAG=tts_edge \
    RECORDER_TAG=rms_recorder \
    ASR_TAG=whisper_groq_asr \
    ASR_LANG=zh \
    ASR_MODEL_NAME_OR_PATH=whisper-large-v3 \
    python -m src.cmd.local-terminal-chat.generate_audio2audio > ./log/std_out.log

TQDM_DISABLE=True \
    TTS_TAG=tts_edge \
    VAD_DETECTOR_TAG=webrtc_silero_vad \
    RECORDER_TAG=wakeword_vad_recorder \
    ASR_TAG=whisper_groq_asr \
    ASR_LANG=zh \
    ASR_MODEL_NAME_OR_PATH=whisper-large-v3 \
    python -m src.cmd.local-terminal-chat.generate_audio2audio > ./log/std_out.log

TQDM_DISABLE=True \
    VAD_DETECTOR_TAG=webrtc_silero_vad \
    RECORDER_TAG=wakeword_vad_recorder \
    ASR_TAG=sense_voice_asr \
    ASR_LANG=zn \
    ASR_MODEL_NAME_OR_PATH=./models/FunAudioLLM/SenseVoiceSmall \
    TTS_TAG=tts_edge \
    python -m src.cmd.local-terminal-chat.generate_audio2audio > ./log/std_out.log

TQDM_DISABLE=True \
    VAD_DETECTOR_TAG=webrtc_silero_vad \
    RECORDER_TAG=wakeword_vad_recorder \
    ASR_TAG=sense_voice_asr \
    ASR_LANG=zn \
    ASR_MODEL_NAME_OR_PATH=./models/FunAudioLLM/SenseVoiceSmall \
    LLM_MODEL_NAME=qwen \
    LLM_MODEL_PATH=./models/qwen2-1_5b-instruct-q8_0.gguf \
    TTS_TAG=tts_edge \
    python -m src.cmd.local-terminal-chat.generate_audio2audio > ./log/std_out.log

TQDM_DISABLE=True TOKENIZERS_PARALLELISM=true \
    ASR_TAG=sense_voice_asr \
    ASR_LANG=zn \
    ASR_MODEL_NAME_OR_PATH=./models/FunAudioLLM/SenseVoiceSmall \
    RECORDER_TAG=wakeword_rms_recorder \
    LLM_MODEL_TYPE=chat-func \
    LLM_MODEL_NAME=functionary \
    LLM_MODEL_PATH=./models/meetkai/functionary-small-v2.4-GGUF/functionary-small-v2.4.Q4_0.gguf \
    LLM_TOKENIZER_PATH=./models/meetkai/functionary-small-v2.4-GGUF \
    LLM_CHAT_FORMAT=functionary-v2 \
    LLM_TOOL_CHOICE=auto \
    TTS_TAG=tts_edge \
    python -m src.cmd.local-terminal-chat.generate_audio2audio > ./log/std_out.log

TQDM_DISABLE=True \
    VAD_DETECTOR_TAG=webrtc_silero_vad \
    RECORDER_TAG=wakeword_vad_recorder \
    ASR_TAG=sense_voice_asr \
    ASR_LANG=zn \
    ASR_MODEL_NAME_OR_PATH=./models/FunAudioLLM/SenseVoiceSmall \
    LLM_TAG=llm_personalai_proxy \
    API_URL=https://personal-ai-ts.weedge.workers.dev/ \
    LLM_MODEL_NAME=llama3-70b-8192 \
    TTS_TAG=tts_edge \
    python -m src.cmd.local-terminal-chat.generate_audio2audio > ./log/std_out.log
TQDM_DISABLE=True \
    VAD_DETECTOR_TAG=webrtc_silero_vad \
    RECORDER_TAG=wakeword_vad_recorder \
    ASR_TAG=sense_voice_asr \
    ASR_LANG=zn \
    ASR_MODEL_NAME_OR_PATH=./models/FunAudioLLM/SenseVoiceSmall \
    LLM_TAG=llm_personalai_proxy \
    API_URL=https://personal-ai-ts.weedge.workers.dev/ \
    LLM_MODEL_NAME=llama3-70b-8192 \
    CHAT_TYPE=chat_with_functions \
    TTS_TAG=tts_edge \
    python -m src.cmd.local-terminal-chat.generate_audio2audio > ./log/std_out.log

TQDM_DISABLE=True \
    AUDIO_IN_STREAM_TAG=pyaudio_in_stream \
    AUDIO_OUT_STREAM_TAG=pyaudio_out_stream \
    VAD_DETECTOR_TAG=webrtc_silero_vad \
    RECORDER_TAG=vad_recorder \
    ASR_TAG=sense_voice_asr \
    ASR_LANG=zn \
    ASR_MODEL_NAME_OR_PATH=./models/FunAudioLLM/SenseVoiceSmall \
    LLM_TAG=llm_personalai_proxy \
    API_URL=https://personal-ai-ts.weedge.workers.dev/ \
    LLM_MODEL_NAME=llama-3.1-70b-versatile \
    CHAT_TYPE=chat_with_functions \
    TTS_TAG=tts_edge \
    python -m src.cmd.local-terminal-chat.generate_audio2audio > ./log/std_out.log

TQDM_DISABLE=True \
    AUDIO_IN_STREAM_TAG=pyaudio_in_stream \
    AUDIO_OUT_STREAM_TAG=daily_room_audio_out_stream \
    MEETING_ROOM_URL=https://weedge.daily.co/chat-bot \
    VAD_DETECTOR_TAG=webrtc_silero_vad \
    RECORDER_TAG=vad_recorder \
    ASR_TAG=sense_voice_asr \
    ASR_LANG=zn \
    ASR_MODEL_NAME_OR_PATH=./models/FunAudioLLM/SenseVoiceSmall \
    LLM_TAG=llm_personalai_proxy \
    API_URL=https://personal-ai-ts.weedge.workers.dev/ \
    LLM_MODEL_NAME=llama-3.1-70b-versatile \
    CHAT_TYPE=chat_with_functions \
    TTS_TAG=tts_edge \
    python -m src.cmd.local-terminal-chat.generate_audio2audio > ./log/std_out.log

TQDM_DISABLE=True \
    AUDIO_IN_STREAM_TAG=daily_room_audio_in_stream \
    AUDIO_OUT_STREAM_TAG=pyaudio_out_stream \
    MEETING_ROOM_URL=https://weedge.daily.co/chat-bot \
    VAD_DETECTOR_TAG=webrtc_silero_vad \
    RECORDER_TAG=vad_recorder \
    ASR_TAG=sense_voice_asr \
    ASR_LANG=zn \
    ASR_MODEL_NAME_OR_PATH=./models/FunAudioLLM/SenseVoiceSmall \
    LLM_TAG=llm_personalai_proxy \
    API_URL=https://personal-ai-ts.weedge.workers.dev/ \
    LLM_MODEL_NAME=llama-3.1-70b-versatile \
    CHAT_TYPE=chat_with_functions \
    TTS_TAG=tts_edge \
    python -m src.cmd.local-terminal-chat.generate_audio2audio > ./log/std_out.log

TQDM_DISABLE=True \
    AUDIO_IN_STREAM_TAG=daily_room_audio_in_stream \
    AUDIO_OUT_STREAM_TAG=daily_room_audio_out_stream \
    MEETING_ROOM_URL=https://weedge.daily.co/chat-bot \
    VAD_DETECTOR_TAG=webrtc_silero_vad \
    RECORDER_TAG=vad_recorder \
    ASR_TAG=sense_voice_asr \
    ASR_LANG=zn \
    ASR_MODEL_NAME_OR_PATH=./models/FunAudioLLM/SenseVoiceSmall \
    LLM_TAG=llm_personalai_proxy \
    API_URL=https://personal-ai-ts.weedge.workers.dev/ \
    LLM_MODEL_NAME=llama-3.1-70b-versatile \
    CHAT_TYPE=chat_with_functions \
    TTS_TAG=tts_edge \
    python -m src.cmd.local-terminal-chat.generate_audio2audio > ./log/std_out.log

TQDM_DISABLE=True TOKENIZERS_PARALLELISM=false \
    AUDIO_IN_STREAM_TAG=pyaudio_in_stream \
    AUDIO_OUT_STREAM_TAG=pyaudio_out_stream \
    VAD_DETECTOR_TAG=webrtc_silero_vad \
    RECORDER_TAG=vad_recorder \
    ASR_TAG=sense_voice_asr \
    ASR_LANG=zn \
    ASR_MODEL_NAME_OR_PATH=./models/FunAudioLLM/SenseVoiceSmall \
    LLM_TAG=llm_transformers_manual \
    LLM_MODEL_NAME_OR_PATH=./models/Qwen/Qwen2-0.5B-Instruct \
    TTS_TAG=tts_edge \
    python -m src.cmd.local-terminal-chat.generate_audio2audio > ./log/std_out.log

TQDM_DISABLE=True TOKENIZERS_PARALLELISM=false \
    AUDIO_IN_STREAM_TAG=pyaudio_in_stream \
    AUDIO_OUT_STREAM_TAG=pyaudio_out_stream \
    VAD_DETECTOR_TAG=webrtc_silero_vad \
    RECORDER_TAG=vad_recorder \
    ASR_TAG=sense_voice_asr \
    ASR_LANG=zn \
    ASR_MODEL_NAME_OR_PATH=./models/FunAudioLLM/SenseVoiceSmall \
    LLM_TAG=llm_transformers_pipeline \
    LLM_MODEL_NAME_OR_PATH=./models/Qwen/Qwen2-0.5B-Instruct \
    TTS_TAG=tts_edge \
    python -m src.cmd.local-terminal-chat.generate_audio2audio > ./log/std_out.log

INIT_TYPE=yaml_config TQDM_DISABLE=True \
    python -m src.cmd.local-terminal-chat.generate_audio2audio > ./log/std_out.log
"""

import multiprocessing
import logging
import os

from src.common.logger import Logger
from src.common.connector.multiprocessing_pipe import MultiprocessingPipeConnector
from src.cmd.be import Audio2AudioChatWorker as ChatWorker
from src.cmd.fe import TerminalChatClient


# global logging
Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=True, is_console=False)


def main():
    mp_conn = MultiprocessingPipeConnector()

    # BE
    be_init_event = multiprocessing.Event()
    c = multiprocessing.Process(target=ChatWorker().run, args=(mp_conn, be_init_event), daemon=True)
    c.start()
    be_init_event.wait()

    # FE
    TerminalChatClient().run(mp_conn)

    if c.is_alive():
        logging.info(f"process {c.pid} Exiting")
        c.join()
        c.terminate()
        c.close()
        logging.info("process Exited!")

    mp_conn.close()


if __name__ == "__main__":
    main()
