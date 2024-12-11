# the first run need download agora sdk core lib(c/c++), maybe need rust reconstruct :)

# video call: https://webdemo.agora.io/basicVideoCall/index.html
# join video channel url:
# https://webdemo.agora.io/basicVideoCall/index.html?appid={APP_ID}&channel={CHANNEL_ID}&token={TOKEN}&uid={UID}

# voice call: https://webdemo.agora.io/basicVoiceCall/index.html
# join voice channel url:
# https://webdemo.agora.io/basicVoiceCall/index.html?appid={APP_ID}&channel={CHANNEL_ID}&token={TOKEN}&uid={UID}

# or use builder to create room and deploy app:  https://appbuilder.agora.io/create

import asyncio
import logging
import os
from signal import SIGINT, SIGTERM, signal, strsignal
from typing import Any
from urllib.parse import quote
from dotenv import load_dotenv

from agora.rtc.rtc_connection import RTCConnection, RTCConnInfo
from agora_realtime_ai_api.rtc import RtcEngine, RtcOptions, Channel

from .utils import PCMWriter

PCM_SAMPLE_RATE = 24000
PCM_CHANNELS = 1


def get_voice_demo_channel_url(
    app_id: str, channel_name: str, token: str = "", uid: str = ""
) -> str:
    demo_url = "https://webdemo.agora.io/basicVoiceCall/index.html"
    url = f"{demo_url}?appid={quote(app_id)}&channel={quote(channel_name)}&token={quote(token)}&uid={quote(uid)}"
    return url


def _monitor_queue_size(queue: asyncio.Queue[bytes], queue_name: str, threshold: int = 5) -> None:
    queue_size = queue.qsize()
    if queue_size > threshold:
        logging.warning(f"Queue {queue_name} size exceeded {threshold}: current size {queue_size}")


async def wait_for_remote_user(channel: Channel, timeout_s: float | None = None) -> int:
    remote_users = list(channel.remote_users.keys())
    if len(remote_users) > 0:
        return remote_users[0]

    future = asyncio.Future[int]()

    channel.once("user_joined", lambda conn, user_id: future.set_result(user_id))

    try:
        # Wait for the remote user with a timeout_s, timeout_s is None , no wait timeout
        remote_user = await asyncio.wait_for(future, timeout=timeout_s)
        return remote_user
    except KeyboardInterrupt:
        future.cancel()

    except Exception as e:
        logging.error(f"Error waiting for remote user: {e}")
        raise


async def bot_process_audio(
    in_audio_queue: asyncio.Queue[bytes],
    out_audio_queue: asyncio.Queue[bytes],
) -> None:
    """
    dont't to process, just echo audio
    """

    try:
        while True:
            # Get audio frame data(bytes) from rtc
            frame_data = await in_audio_queue.get()
            logging.debug(f"bot_process_audio len: {len(frame_data)}")

            # Put audio data to out
            await out_audio_queue.put(frame_data)

    except asyncio.CancelledError:
        # Write any remaining PCM data before exiting
        raise  # Re-raise the cancelled exception to properly exit the task


async def rtc_to_bot(
    subscribe_user: int,
    channel: Channel,
    in_audio_queue: asyncio.Queue[bytes],
    out_audio_queue: asyncio.Queue[bytes],
    write_pcm: bool = False,
) -> None:
    while subscribe_user is None or channel.get_audio_frames(subscribe_user) is None:
        await asyncio.sleep(0.1)

    audio_frames = channel.get_audio_frames(subscribe_user)

    # Initialize PCMWriter for receiving audio
    pcm_writer = PCMWriter(prefix="rtc_to_model", write_pcm=write_pcm)

    try:
        async for audio_frame in audio_frames:
            # Process received audio (send to model)
            _monitor_queue_size(out_audio_queue, "out_audio_queue")

            # put audio to in queue for Bot process audio
            await in_audio_queue.put(audio_frame.data)

            # Write PCM data if enabled
            await pcm_writer.write(audio_frame.data)

            await asyncio.sleep(0)  # Yield control to allow other tasks to run

    except asyncio.CancelledError:
        # Write any remaining PCM data before exiting
        await pcm_writer.flush()
        raise  # Re-raise the exception to propagate cancellation


async def bot_to_rtc(
    channel: Channel, out_audio_queue: asyncio.Queue[bytes], write_pcm: bool = False
) -> None:
    # Initialize PCMWriter for sending audio
    pcm_writer = PCMWriter(prefix="model_to_rtc", write_pcm=write_pcm)

    try:
        while True:
            # Get audio frame from the model output
            frame_data = await out_audio_queue.get()

            # Process sending audio (to RTC)
            await channel.push_audio_frame(frame_data)

            # Write PCM data if enabled
            await pcm_writer.write(frame_data)

    except asyncio.CancelledError:
        # Write any remaining PCM data before exiting
        await pcm_writer.flush()
        raise  # Re-raise the cancelled exception to properly exit the task


async def run(channel: Channel, join_url: str) -> None:
    try:

        def log_exception(t: asyncio.Task[Any]) -> None:
            if not t.cancelled() and t.exception():
                logging.error(
                    "unhandled exception",
                    exc_info=t.exception(),
                )

        def on_stream_message(agora_local_user, user_id, stream_id, data, length) -> None:
            logging.info(
                f"{agora_local_user} Received stream({stream_id}) message from {user_id} with length: {length}"
            )

        channel.on("stream_message", on_stream_message)

        logging.info(
            f"Waiting for remote user to join, u can use demo voice channel url: {join_url}"
        )
        subscribe_user = await wait_for_remote_user(channel)
        logging.info(f"Subscribing to user {subscribe_user}")
        await channel.subscribe_audio(subscribe_user)

        async def on_user_left(agora_rtc_conn: RTCConnection, user_id: int, reason: int):
            nonlocal subscribe_user
            logging.info(f"User left: {user_id}")
            if subscribe_user == user_id:
                subscribe_user = None
                logging.info("Subscribed user left, disconnecting")
                await channel.disconnect()

        channel.on("user_left", on_user_left)

        disconnected_future = asyncio.Future[None]()

        def callback(agora_rtc_conn: RTCConnection, conn_info: RTCConnInfo, reason):
            logging.info(f"Connection state changed: {conn_info.state}")
            if conn_info.state == 1:
                if not disconnected_future.done():
                    disconnected_future.set_result(None)

        channel.on("connection_state_changed", callback)

        out_audio_queue = asyncio.Queue[bytes]()
        in_audio_queue = asyncio.Queue[bytes]()
        asyncio.create_task(
            rtc_to_bot(subscribe_user, channel, in_audio_queue, out_audio_queue, write_pcm=False)
        ).add_done_callback(log_exception)
        asyncio.create_task(
            bot_to_rtc(channel, out_audio_queue, write_pcm=False)
        ).add_done_callback(log_exception)
        asyncio.create_task(bot_process_audio(in_audio_queue, out_audio_queue)).add_done_callback(
            log_exception
        )

        await disconnected_future
        logging.info("Agent finished running")
    except asyncio.CancelledError:
        logging.info("Agent cancelled")
    except Exception as e:
        logging.error(f"Error running agent: {e}")
        raise


def handle_agent_proc_signal(signum, frame):
    logging.info(f"Agent process received signal {strsignal(signum)} {frame}. Exiting...")
    os._exit(0)


async def main():
    load_dotenv(override=True)
    app_id = os.environ.get("AGORA_APP_ID")
    app_cert = os.environ.get("AGORA_APP_CERT")

    if not app_id:
        raise ValueError("AGORA_APP_ID must be set in the environment.")

    engine = RtcEngine(appid=app_id, appcert=app_cert)

    channel_name = os.environ.get("AGORA_CHANNEL_NAME", "chat-room")
    uid = int(os.environ.get("AGORA_UID", "0"))
    options = RtcOptions(
        channel_name=channel_name,
        uid=uid,
        sample_rate=PCM_SAMPLE_RATE,
        channels=PCM_CHANNELS,
        enable_pcm_dump=os.environ.get("WRITE_RTC_PCM", "false") == "true",
    )
    channel = engine.create_channel(options)
    join_url = get_voice_demo_channel_url(app_id, channel_name, token=channel.token)
    await channel.connect()
    await run(channel, join_url)
    await channel.disconnect()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(funcName)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    signal(SIGINT, handle_agent_proc_signal)  # Forward SIGINT
    signal(SIGTERM, handle_agent_proc_signal)  # Forward SIGTERM

    asyncio.run(main())
