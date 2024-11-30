# the first run need download agora sdk core lib(c++)

# video call: https://webdemo.agora.io/basicVideoCall/index.html
# voice call: https://webdemo.agora.io/basicVoiceCall/index.html

# or use builder to create room and deploy app:  https://appbuilder.agora.io/create

import asyncio
import logging
import os
from typing import Any
from dotenv import load_dotenv

from agora.rtc.rtc_connection import RTCConnection, RTCConnInfo
from agora_realtime_ai_api.rtc import RtcEngine, RtcOptions, Channel

from .utils import PCMWriter

PCM_SAMPLE_RATE = 24000
PCM_CHANNELS = 1

load_dotenv(override=True)
app_id = os.environ.get("AGORA_APP_ID")
app_cert = os.environ.get("AGORA_APP_CERT")

if not app_id:
    raise ValueError("AGORA_APP_ID must be set in the environment.")


def _monitor_queue_size(queue: asyncio.Queue[bytes], queue_name: str, threshold: int = 5) -> None:
    queue_size = queue.qsize()
    if queue_size > threshold:
        logging.warning(f"Queue {queue_name} size exceeded {threshold}: current size {queue_size}")


async def wait_for_remote_user(channel: Channel) -> int:
    remote_users = list(channel.remote_users.keys())
    if len(remote_users) > 0:
        return remote_users[0]

    future = asyncio.Future[int]()

    channel.once("user_joined", lambda conn, user_id: future.set_result(user_id))

    try:
        # Wait for the remote user with a timeout of 30 seconds
        remote_user = await asyncio.wait_for(future, timeout=15.0)
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
            # Get audio frame from rtc
            audio_frame = await in_audio_queue.get()
            logging.info(f"bot_process_audio len: {len(audio_frame.data)}")

            # Put audio data to out
            await out_audio_queue.put_nowait(audio_frame.data)

    except asyncio.CancelledError:
        # Write any remaining PCM data before exiting
        raise  # Re-raise the cancelled exception to properly exit the task


async def rtc_to_bot(
        subscribe_user: int,
        channel: Channel,
        in_audio_queue: asyncio.Queue[bytes],
        out_audio_queue: asyncio.Queue[bytes],
        write_pcm: bool = False) -> None:
    while subscribe_user is None or channel.get_audio_frames(subscribe_user) is None:
        await asyncio.sleep(0.1)

    audio_frames = channel.get_audio_frames(subscribe_user)

    # Initialize PCMWriter for receiving audio
    pcm_writer = PCMWriter(prefix="rtc_to_model", write_pcm=write_pcm)

    try:
        async for audio_frame in audio_frames:
            # Process received audio (send to model)
            _monitor_queue_size(out_audio_queue, "out_audio_queue")

            # Bot process audio
            await bot_process_audio(in_audio_queue, audio_frame.data)

            # Write PCM data if enabled
            await pcm_writer.write(audio_frame.data)

            await asyncio.sleep(0)  # Yield control to allow other tasks to run

    except asyncio.CancelledError:
        # Write any remaining PCM data before exiting
        await pcm_writer.flush()
        raise  # Re-raise the exception to propagate cancellation


async def bot_to_rtc(
        channel: Channel,
        out_audio_queue: asyncio.Queue[bytes],
        write_pcm: bool = False) -> None:
    # Initialize PCMWriter for sending audio
    pcm_writer = PCMWriter(prefix="model_to_rtc", write_pcm=write_pcm)

    try:
        while True:
            # Get audio frame from the model output
            frame = await out_audio_queue.get()

            # Process sending audio (to RTC)
            await channel.push_audio_frame(frame)

            # Write PCM data if enabled
            await pcm_writer.write(frame)

    except asyncio.CancelledError:
        # Write any remaining PCM data before exiting
        await pcm_writer.flush()
        raise  # Re-raise the cancelled exception to properly exit the task


async def run(channel: Channel) -> None:
    try:
        def log_exception(t: asyncio.Task[Any]) -> None:
            if not t.cancelled() and t.exception():
                logging.error(
                    "unhandled exception",
                    exc_info=t.exception(),
                )

        def on_stream_message(agora_local_user, user_id, stream_id, data, length) -> None:
            logging.info(
                f"{agora_local_user} Received stream({stream_id}) message from {user_id} with length: {length}")

        channel.on("stream_message", on_stream_message)

        logging.info("Waiting for remote user to join")
        subscribe_user = await wait_for_remote_user(channel)
        logging.info(f"Subscribing to user {subscribe_user}")
        await channel.subscribe_audio(subscribe_user)

        async def on_user_left(
            agora_rtc_conn: RTCConnection, user_id: int, reason: int
        ):
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
            rtc_to_bot(
                subscribe_user,
                channel,
                in_audio_queue,
                out_audio_queue,
                write_pcm=False)).add_done_callback(log_exception)
        asyncio.create_task(
            bot_to_rtc(
                channel,
                out_audio_queue,
                write_pcm=False)).add_done_callback(log_exception)
        asyncio.create_task(
            bot_process_audio(
                in_audio_queue,
                out_audio_queue)).add_done_callback(log_exception)

        await disconnected_future
        logging.info("Agent finished running")
    except asyncio.CancelledError:
        logging.info("Agent cancelled")
    except Exception as e:
        logging.error(f"Error running agent: {e}")
        raise


async def main():
    engine = RtcEngine(appid=app_id, appcert=app_cert)
    print(engine)

    channel_name = os.environ.get("AGORA_CHANNEL_NAME", "chat-room")
    uid = os.environ.get("AGORA_UID", "weedge")
    options = RtcOptions(
        channel_name=channel_name,
        uid=uid,
        sample_rate=PCM_SAMPLE_RATE,
        channels=PCM_CHANNELS,
        enable_pcm_dump=os.environ.get("WRITE_RTC_PCM", "false") == "true"
    )
    channel = engine.create_channel(options)
    print(channel)
    await channel.connect()
    await run(channel)
    await channel.disconnect()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(funcName)s - %(message)s',
        handlers=[
            logging.StreamHandler()],
    )
    asyncio.run(main())
