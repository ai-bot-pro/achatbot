import asyncio
import logging
from enum import Enum
from typing import Optional, Callable
import contextlib


ERROR_INSTALL_INSTRUCTIONS = """
FFmpeg is not installed or not found in your system's PATH.
Please install FFmpeg to enable audio processing.

Installation instructions:

# Ubuntu/Debian:
sudo apt update && sudo apt install ffmpeg

# macOS (using Homebrew):
brew install ffmpeg

# Windows:
# 1. Download the latest static build from https://ffmpeg.org/download.html
# 2. Extract the archive (e.g., to C:\\FFmpeg).
# 3. Add the 'bin' directory (e.g., C:\\FFmpeg\\bin) to your system's PATH environment variable.

After installation, please restart the application.
"""


class FFmpegState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    RESTARTING = "restarting"
    FAILED = "failed"


class FFmpegPiper:
    def __init__(
        self,
        in_sample_rate: int = 16000,
        in_channels: int = 1,
        in_sample_width: int = 2,
        in_format: str = "s16le",  # 设置输出格式为有符号16位小端序PCM
        in_acodec="pcm_s16le",  # 设置音频编码器为PCM 16位小端序
        out_sample_rate: int = 16000,
        out_channels: int = 1,
        out_sample_width: int = 2,
        out_format: str = "s16le",
        out_acodec="pcm_s16le",
    ):
        self.in_sample_rate = in_sample_rate
        self.in_channels = in_channels
        self.in_sample_width = in_sample_width
        self.in_format = in_format
        self.in_acodec = in_acodec
        self.out_sample_rate = out_sample_rate
        self.out_channels = out_channels
        self.out_sample_width = out_sample_width
        self.out_format = out_format
        self.out_acodec = out_acodec

        self.process: Optional[asyncio.subprocess.Process] = None
        self._stderr_task: Optional[asyncio.Task] = None

        self.on_error_callback: Optional[Callable[[str], None]] = None

        self.state = FFmpegState.STOPPED
        self._state_lock = asyncio.Lock()

        self.read_timeout = 3.0

    async def start(self) -> bool:
        async with self._state_lock:
            if self.state != FFmpegState.STOPPED:
                logging.warning(f"FFmpeg already running in state: {self.state}")
                return False
            self.state = FFmpegState.STARTING

        try:
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "debug",
                "-probesize",
                "32",  # 将探测大小设置为允许的最小值（32字节），以减少初始缓冲
                "-analyzeduration",
                "0",  # 不花时间分析流的持续时间。
                # Input
                "-acodec",
                self.in_acodec,
                "-f",
                self.in_format,
                "-ac",
                str(self.in_channels),
                "-ar",
                str(self.in_sample_rate),
                "-i",
                "pipe:0",
                # Output
                "-f",
                self.out_format,
                "-acodec",
                self.out_acodec,
                "-ac",
                str(self.out_channels),
                "-ar",
                str(self.out_sample_rate),
                "-flush_packets",
                "1",  # 立即输出解码包
                "pipe:1",
            ]

            print(" ".join(cmd))
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            self._stderr_task = asyncio.create_task(self._drain_stderr())

            async with self._state_lock:
                self.state = FFmpegState.RUNNING

            logging.info("FFmpeg started.")
            return True

        except FileNotFoundError:
            logging.error(ERROR_INSTALL_INSTRUCTIONS)
            async with self._state_lock:
                self.state = FFmpegState.FAILED
            if self.on_error_callback:
                await self.on_error_callback("ffmpeg_not_found")
            return False

        except Exception as e:
            logging.error(f"Error starting FFmpeg: {e}")
            async with self._state_lock:
                self.state = FFmpegState.FAILED
            if self.on_error_callback:
                await self.on_error_callback("start_failed")
            return False

    async def stop(self):
        async with self._state_lock:
            if self.state == FFmpegState.STOPPED:
                return
            self.state = FFmpegState.STOPPED

        if self.process:
            if self.process.stdin and not self.process.stdin.is_closing():
                self.process.stdin.close()
                await self.process.stdin.wait_closed()
            await self.process.wait()
            self.process = None

        if self._stderr_task:
            self._stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stderr_task

        logging.info("FFmpeg stopped.")

    async def write_data(self, data: bytes) -> bool:
        async with self._state_lock:
            if self.state != FFmpegState.RUNNING:
                logging.warning(f"Cannot write, FFmpeg state: {self.state}")
                return False

        try:
            self.process.stdin.write(data)
            await self.process.stdin.drain()
            return True
        except Exception as e:
            logging.error(f"Error writing to FFmpeg: {e}")
            if self.on_error_callback:
                await self.on_error_callback("write_error")
            return False

    async def read_data(self, size: int) -> Optional[bytes]:
        async with self._state_lock:
            if self.state != FFmpegState.RUNNING:
                logging.warning(f"Cannot read, FFmpeg state: {self.state}")
                return None

        try:
            # 读取完整的指定大小数据
            buffer = bytearray()
            remaining = size

            start_time = asyncio.get_event_loop().time()

            while remaining > 0:
                # 计算剩余超时时间
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= self.read_timeout:
                    raise asyncio.TimeoutError("Read operation timed out")

                # 读取剩余数据
                chunk = await asyncio.wait_for(
                    self.process.stdout.read(remaining), timeout=self.read_timeout - elapsed
                )

                # 如果读取到EOF (chunk为空)，则退出循环
                if not chunk:
                    logging.debug(f"EOF reached after reading {len(buffer)} bytes")
                    break

                buffer.extend(chunk)
                remaining -= len(chunk)

            return bytes(buffer)
        except asyncio.TimeoutError:
            logging.warning(
                f"FFmpeg read timeout after reading {len(buffer) if 'buffer' in locals() else 0} bytes."
            )
            return None if "buffer" not in locals() or not buffer else bytes(buffer)
        except Exception as e:
            logging.error(f"Error reading from FFmpeg: {e}")
            if self.on_error_callback:
                await self.on_error_callback("read_error")
            return None

    async def get_state(self) -> FFmpegState:
        async with self._state_lock:
            return self.state

    async def restart(self) -> bool:
        async with self._state_lock:
            if self.state == FFmpegState.RESTARTING:
                logging.warning("Restart already in progress.")
                return False
            self.state = FFmpegState.RESTARTING

        logging.info("Restarting FFmpeg...")

        try:
            await self.stop()
            await asyncio.sleep(1)  # short delay before restarting
            return await self.start()
        except Exception as e:
            logging.error(f"Error during FFmpeg restart: {e}")
            async with self._state_lock:
                self.state = FFmpegState.FAILED
            if self.on_error_callback:
                await self.on_error_callback("restart_failed")
            return False

    async def _drain_stderr(self):
        try:
            while True:
                line = await self.process.stderr.readline()
                if not line:
                    break
                logging.debug(f"FFmpeg stderr: {line.decode(errors='ignore').strip()}")
        except asyncio.CancelledError:
            logging.info("FFmpeg stderr drain task cancelled.")
        except Exception as e:
            logging.error(f"Error draining FFmpeg stderr: {e}")


"""
python -m src.common.utils.ffmpeg_piper
"""
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    piper = FFmpegPiper()

    async def run():
        res = await piper.start()
        assert res is True

        size = 4096
        bytes_data = b"\x00" * int(size * 2.5)
        res = await piper.write_data(bytes_data)
        assert res is True
        read_bytes = await piper.read_data(size)
        print(len(read_bytes))
        assert len(read_bytes) == size
        assert await piper.get_state() == FFmpegState.RUNNING

        read_bytes = await piper.read_data(size)
        print(len(read_bytes))
        assert len(read_bytes) == size
        assert await piper.get_state() == FFmpegState.RUNNING

        read_bytes = await piper.read_data(size)
        print(len(read_bytes))
        assert len(read_bytes) == size / 2
        assert await piper.get_state() == FFmpegState.RUNNING

        await piper.stop()
        assert await piper.get_state() == FFmpegState.STOPPED

    asyncio.run(run())
