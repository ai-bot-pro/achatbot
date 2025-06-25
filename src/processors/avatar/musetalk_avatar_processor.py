import os
import time
import logging
import asyncio
from typing import AsyncGenerator

import av
import cv2
import librosa
import numpy as np
import soundfile as sf
import torch
from apipeline.frames import Frame, StartFrame, EndFrame, CancelFrame

from src.modules.avatar.musetalk import MusetalkAvatar
from src.processors.avatar.base import AvatarProcessorBase, SegmentedAvatarProcessor
from src.types.frames import AudioRawFrame, OutputAudioRawFrame, OutputImageRawFrame
from src.types.avatar.musetalk import AvatarMuseTalkConfig
from src.types.avatar import AvatarStatus, SpeechAudio
from src.types.avatar.lite_avatar import AudioResult, VideoResult
from src.processors.avatar.help.speech_audio_slicer import SpeechAudioSlicer
from src.common.utils.audio_utils import bytes2NpArrayWith16


# class MusetalkAvatarProcessor(AvatarProcessorBase):
class MusetalkAvatarProcessor(SegmentedAvatarProcessor):
    def __init__(self, avatar: MusetalkAvatar, config: AvatarMuseTalkConfig, **kwargs):
        self._avatar = avatar
        self._config = config
        super().__init__(sample_rate=config.algo_audio_sample_rate, **kwargs)
        logging.info(f"init {__name__} init_config: {self._config}")

        # Internal algorithm sample rate, fixed at 16000
        self._algo_audio_sample_rate = config.algo_audio_sample_rate
        self._output_audio_sample_rate = config.output_audio_sample_rate

        # Internal queues
        self._audio_queue: asyncio.Queue = None  # Input audio queue
        self._whisper_queue: asyncio.Queue = None  # Whisper feature queue
        self._frame_id_queue: asyncio.Queue = None  # Frame ID allocation queue
        self._compose_queue: asyncio.Queue = None  # Frame composition queue
        self._output_queue: asyncio.Queue = None  # Output queue after composition

        # task
        self._feature_task: asyncio.Task = None
        self._frame_gen_task: asyncio.Task = None
        self._frame_collect_task: asyncio.Task = None
        self._compose_task: asyncio.Task = None
        self._session_running = False

        # warmup event
        self._feature_extractor_warmup_event = asyncio.Event()
        self._frame_generator_warmup_event = asyncio.Event()

        # Avatar status
        self._callback_avatar_status = AvatarStatus.LISTENING
        self._last_speech_id = None

        # Audio duration statistics
        self._first_add_audio_time = None
        self._audio_duration_sum = 0.0

        # Audio cache for each speech_id
        self._audio_cache = {}

        # audio input slice
        self._speech_audio_slicer: SpeechAudioSlicer = None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._session_running = True
        self._init()
        await self._start_tasks()
        self._session_start_time = time.time()
        logging.info(f"{__name__} started")

    def _init(self):
        self._speech_audio_slicer = SpeechAudioSlicer(
            input_sample_rate=self._config.input_audio_sample_rate,  # input
            output_sample_rate=self._config.algo_audio_sample_rate,  # output for avatar input audio sample rate
            audio_slice_duration=self._config.input_audio_slice_duration,
        )

    async def _start_tasks(self):
        self._audio_queue = asyncio.Queue()  # Input audio queue
        self._whisper_queue = asyncio.Queue()  # Whisper feature queue
        self._frame_id_queue = asyncio.Queue()  # Frame ID allocation queue
        self._compose_queue = asyncio.Queue()  # Frame composition queue
        self._output_queue = asyncio.Queue()  # Output queue after composition

        self._feature_task: asyncio.Task = self.get_event_loop().create_task(
            self._feature_extractor_loop()
        )
        self._frame_gen_task: asyncio.Task = self.get_event_loop().create_task(
            self._frame_generator_loop()
        )
        self._frame_collect_task: asyncio.Task = self.get_event_loop().create_task(
            self._frame_collector_loop()
        )
        self._compose_task: asyncio.Task = self.get_event_loop().create_task(self._compose_loop())

        await self._feature_extractor_warmup_event.wait()
        self._feature_extractor_warmup_event.clear()

        await self._frame_generator_warmup_event.wait()
        self._frame_generator_warmup_event.clear()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._stop()

    async def cancle(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._stop()

    async def _stop(self):
        logging.info(
            f"stop avatar processor {__name__}, totol session time {time.time() - self._session_start_time:.3f}",
        )
        self._feature_extractor_warmup_event.clear()
        self._frame_generator_warmup_event.clear()
        self._session_running = False
        if self._feature_task is not None:
            self._feature_task.cancel()
            await self._feature_task
            self._signal2img_task = None
        if self._frame_gen_task is not None:
            self._frame_gen_task.cancel()
            await self._frame_gen_task
            self._frame_gen_task = None
        if self._frame_collect_task is not None:
            self._frame_collect_task.cancel()
            await self._frame_collect_task
            self._frame_collect_task = None
        if self._compose_task is not None:
            self._compose_task.cancel()
            await self._compose_task
            self._compose_task = None
        logging.info(f"avatar processor {__name__} stopped")

    async def run_avatar(self, frame: AudioRawFrame) -> AsyncGenerator[Frame, None]:
        await self._add_audio(
            SpeechAudio(
                end_of_speech=True,
                speech_id="",
                sample_rate=frame.sample_rate,
                audio_data=frame.audio,
            )
        )

        # inner push/queue audio/image frame, yield None
        yield None

    async def _add_audio(self, speech_audio: SpeechAudio):
        """
        Add an audio segment to the processing queue. The segment length must not exceed 1 second. No resampling is performed here.
        Args:
            speech_audio (SpeechAudio): Audio segment to add.
        """
        if self._config.debug:
            now = time.time()
            # Record the first add_audio time
            if self._first_add_audio_time is None:
                self._first_add_audio_time = now
            # Calculate audio duration
            audio_len = len(speech_audio.audio_data)
            sample_rate = speech_audio.sample_rate
            audio_duration = audio_len / 4 / sample_rate  # float32, 4 bytes
            self._audio_duration_sum += audio_duration
            # Calculate cumulative interval
            total_interval = now - self._first_add_audio_time
            # Log output
            log_msg = (
                f"Received add_audio: speech_id={speech_audio.speech_id}, end_of_speech={speech_audio.end_of_speech}, "
                f"sample_rate={sample_rate}, audio_len={audio_len}, audio_duration={audio_duration:.3f}s, "
                f"cumulative_audio_duration={self._audio_duration_sum:.3f}s, total_interval={total_interval:.3f}s"
            )
            if self._audio_duration_sum < total_interval:
                logging.error(
                    log_msg + " [Cumulative audio duration < total interval, audio is too slow!]"
                )
            else:
                logging.info(log_msg)
            # Output cumulative duration and interval when end_of_speech is reached and reset
            if speech_audio.end_of_speech:
                logging.info(
                    f"[add_audio] speech_id={speech_audio.speech_id} segment cumulative_audio_duration: {self._audio_duration_sum:.3f}s, total_interval: {total_interval:.3f}s"
                )
                self._audio_duration_sum = 0.0
                self._first_add_audio_time = None

        audio_slices = self._speech_audio_slicer.get_speech_audio_slice(speech_audio)
        if self._config.debug:
            logging.info(f"audio_slices_len: {len(audio_slices)}")
        for audio_slice in audio_slices:
            if self._config.debug:
                logging.info(f"audio_slice: {str(audio_slice)}")

            audio_data = audio_slice.algo_audio_data

            # if len(audio_data) == 0:
            #    logging.error(f"Input audio is empty, speech_id={audio_slice.speech_id}")
            #    return

            ## Length check, process 1s  audio
            # if len(audio_data) > self._output_audio_sample_rate:
            #    logging.error(
            #        f"Audio segment too long: {len(audio_data)} > {self._algo_audio_sample_rate}, speech_id={audio_slice.speech_id}"
            #    )
            #    return

            assert speech_audio.sample_rate == self._output_audio_sample_rate, (
                f"{speech_audio.sample_rate=} != {self._output_audio_sample_rate=}"
            )

            await self._audio_queue.put(
                {
                    "audio_data": audio_data,  # Segment at algorithm sample rate (actually original segment)
                    "speech_id": audio_slice.speech_id,
                    "end_of_speech": audio_slice.end_of_speech,
                }
            )

    async def _feature_extractor_loop(self):
        """
        Worker thread for extracting audio features.
        """
        # warmup: ensure CUDA context and memory allocation (for whisper feature extraction)
        if torch.cuda.is_available():
            t0 = time.time()
            warmup_sr = 16000
            dummy_audio = np.zeros(warmup_sr, dtype=np.float32)
            await asyncio.to_thread(self._avatar.extract_whisper_feature, dummy_audio, warmup_sr)
            torch.cuda.synchronize()
            t1 = time.time()
            logging.info(
                f"_feature_extractor_loop whisper feature warmup once done, time: {(t1 - t0) * 1000:.1f} ms"
            )
            self._feature_extractor_warmup_event.set()

        while self._session_running is True:
            try:
                t_start = time.time()
                item = await asyncio.wait_for(self._audio_queue.get(), timeout=0.1)
                audio_data = item["audio_data"]
                speech_id = item["speech_id"]
                end_of_speech = item["end_of_speech"]
                fps = self._config.fps if hasattr(self._config, "fps") else 25

                segment = bytes2NpArrayWith16(audio_data)
                # Resample to algorithm sample rate
                if self._output_audio_sample_rate != self._algo_audio_sample_rate:
                    segment = librosa.resample(
                        segment,
                        orig_sr=self._output_audio_sample_rate,
                        target_sr=self._algo_audio_sample_rate,
                    )
                target_len = self._algo_audio_sample_rate  # 1 second
                if len(segment) < target_len:
                    segment = np.pad(segment, (0, target_len - len(segment)), mode="constant")

                # Feature extraction
                whisper_chunks = await asyncio.to_thread(
                    self._avatar.extract_whisper_feature, segment, self._algo_audio_sample_rate
                )

                audio_data = np.frombuffer(audio_data, dtype=np.float32)
                orig_audio_data_len = len(audio_data)
                orig_samples_per_frame = self._output_audio_sample_rate // fps
                actual_audio_len = orig_audio_data_len
                num_frames = int(np.ceil(actual_audio_len / orig_samples_per_frame))
                whisper_chunks = whisper_chunks[:num_frames]
                target_audio_len = num_frames * orig_samples_per_frame
                if len(audio_data) < target_audio_len:
                    audio_data = np.pad(
                        audio_data, (0, target_audio_len - len(audio_data)), mode="constant"
                    )
                else:
                    audio_data = audio_data[:target_audio_len]
                padded_audio_data_len = len(audio_data)
                await self._whisper_queue.put(
                    {
                        "whisper_chunks": whisper_chunks,
                        "speech_id": speech_id,
                        "end_of_speech": end_of_speech,
                        "audio_data": audio_data,
                    }
                )
                t_end = time.time()
                if self._config.debug:
                    logging.info(
                        f"[FEATURE_WORKER] speech_id={speech_id}, total_time={(t_end - t_start) * 1000:.1f}ms, whisper_chunks_frames={whisper_chunks.shape[0]}, audio_data_original_length={orig_audio_data_len}, audio_data_padded_length={padded_audio_data_len}, end_of_speech={end_of_speech}"
                    )
            except asyncio.CancelledError:
                logging.warning("feature_extractor_loop task cancelled")
                break
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.exception(f"feature_extractor_loop task error: {e}")
                continue
        logging.info("feature_extractor_loop stopped")

    def _get_audio_for_frame(self, audio_data, chunk_idx, num_chunks, orig_samples_per_frame):
        try:
            start_sample = chunk_idx * orig_samples_per_frame
            if chunk_idx == num_chunks - 1:
                audio_for_frame = audio_data[start_sample:]
                if len(audio_for_frame) < orig_samples_per_frame:
                    audio_for_frame = np.pad(
                        audio_for_frame,
                        (0, orig_samples_per_frame - len(audio_for_frame)),
                        mode="constant",
                    )
            else:
                end_sample = start_sample + orig_samples_per_frame
                audio_for_frame = audio_data[start_sample:end_sample]
            if len(audio_for_frame) < orig_samples_per_frame:
                logging.warning(
                    f"[AUDIO_PAD] Frame audio padding: {len(audio_for_frame)}->{orig_samples_per_frame}, chunk_idx={chunk_idx}"
                )
                audio_for_frame = np.pad(
                    audio_for_frame,
                    (0, orig_samples_per_frame - len(audio_for_frame)),
                    mode="constant",
                )
            return audio_for_frame
        except Exception as e:
            logging.error(
                f"[AUDIO_FOR_FRAME_ERROR] chunk_idx={chunk_idx}, error: {e}", exc_info=True
            )
            return np.zeros(orig_samples_per_frame, dtype=np.float32)

    async def _frame_generator_loop(self):
        """
        Generate speaking frames only, with rate control when queue buffer is full.
        Uses global frame_id allocation to ensure unique and continuous frame numbering for speaking frames.
        """
        from collections import namedtuple

        CurrentSpeechItem = namedtuple(
            "CurrentSpeechItem",
            ["whisper_chunks", "audio_data", "speech_id", "end_of_speech", "num_chunks"],
        )
        current_item = None
        chunk_idx = 0
        fps = self._config.fps
        orig_samples_per_frame = int(self._output_audio_sample_rate / fps)
        batch_size = self._config.batch_size  # Can be adjusted based on actual needs
        # max_speaking_buffer = batch_size * 5  # Maximum length of speaking frame buffer

        # warmup, ensure CUDA context and memory allocation
        if torch.cuda.is_available():
            t0 = time.time()
            # Regular batch_size warmup
            dummy_whisper = torch.zeros(
                batch_size, 50, 384, device=self._avatar.device, dtype=self._avatar.weight_dtype
            )
            await asyncio.to_thread(self._avatar.generate_frames, dummy_whisper, 0, batch_size)
            # Remainder batch_size warmup (only when there's a remainder)
            remain = fps % batch_size
            if remain > 0:
                dummy_whisper_remain = torch.zeros(
                    remain, 50, 384, device=self._avatar.device, dtype=self._avatar.weight_dtype
                )
                await asyncio.to_thread(
                    self._avatar.generate_frames, dummy_whisper_remain, 0, remain
                )
            torch.cuda.synchronize()
            t1 = time.time()
            logging.info(f"_frame_generator_loop self-warmup done, time: {(t1 - t0) * 1000:.1f} ms")
            self._frame_generator_warmup_event.set()

        while self._session_running is True:
            try:
                if current_item is None:
                    item = await asyncio.wait_for(self._whisper_queue.get(), timeout=0.1)
                    fetched_chunks = item["whisper_chunks"]
                    num_fetched_chunks = (
                        fetched_chunks.shape[0]
                        if isinstance(fetched_chunks, torch.Tensor) and fetched_chunks.ndim > 0
                        else 0
                    )
                    if num_fetched_chunks > 0:
                        current_item = CurrentSpeechItem(
                            whisper_chunks=fetched_chunks,
                            audio_data=item["audio_data"],
                            speech_id=item["speech_id"],
                            end_of_speech=item["end_of_speech"],
                            num_chunks=num_fetched_chunks,
                        )
                        chunk_idx = 0

                # Batch inference for speaking frames
                if current_item is not None and chunk_idx < current_item.num_chunks:
                    remain = current_item.num_chunks - chunk_idx
                    cur_batch = min(batch_size, remain)
                    batch_start_time = time.time()
                    # Get frame_id from frame_id queue
                    t1 = time.time()
                    frame_ids = []
                    for _ in range(cur_batch):
                        try:
                            item = self._frame_id_queue.get_nowait()
                            frame_ids.append(item)
                        except asyncio.QueueEmpty:
                            await asyncio.sleep(0.1)
                            continue
                    cost = time.time() - t1
                    print(f"{cost=}")
                    whisper_batch = current_item.whisper_chunks[chunk_idx : chunk_idx + cur_batch]
                    try:
                        recon_idx_list = await asyncio.to_thread(
                            self._avatar.generate_frames, whisper_batch, frame_ids[0], cur_batch
                        )
                    except Exception as e:
                        logging.error(
                            f"[GEN_FRAME_ERROR] frame_id={frame_ids[0]}, speech_id={current_item.speech_id}, error: {e}",
                            exc_info=True,
                        )
                        recon_idx_list = [
                            (np.zeros((256, 256, 3), dtype=np.uint8), frame_ids[0] + i)
                            for i in range(cur_batch)
                        ]
                    batch_end_time = time.time()
                    if self._config.debug:
                        logging.info(
                            f"[FRAME_GEN] Generated speaking frame: speech_id={current_item.speech_id}, chunk_idx={chunk_idx}, cur_batch={cur_batch}, batch_time={(batch_end_time - batch_start_time) * 1000:.1f}ms"
                        )
                    for i in range(cur_batch):
                        recon, idx = recon_idx_list[i]
                        audio = self._get_audio_for_frame(
                            current_item.audio_data,
                            chunk_idx + i,
                            current_item.num_chunks,
                            orig_samples_per_frame,
                        )
                        is_last = chunk_idx + i == current_item.num_chunks - 1
                        eos = current_item.end_of_speech and is_last
                        # Directly put into compose queue, no longer controlled by fps
                        compose_item = {
                            "recon": recon,
                            "idx": idx,
                            "speech_id": current_item.speech_id,
                            "avatar_status": AvatarStatus.SPEAKING,
                            "end_of_speech": eos,
                            "audio_segment": audio,
                            "frame_id": idx,
                            "timestamp": time.time(),
                        }
                        await self._compose_queue.put(compose_item)
                    if chunk_idx + cur_batch >= current_item.num_chunks:
                        current_item = None
                        chunk_idx = 0
                    else:
                        chunk_idx += cur_batch
                    continue
            except asyncio.CancelledError:
                logging.warning("frame_generator_loop task cancelled")
                break
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.exception(f"frame_generator_loop task error: {e}")
                continue
        logging.info("frame_generator loop stopped")

    async def _compose_loop(self):
        """
        Responsible for executing res2combined and putting the synthesis results into _output_queue
        """
        while self._session_running is True:
            try:
                item = await asyncio.wait_for(self._compose_queue.get(), timeout=0.1)
                recon = item["recon"]
                idx = item["idx"]
                frame = await asyncio.to_thread(self._avatar.res2combined, recon, idx)
                item["frame"] = frame
                await self._output_queue.put(item)
            except asyncio.CancelledError:
                logging.warning("compose_loop task cancelled")
                break
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.exception(f"compose_loop task error: {e}")
                continue
        logging.info("compose_loop stopped")

    async def _frame_collector_loop(self):
        """
        Collector strictly outputs at fps, with frame numbers matching the frame_id assigned by the inference thread.
        """
        fps = self._config.fps
        frame_interval = 1.0 / fps
        start_time = time.perf_counter()
        local_frame_id = 0
        last_active_speech_id = None
        last_speaking = False
        last_end_of_speech = False
        current_speech_id = None
        while self._session_running is True:
            try:
                # Control fps
                target_time = start_time + local_frame_id * frame_interval
                now = time.perf_counter()
                sleep_time = target_time - now
                if sleep_time > 0.002:
                    await asyncio.sleep(sleep_time - 0.001)
                # while time.perf_counter() < target_time:
                #    pass

                # Record the start time for profiling
                t_frame_start = time.perf_counter()
                # Allocate frame_id
                await self._frame_id_queue.put(local_frame_id)

                if self._output_queue.empty():
                    frame = self._avatar.generate_idle_frame(local_frame_id)
                    speech_id = last_active_speech_id
                    avatar_status = AvatarStatus.LISTENING
                    end_of_speech = False
                    frame_timestamp = time.time()
                    audio_segment = None
                else:
                    output_item = self._output_queue.get_nowait()
                    frame = output_item["frame"]
                    speech_id = output_item["speech_id"]
                    avatar_status = output_item["avatar_status"]
                    end_of_speech = output_item["end_of_speech"]
                    frame_timestamp = output_item.get("timestamp", None)
                    audio_segment = output_item["audio_segment"]

                # Notify video
                video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
                video_result = VideoResult(
                    video_frame=video_frame,
                    speech_id=speech_id,
                    avatar_status=avatar_status,
                    end_of_speech=end_of_speech,
                )
                await self._callback_image(video_result)

                # Logging logic
                is_idle = avatar_status == AvatarStatus.LISTENING and speech_id is None
                is_speaking = avatar_status == AvatarStatus.SPEAKING
                is_end_of_speech = bool(end_of_speech)
                if self._config.debug:
                    if is_speaking:
                        # First speaking frame
                        if speech_id != current_speech_id:
                            logging.info(
                                f"[SPEAKING_FRAME][START] frame_id={local_frame_id}, speech_id={speech_id}, status={avatar_status}, end_of_speech={end_of_speech}, video_timestamp={frame_timestamp}"
                            )
                            current_speech_id = speech_id
                        # Last speaking frame
                        if is_end_of_speech:
                            logging.info(
                                f"[SPEAKING_FRAME][END] frame_id={local_frame_id}, speech_id={speech_id}, status={avatar_status}, end_of_speech={end_of_speech}, video_timestamp={frame_timestamp}"
                            )
                            current_speech_id = None
                        # Middle speaking frame
                        if not is_end_of_speech and (speech_id == current_speech_id):
                            logging.info(
                                f"[SPEAKING_FRAME] frame_id={local_frame_id}, speech_id={speech_id}, status={avatar_status}, end_of_speech={end_of_speech}, video_timestamp={frame_timestamp}"
                            )
                    elif is_idle and last_speaking:
                        if last_end_of_speech:
                            logging.info(
                                f"[IDLE_FRAME] Start after speaking: frame_id={local_frame_id}, status={avatar_status}"
                            )
                        else:
                            logging.warning(
                                f"[IDLE_FRAME] Inserted idle during speaking: frame_id={local_frame_id}"
                            )
                else:
                    if is_speaking and speech_id != current_speech_id:
                        logging.info(
                            f"[SPEAKING_FRAME] Start: frame_id={local_frame_id}, speech_id={speech_id}"
                        )
                        current_speech_id = speech_id
                    # Last speaking frame
                    if is_speaking and is_end_of_speech:
                        logging.info(
                            f"[SPEAKING_FRAME] End: frame_id={local_frame_id}, speech_id={speech_id}, end_of_speech=True"
                        )
                        current_speech_id = None
                    # Idle frame: distinguish between inserted idle during speaking and first idle after speaking
                    if is_idle and last_speaking:
                        if last_end_of_speech:
                            logging.info(
                                f"[IDLE_FRAME] Start after speaking: frame_id={local_frame_id}"
                            )
                        else:
                            logging.warning(
                                f"[IDLE_FRAME] Inserted idle during speaking: frame_id={local_frame_id}"
                            )

                # Audio related
                audio_len = len(audio_segment) if audio_segment is not None else 0
                if audio_len > 0:
                    audio_np = np.asarray(audio_segment, dtype=np.float32)
                    if audio_np.ndim == 1:
                        audio_np = audio_np[np.newaxis, :]
                    audio_frame = av.AudioFrame.from_ndarray(audio_np, format="flt", layout="mono")
                    audio_frame.sample_rate = self._output_audio_sample_rate
                    audio_result = AudioResult(
                        audio_frame=audio_frame, speech_id=speech_id, end_of_speech=end_of_speech
                    )
                    if speech_id not in self._audio_cache:
                        self._audio_cache[speech_id] = []
                    self._audio_cache[speech_id].append(
                        audio_np[0] if audio_np.ndim == 2 else audio_np
                    )
                    audio_len_sum = (
                        sum([len(seg) for seg in self._audio_cache[speech_id]])
                        / self._output_audio_sample_rate
                    )
                    if self._config.debug:
                        logging.info(
                            f"[AUDIO_FRAME] frame_id={local_frame_id}, speech_id={speech_id}, end_of_speech={end_of_speech}, audio_timestamp={frame_timestamp}, Cumulative audio duration={audio_len_sum:.3f}s"
                        )
                    await self._callback_audio(audio_result)

                # Status switching etc.
                if end_of_speech:
                    logging.info(f"Status change: SPEAKING -> LISTENING, speech_id={speech_id}")
                    try:
                        if getattr(self._config, "debug_save_handler_audio", False):
                            all_audio = np.concatenate(self._audio_cache[speech_id], axis=-1)
                            save_dir = "logs/audio_segments"
                            os.makedirs(save_dir, exist_ok=True)
                            wav_path = os.path.join(save_dir, f"{speech_id}_all.wav")
                            sf.write(
                                wav_path,
                                all_audio,
                                self._output_audio_sample_rate,
                                subtype="PCM_16",
                            )
                            logging.info(f"[AUDIO_FRAME] saved full wav: {wav_path}")
                    except Exception as e:
                        logging.error(f"[AUDIO_FRAME] save full wav error: {e}")
                    del self._audio_cache[speech_id]
                    self._notify_status_change(speech_id, AvatarStatus.LISTENING)

                t_frame_end = time.perf_counter()
                if self._config.debug and (t_frame_end - t_frame_start > frame_interval):
                    logging.warning(
                        f"[PROFILE] frame_id={local_frame_id} total={t_frame_end - t_frame_start:.4f}s (>{frame_interval:.4f}s)"
                    )

                local_frame_id += 1
                last_speaking = is_speaking
                last_end_of_speech = is_end_of_speech
            except asyncio.CancelledError:
                logging.warning("frame_collector_loop task cancelled")
                break
            except Exception as e:
                logging.exception(f"frame_collector_loop task error: {e}")
                continue
        logging.info("frame_collector_loop loop ended")

    async def _callback_image(self, image_result: VideoResult):
        video_frame: av.VideoFrame = image_result.video_frame
        img = video_frame.to_image()
        if self._session_running:
            await self.queue_frame(
                OutputImageRawFrame(
                    image=img.tobytes(),
                    size=img.size,
                    format=img.format,
                    mode=img.mode,
                )
            )

    async def _callback_audio(self, audio_result: AudioResult):
        audio_frame: av.AudioFrame = audio_result.audio_frame
        if self._session_running:
            await self.queue_frame(
                OutputAudioRawFrame(
                    audio=audio_frame.to_ndarray().tobytes(),
                    sample_rate=audio_frame.sample_rate,
                )
            )

    def _notify_status_change(self, speech_id: str, status: AvatarStatus):
        pass
