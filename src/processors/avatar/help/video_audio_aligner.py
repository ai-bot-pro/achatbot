import logging


class VideoAudioAligner:
    """
    视频音频对齐器，用于将音频数据与视频帧进行同步对齐处理。

    主要功能：
    - 根据视频帧率(FPS)和音频采样率计算每帧对应的音频长度
    - 处理不同语音片段(speech_id)的音频数据
    - 在语音结束时进行音频长度对齐（补零或截断）

    属性：
        _fps: 视频帧率
        _current_speech_id: 当前处理的语音ID
        _audio_data_current_speech: 当前语音的音频数据缓存
        _total_frame_count_current_speech: 当前语音对应的总帧数

    方法：
        get_speech_level_algined_audio(): 获取与视频帧对齐的音频数据
            参数:
                audio_data: 原始音频数据(bytearray)
                origin_sample_rate: 音频采样率
                frame_count: 当前处理的视频帧数
                speech_id: 语音片段ID
                end_of_speech: 是否语音结束标志
            返回:
                对齐后的音频数据(bytearray)
    """

    def __init__(self, fps):
        self._fps = fps

        self._current_speech_id = ""
        self._audio_byte_length_current_speech = 0
        self._audio_data_current_speech = bytearray()
        self._total_frame_count_current_speech = 0
        self._audio_start_idx = 0
        self._returned_audio_length_current_speech = 0

    def get_aligned_audio(self):
        raise NotImplementedError()

    def get_speech_level_algined_audio(
        self, audio_data, origin_sample_rate, frame_count, speech_id, end_of_speech
    ):
        if speech_id != self._current_speech_id:
            self._audio_byte_length_current_speech = 0
            self._audio_data_current_speech = bytearray()
            self._current_speech_id = speech_id
            self._total_frame_count_current_speech = 0
            self._audio_start_idx = 0
            self._returned_audio_length_current_speech = 0
        self._audio_data_current_speech += audio_data
        self._total_frame_count_current_speech += frame_count

        audio_length_per_frame = origin_sample_rate / self._fps * 2
        assert audio_length_per_frame.is_integer()

        total_audio_length = int(self._total_frame_count_current_speech * audio_length_per_frame)

        if not end_of_speech:
            ret_audio = audio_data
        else:
            diff = total_audio_length - len(self._audio_data_current_speech)
            if diff > 0:
                logging.info(f"align video: add extra audio of length {diff}")
                self._audio_data_current_speech += bytearray(diff)
            else:
                logging.info(f"align video: remove tail audio of length {diff}")
                self._audio_data_current_speech = self._audio_data_current_speech[
                    :total_audio_length
                ]
            ret_audio = self._audio_data_current_speech[self._audio_start_idx :]
        self._returned_audio_length_current_speech += len(ret_audio)
        logging.info(
            f"audio of speech {speech_id}, end of speech {end_of_speech}, "
            f"total returned audio length {self._returned_audio_length_current_speech}, "
            f"actual audio length {total_audio_length}, start index {self._audio_start_idx}",
        )
        self._audio_start_idx += len(ret_audio)
        return ret_audio
