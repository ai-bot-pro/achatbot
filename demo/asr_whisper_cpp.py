import os
import ctypes
import pathlib


whisper_cpp_dir = os.getenv("WHISPER_CPP_DIR", "/Users/wuyong/project/pywhispercpp/whisper.cpp")
library_path = os.path.join(whisper_cpp_dir, "build/src/libwhisper.dylib")
whisper_model_path = "./models/ggml-base.bin"
vad_model_path = "./models/ggml-silero-v5.1.2.bin"
wav_file = os.path.join(whisper_cpp_dir, "samples/jfk.wav")


# this needs to match the C struct in whisper.h
class WhisperFullParams(ctypes.Structure):
    _fields_ = [
        ("strategy", ctypes.c_int),
        #
        ("n_max_text_ctx", ctypes.c_int),
        ("n_threads", ctypes.c_int),
        ("offset_ms", ctypes.c_int),
        ("duration_ms", ctypes.c_int),
        #
        ("translate", ctypes.c_bool),
        ("no_context", ctypes.c_bool),
        ("single_segment", ctypes.c_bool),
        ("print_special", ctypes.c_bool),
        ("print_progress", ctypes.c_bool),
        ("print_realtime", ctypes.c_bool),
        ("print_timestamps", ctypes.c_bool),
        #
        ("token_timestamps", ctypes.c_bool),
        ("thold_pt", ctypes.c_float),
        ("thold_ptsum", ctypes.c_float),
        ("max_len", ctypes.c_int),
        ("max_tokens", ctypes.c_int),
        #
        ("speed_up", ctypes.c_bool),
        ("audio_ctx", ctypes.c_int),
        #
        ("prompt_tokens", ctypes.c_void_p),
        ("prompt_n_tokens", ctypes.c_int),
        #
        ("language", ctypes.c_char_p),
        #
        ("suppress_blank", ctypes.c_bool),
        #
        ("temperature_inc", ctypes.c_float),
        ("entropy_thold", ctypes.c_float),
        ("logprob_thold", ctypes.c_float),
        ("no_speech_thold", ctypes.c_float),
        #
        ("greedy", ctypes.c_int * 1),
        ("beam_search", ctypes.c_int * 3),
        #
        ("new_segment_callback", ctypes.c_void_p),
        ("new_segment_callback_user_data", ctypes.c_void_p),
        #
        ("encoder_begin_callback", ctypes.c_void_p),
        ("encoder_begin_callback_user_data", ctypes.c_void_p),
    ]


def demo():
    # load library and model
    whisper = ctypes.CDLL(library_path)

    # tell Python what are the return types of the functions
    whisper.whisper_init_from_file.restype = ctypes.c_void_p
    whisper.whisper_full_default_params.restype = WhisperFullParams
    whisper.whisper_full_get_segment_text.restype = ctypes.c_char_p

    # initialize whisper.cpp context
    ctx = whisper.whisper_init_from_file(whisper_model_path.encode("utf-8"))

    # get default whisper parameters and adjust as needed
    params = whisper.whisper_full_default_params()
    print(params)
    params.print_realtime = True
    params.print_progress = False

    # load WAV file
    # this is needed to read the WAV file properly
    from scipy.io import wavfile

    samplerate, data = wavfile.read(wav_file)

    # convert to 32-bit float
    data = data.astype("float32") / 32768.0

    # run the inference
    result = whisper.whisper_full(
        ctypes.c_void_p(ctx),
        params,
        data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        len(data),
    )
    if result != 0:
        print("Error: {}".format(result))
        exit(1)

    # print results from Python
    # print("\nResults from Python:\n")
    n_segments = whisper.whisper_full_n_segments(ctypes.c_void_p(ctx))
    for i in range(n_segments):
        t0 = whisper.whisper_full_get_segment_t0(ctypes.c_void_p(ctx), i)
        t1 = whisper.whisper_full_get_segment_t1(ctypes.c_void_p(ctx), i)
        txt = whisper.whisper_full_get_segment_text(ctypes.c_void_p(ctx), i)

        print(f"{t0 / 1000.0:.3f} - {t1 / 1000.0:.3f} : {txt.decode('utf-8')}")

    # free the memory
    whisper.whisper_free(ctypes.c_void_p(ctx))


def transcribe():
    import soundfile as sf

    from whispercpy import WhisperCPP
    from whispercpy.utils import to_timestamp

    # reading audio
    data, sr = sf.read(wav_file, dtype="float32")

    # load model
    model = WhisperCPP(
        library_path, whisper_model_path, vad_model_path, use_gpu=False, verbose=True
    )

    print("--------- VAD Result ----------")

    # get vad results
    for segment in model.vad(data):
        print(
            f"[{to_timestamp(segment.t0, False)}" + " --> " + f"{to_timestamp(segment.t1, False)}]"
        )

    print("--------- ASR Result ----------")

    # get asr results
    for segment in model.transcribe(data, language="en", beam_size=5, token_timestamps=True):
        print(f"{segment=}")
        print(
            f"[{to_timestamp(segment.t0, False)}"
            + " --> "
            + f"{to_timestamp(segment.t1, False)}] "
            + segment.text
        )
        print("--------- Token Info ----------")
        print(
            "\n".join(
                [
                    f"[{to_timestamp(token.t0, False)}"
                    + " --> "
                    + f"{to_timestamp(token.t1, False)}] {token.text}"
                    for token in segment.tokens
                ]
            )
        )
        print("-------------------------------")


count = 0


def live():
    import sounddevice as sd

    from whispercpy import WhisperCPP, WhisperStream
    from whispercpy.utils import to_timestamp
    from whispercpy.constant import STREAMING_ENDING

    core = WhisperCPP(library_path, whisper_model_path, use_gpu=False)
    asr = WhisperStream(core, language="zh", return_token=True)

    samplerate = 16000
    block_duration = 0.25
    block_size = int(samplerate * block_duration)
    channels = 1

    def callback(indata, frames, time, status):
        global count

        chunk = indata.copy().tobytes()
        asr.pipe(chunk)
        transcript = asr.get_transcript()
        transcripts = asr.get_transcripts()

        if len(transcripts) > count:
            print("\r" + transcripts[-1].text)
            print("--")
            count += 1
        else:
            print(f"\r{transcript.text}", end="", flush=True)

    def print_result():
        transcripts = asr.get_transcripts()

        for transcript in transcripts:
            print(
                f"[{to_timestamp(transcript.t0, False)}"
                + " --> "
                + f"{to_timestamp(transcript.t1, False)}] "
                + transcript.text
            )
            print("-------------------------------")
            print(
                "\n".join(
                    [
                        f"[{to_timestamp(token.t0, False)}"
                        + " --> "
                        + f"{to_timestamp(token.t1, False)}] {token.text}"
                        for token in transcript.tokens
                    ]
                )
            )
            print("-------------------------------")

    # Recording
    try:
        with sd.InputStream(
            samplerate=samplerate,
            channels=channels,
            callback=callback,
            blocksize=block_size,
            dtype="float32",
        ):
            print("🎤 Recording for ASR... Press Ctrl+C to stop.")
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("⏹️ Recording stopped.")
        # send end signal, and await
        asr.pipe(STREAMING_ENDING).join()
        print_result()


if __name__ == "__main__":
    # demo()
    transcribe()
    # live()
