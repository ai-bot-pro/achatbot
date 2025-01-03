from funasr import AutoModel
import soundfile

chunk_size = 64  # ms
model = AutoModel(model="fsmn-vad", model_revision="v2.0.4")


wav_file = f"{model.model_path}/example/vad_example.wav"
speech, sample_rate = soundfile.read(wav_file)
chunk_stride = int(chunk_size * sample_rate / 1000)  # 64 ms (16000) -> 1024
print(chunk_stride, sample_rate)

cache = {}
total_chunk_num = int((len(speech) - 1) / chunk_stride + 1)
print(type(speech), len(speech), len(speech.tobytes()), total_chunk_num)
# for i in range(0, len(speech), chunk_stride):
#     speech_chunk = speech[i : i + chunk_stride]
for i in range(total_chunk_num):
    speech_chunk = speech[i * chunk_stride : (i + 1) * chunk_stride]
    is_final = i == total_chunk_num - 1
    # print(cache)
    res = model.generate(
        input=speech_chunk, cache=cache, is_final=is_final, chunk_size=chunk_size, disable_pbar=True
    )
    if len(res[0]["value"]):
        print(res)

    if len(res[0]["value"]):
        if res[0]["value"][0][0] != -1:
            print("voice", res)
