from funasr import AutoModel
import soundfile
import typer

app = typer.Typer()


@app.command()
def online():
    """
    https://modelscope.cn/iic/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary
    """
    chunk_size = 200  # ms
    model = AutoModel(model="fsmn-vad", model_revision="v2.0.4")

    wav_file = f"{model.model_path}/example/vad_example.wav"
    speech, sample_rate = soundfile.read(wav_file)
    chunk_stride = int(chunk_size * sample_rate / 1000)

    cache = {}
    total_chunk_num = int(len((speech) - 1) / chunk_stride + 1)
    for i in range(total_chunk_num):
        speech_chunk = speech[i * chunk_stride : (i + 1) * chunk_stride]
        is_final = i == total_chunk_num - 1
        res = model.generate(
            input=speech_chunk,
            cache=cache,
            is_final=is_final,
            chunk_size=chunk_size,
            disable_pbar=True,
        )
        if len(res[0]["value"]):
            print(res)


@app.command()
def offline():
    """
    https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary
    """
    model = AutoModel(model="fsmn-vad", model_revision="v2.0.4")
    print(model.model_path)

    wav_file = f"{model.model_path}/example/vad_example.wav"
    res = model.generate(input=wav_file, disable_pbar=True)
    print(res)


"""
python demo/funasr/fsmn_vad.py offline
python demo/funasr/fsmn_vad.py online
"""
if __name__ == "__main__":
    app()
