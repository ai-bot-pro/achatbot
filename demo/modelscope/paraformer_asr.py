import os

from funasr import AutoModel
import librosa
import soundfile
import typer

app = typer.Typer()


@app.command()
def online():
    """
    KPI:...
    - https://www.modelscope.cn/models/iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online/summary
    - https://www.modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/summary
    """
    chunk_size = [0, 10, 5]  # [0, 10, 5] 600ms
    # chunk_size = [0, 8, 4]  # [0, 8, 4] 480ms
    encoder_chunk_look_back = 4  # number of chunks to lookback for encoder self-attention
    decoder_chunk_look_back = 1  # number of encoder chunks to lookback for decoder cross-attention

    model = AutoModel(
        # model="iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online",
        # model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
        model="paraformer-zh-streaming",
        model_revision="v2.0.4",
    )  # large
    with open("./paraformer_model.txt", "w") as f:
        print(model.model, file=f, flush=True)

    # wav_file = os.path.join(model.model_path, "example/asr_example.wav")
    wav_file = "./assets/Chinese_prompt.wav"
    print(wav_file)
    speech, sample_rate = soundfile.read(wav_file)
    print(sample_rate, speech.shape)
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        speech = librosa.resample(speech, orig_sr=sample_rate, target_sr=target_sample_rate)
        sample_rate = target_sample_rate
        print(f"Resampled to {target_sample_rate}Hz {speech.shape=}")

    chunk_stride = chunk_size[1] * 960  # 600ms

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
            encoder_chunk_look_back=encoder_chunk_look_back,
            decoder_chunk_look_back=decoder_chunk_look_back,
            disable_pbar=True,
        )
        print(res)


@app.command()
def offline():
    """
    - KPI: ...
    - https://www.modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
    - long https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch
    - speaker https://modelscope.cn/models/damo/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn
    - hotword https://www.modelscope.cn/models/iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404
    """
    # paraformer-zh is a multi-functional asr model
    # use vad, punc, spk or not as you need
    model = AutoModel(
        model="paraformer-zh",
        model_revision="v2.0.4",
        vad_model="fsmn-vad",
        vad_model_revision="v2.0.4",
        punc_model="ct-punc-c",  # https://modelscope.cn/models/iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727/summary
        punc_model_revision="v2.0.4",
        # spk_model="cam++", spk_model_revision="v2.0.2",
    )
    with open("./paraformer_model.txt", "w") as f:
        print(model.model, file=f, flush=True)
    input = "./assets/Chinese_prompt.wav"
    # input = f"{model.model_path}/example/asr_example.wav"
    res = model.generate(input=input, batch_size_s=300, hotword="魔搭", disable_pbar=True)
    print(res)


"""
python demo/modelscope/paraformer_asr.py offline
python demo/modelscope/paraformer_asr.py online
"""
if __name__ == "__main__":
    app()

"""
Cif:
  (predictor): CifPredictorV2(
    (pad): ConstantPad1d(padding=(1, 1), value=0)
    (cif_conv1d): Conv1d(320, 320, kernel_size=(3,), stride=(1,))
    (cif_output): Linear(in_features=320, out_features=1, bias=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
"""
