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
    print(model.model)

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
    print(model.model)

    wav_file = f"{model.model_path}/example/vad_example.wav"
    res = model.generate(input=wav_file, disable_pbar=True)
    print(res)


"""
python demo/modelscope/fsmn_vad.py offline
python demo/modelscope/fsmn_vad.py online
"""
if __name__ == "__main__":
    app()

"""
FsmnVADStreaming(
  (encoder): FSMN(
    (in_linear1): AffineTransform(
      (linear): Linear(in_features=400, out_features=140, bias=True)
    )
    (in_linear2): AffineTransform(
      (linear): Linear(in_features=140, out_features=250, bias=True)
    )
    (relu): RectifiedLinear(
      (relu): ReLU()
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (fsmn): FsmnStack(
      (0): BasicBlock(
        (linear): LinearTransform(
          (linear): Linear(in_features=250, out_features=128, bias=False)
        )
        (fsmn_block): FSMNBlock(
          (conv_left): Conv2d(128, 128, kernel_size=(20, 1), stride=(1, 1), groups=128, bias=False)
        )
        (affine): AffineTransform(
          (linear): Linear(in_features=128, out_features=250, bias=True)
        )
        (relu): RectifiedLinear(
          (relu): ReLU()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (1): BasicBlock(
        (linear): LinearTransform(
          (linear): Linear(in_features=250, out_features=128, bias=False)
        )
        (fsmn_block): FSMNBlock(
          (conv_left): Conv2d(128, 128, kernel_size=(20, 1), stride=(1, 1), groups=128, bias=False)
        )
        (affine): AffineTransform(
          (linear): Linear(in_features=128, out_features=250, bias=True)
        )
        (relu): RectifiedLinear(
          (relu): ReLU()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (2): BasicBlock(
        (linear): LinearTransform(
          (linear): Linear(in_features=250, out_features=128, bias=False)
        )
        (fsmn_block): FSMNBlock(
          (conv_left): Conv2d(128, 128, kernel_size=(20, 1), stride=(1, 1), groups=128, bias=False)
        )
        (affine): AffineTransform(
          (linear): Linear(in_features=128, out_features=250, bias=True)
        )
        (relu): RectifiedLinear(
          (relu): ReLU()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (3): BasicBlock(
        (linear): LinearTransform(
          (linear): Linear(in_features=250, out_features=128, bias=False)
        )
        (fsmn_block): FSMNBlock(
          (conv_left): Conv2d(128, 128, kernel_size=(20, 1), stride=(1, 1), groups=128, bias=False)
        )
        (affine): AffineTransform(
          (linear): Linear(in_features=128, out_features=250, bias=True)
        )
        (relu): RectifiedLinear(
          (relu): ReLU()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (out_linear1): AffineTransform(
      (linear): Linear(in_features=250, out_features=140, bias=True)
    )
    (out_linear2): AffineTransform(
      (linear): Linear(in_features=140, out_features=248, bias=True)
    )
    (softmax): Softmax(dim=-1)
  )
)
"""