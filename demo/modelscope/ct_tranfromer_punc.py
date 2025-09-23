from modelscope import AutoModel
import typer

app = typer.Typer()


@app.command()
def online():
    """
    https://modelscope.cn/models/iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727/summary
    """
    model = AutoModel(model="iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727")
    print(model.model)

    inputs = "跨境河流是养育沿岸|人民的生命之源长期以来为帮助下游地区防灾减灾中方技术人员|在上游地区极为恶劣的自然条件下克服巨大困难甚至冒着生命危险|向印方提供汛期水文资料处理紧急事件中方重视印方在跨境河流问题上的关切|愿意进一步完善双方联合工作机制|凡是|中方能做的我们|都会去做而且会做得更好我请印度朋友们放心中国在上游的|任何开发利用都会经过科学|规划和论证兼顾上下游的利益"
    vads = inputs.split("|")
    rec_result_all = "outputs: "
    cache = {}
    for vad in vads:
        rec_result = model.generate(input=vad, cache=cache, disable_pbar=True)
        print(rec_result)
        rec_result_all += rec_result[0]["text"]

    print(rec_result_all)


@app.command()
def offline():
    """
    - https://modelscope.cn/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary
    - https://modelscope.cn/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large/summary
    """
    model = AutoModel(
        model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        model_revision=None,
    )
    print(model.model)

    res = model.generate(input="那今天的会就到这里吧 happy new year 明年见", disable_pbar=True)
    print(res)


"""
python demo/modelscope/ct_tranfromer_punc.py offline
python demo/modelscope/ct_tranfromer_punc.py online
"""
if __name__ == "__main__":
    app()


"""
CTTransformerStreaming(
  (embed): Embedding(272727, 256)
  (decoder): Linear(in_features=256, out_features=6, bias=True)
  (encoder): SANMVadEncoder(
    (embed): SinusoidalPositionEncoder()
    (encoders0): MultiSequential(
      (0): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANMwithMask(
          (linear_out): Linear(in_features=256, out_features=256, bias=True)
          (linear_q_k_v): Linear(in_features=256, out_features=768, bias=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (fsmn_block): Conv1d(256, 256, kernel_size=(11,), stride=(1,), groups=256, bias=False)
          (pad_fn): ConstantPad1d(padding=(10, 0), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=256, out_features=1024, bias=True)
          (w_2): Linear(in_features=1024, out_features=256, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
    (encoders): MultiSequential(
      (0): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANMwithMask(
          (linear_out): Linear(in_features=256, out_features=256, bias=True)
          (linear_q_k_v): Linear(in_features=256, out_features=768, bias=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (fsmn_block): Conv1d(256, 256, kernel_size=(11,), stride=(1,), groups=256, bias=False)
          (pad_fn): ConstantPad1d(padding=(10, 0), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=256, out_features=1024, bias=True)
          (w_2): Linear(in_features=1024, out_features=256, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (1): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANMwithMask(
          (linear_out): Linear(in_features=256, out_features=256, bias=True)
          (linear_q_k_v): Linear(in_features=256, out_features=768, bias=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (fsmn_block): Conv1d(256, 256, kernel_size=(11,), stride=(1,), groups=256, bias=False)
          (pad_fn): ConstantPad1d(padding=(10, 0), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=256, out_features=1024, bias=True)
          (w_2): Linear(in_features=1024, out_features=256, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
    (after_norm): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
)
"""
