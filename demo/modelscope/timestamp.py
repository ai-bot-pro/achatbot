import typer

from src.common.time_utils import to_timestamp

app = typer.Typer()


@app.command()
def offline():
    """
    - https://modelscope.cn/models/iic/speech_timestamp_prediction-v1-16k-offline
    """
    from modelscope import AutoModel

    model = AutoModel(model="fa-zh", model_revision="v2.0.4")
    with open("./fa_zh_model.txt", "w") as f:
        print(model.model, file=f, flush=True)

    wav_file = f"{model.model_path}/example/asr_example.wav"
    text_file = f"{model.model_path}/example/text.txt"
    wav_file = f"/Users/wuyong/project/python/chat-bot/assets/Chinese_prompt.wav"
    text_file = "全 民 制 作 人 们 大 家 好 我 是 练 习 时 长 两 年 半 的 个 人 练 习 生 蔡 徐 坤 喜 欢 唱 跳 rap 篮 球 music"
    print(wav_file, text_file)
    results = model.generate(input=(wav_file, text_file), data_type=("sound", "text"))
    assert len(results) > 0
    res = results[0]
    print(res)
    for i, word in enumerate(res["text"].split()):
        item = {
            "word": word,
            "timestamp": res["timestamp"][i],
            "start": to_timestamp(res["timestamp"][i][0], msec=1),
            "end": to_timestamp(res["timestamp"][i][1], msec=1),
        }
        print(item)


"""
https://modelscope.cn/models/iic/speech_timestamp_prediction-v1-16k-offline/summary

python -m demo.modelscope.timestamp
"""
if __name__ == "__main__":
    app()

"""
MonotonicAligner(
  (specaug): SpecAugLFR(
    (freq_mask): MaskAlongAxisLFR(mask_width_range=[0, 30], num_mask=1, axis=freq)
    (time_mask): MaskAlongAxisLFR(mask_width_range=[0, 12], num_mask=1, axis=time)
  )
  (encoder): SANMEncoder(
    (embed): SinusoidalPositionEncoder()
    (encoders0): MultiSequential(
      (0): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=320, out_features=320, bias=True)
          (linear_q_k_v): Linear(in_features=560, out_features=960, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(320, 320, kernel_size=(11,), stride=(1,), groups=320, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=320, out_features=1280, bias=True)
          (w_2): Linear(in_features=1280, out_features=320, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((560,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
    (encoders): MultiSequential(
      (0): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=320, out_features=320, bias=True)
          (linear_q_k_v): Linear(in_features=320, out_features=960, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(320, 320, kernel_size=(11,), stride=(1,), groups=320, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=320, out_features=1280, bias=True)
          (w_2): Linear(in_features=1280, out_features=320, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (1): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=320, out_features=320, bias=True)
          (linear_q_k_v): Linear(in_features=320, out_features=960, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(320, 320, kernel_size=(11,), stride=(1,), groups=320, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=320, out_features=1280, bias=True)
          (w_2): Linear(in_features=1280, out_features=320, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (2): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=320, out_features=320, bias=True)
          (linear_q_k_v): Linear(in_features=320, out_features=960, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(320, 320, kernel_size=(11,), stride=(1,), groups=320, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=320, out_features=1280, bias=True)
          (w_2): Linear(in_features=1280, out_features=320, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (3): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=320, out_features=320, bias=True)
          (linear_q_k_v): Linear(in_features=320, out_features=960, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(320, 320, kernel_size=(11,), stride=(1,), groups=320, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=320, out_features=1280, bias=True)
          (w_2): Linear(in_features=1280, out_features=320, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (4): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=320, out_features=320, bias=True)
          (linear_q_k_v): Linear(in_features=320, out_features=960, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(320, 320, kernel_size=(11,), stride=(1,), groups=320, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=320, out_features=1280, bias=True)
          (w_2): Linear(in_features=1280, out_features=320, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (5): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=320, out_features=320, bias=True)
          (linear_q_k_v): Linear(in_features=320, out_features=960, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(320, 320, kernel_size=(11,), stride=(1,), groups=320, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=320, out_features=1280, bias=True)
          (w_2): Linear(in_features=1280, out_features=320, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (6): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=320, out_features=320, bias=True)
          (linear_q_k_v): Linear(in_features=320, out_features=960, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(320, 320, kernel_size=(11,), stride=(1,), groups=320, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=320, out_features=1280, bias=True)
          (w_2): Linear(in_features=1280, out_features=320, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (7): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=320, out_features=320, bias=True)
          (linear_q_k_v): Linear(in_features=320, out_features=960, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(320, 320, kernel_size=(11,), stride=(1,), groups=320, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=320, out_features=1280, bias=True)
          (w_2): Linear(in_features=1280, out_features=320, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (8): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=320, out_features=320, bias=True)
          (linear_q_k_v): Linear(in_features=320, out_features=960, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(320, 320, kernel_size=(11,), stride=(1,), groups=320, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=320, out_features=1280, bias=True)
          (w_2): Linear(in_features=1280, out_features=320, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (9): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=320, out_features=320, bias=True)
          (linear_q_k_v): Linear(in_features=320, out_features=960, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(320, 320, kernel_size=(11,), stride=(1,), groups=320, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=320, out_features=1280, bias=True)
          (w_2): Linear(in_features=1280, out_features=320, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (10): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=320, out_features=320, bias=True)
          (linear_q_k_v): Linear(in_features=320, out_features=960, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(320, 320, kernel_size=(11,), stride=(1,), groups=320, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=320, out_features=1280, bias=True)
          (w_2): Linear(in_features=1280, out_features=320, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (11): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=320, out_features=320, bias=True)
          (linear_q_k_v): Linear(in_features=320, out_features=960, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(320, 320, kernel_size=(11,), stride=(1,), groups=320, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=320, out_features=1280, bias=True)
          (w_2): Linear(in_features=1280, out_features=320, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (12): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=320, out_features=320, bias=True)
          (linear_q_k_v): Linear(in_features=320, out_features=960, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(320, 320, kernel_size=(11,), stride=(1,), groups=320, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=320, out_features=1280, bias=True)
          (w_2): Linear(in_features=1280, out_features=320, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (13): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=320, out_features=320, bias=True)
          (linear_q_k_v): Linear(in_features=320, out_features=960, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(320, 320, kernel_size=(11,), stride=(1,), groups=320, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=320, out_features=1280, bias=True)
          (w_2): Linear(in_features=1280, out_features=320, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (14): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=320, out_features=320, bias=True)
          (linear_q_k_v): Linear(in_features=320, out_features=960, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(320, 320, kernel_size=(11,), stride=(1,), groups=320, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=320, out_features=1280, bias=True)
          (w_2): Linear(in_features=1280, out_features=320, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (15): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=320, out_features=320, bias=True)
          (linear_q_k_v): Linear(in_features=320, out_features=960, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(320, 320, kernel_size=(11,), stride=(1,), groups=320, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=320, out_features=1280, bias=True)
          (w_2): Linear(in_features=1280, out_features=320, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (16): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=320, out_features=320, bias=True)
          (linear_q_k_v): Linear(in_features=320, out_features=960, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(320, 320, kernel_size=(11,), stride=(1,), groups=320, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=320, out_features=1280, bias=True)
          (w_2): Linear(in_features=1280, out_features=320, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (17): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=320, out_features=320, bias=True)
          (linear_q_k_v): Linear(in_features=320, out_features=960, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(320, 320, kernel_size=(11,), stride=(1,), groups=320, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=320, out_features=1280, bias=True)
          (w_2): Linear(in_features=1280, out_features=320, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (18): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=320, out_features=320, bias=True)
          (linear_q_k_v): Linear(in_features=320, out_features=960, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(320, 320, kernel_size=(11,), stride=(1,), groups=320, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=320, out_features=1280, bias=True)
          (w_2): Linear(in_features=1280, out_features=320, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (19): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=320, out_features=320, bias=True)
          (linear_q_k_v): Linear(in_features=320, out_features=960, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(320, 320, kernel_size=(11,), stride=(1,), groups=320, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=320, out_features=1280, bias=True)
          (w_2): Linear(in_features=1280, out_features=320, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (20): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=320, out_features=320, bias=True)
          (linear_q_k_v): Linear(in_features=320, out_features=960, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(320, 320, kernel_size=(11,), stride=(1,), groups=320, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=320, out_features=1280, bias=True)
          (w_2): Linear(in_features=1280, out_features=320, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (21): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=320, out_features=320, bias=True)
          (linear_q_k_v): Linear(in_features=320, out_features=960, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(320, 320, kernel_size=(11,), stride=(1,), groups=320, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=320, out_features=1280, bias=True)
          (w_2): Linear(in_features=1280, out_features=320, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (22): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=320, out_features=320, bias=True)
          (linear_q_k_v): Linear(in_features=320, out_features=960, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(320, 320, kernel_size=(11,), stride=(1,), groups=320, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=320, out_features=1280, bias=True)
          (w_2): Linear(in_features=1280, out_features=320, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (23): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=320, out_features=320, bias=True)
          (linear_q_k_v): Linear(in_features=320, out_features=960, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(320, 320, kernel_size=(11,), stride=(1,), groups=320, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=320, out_features=1280, bias=True)
          (w_2): Linear(in_features=1280, out_features=320, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (24): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=320, out_features=320, bias=True)
          (linear_q_k_v): Linear(in_features=320, out_features=960, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(320, 320, kernel_size=(11,), stride=(1,), groups=320, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=320, out_features=1280, bias=True)
          (w_2): Linear(in_features=1280, out_features=320, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (25): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=320, out_features=320, bias=True)
          (linear_q_k_v): Linear(in_features=320, out_features=960, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(320, 320, kernel_size=(11,), stride=(1,), groups=320, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=320, out_features=1280, bias=True)
          (w_2): Linear(in_features=1280, out_features=320, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (26): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=320, out_features=320, bias=True)
          (linear_q_k_v): Linear(in_features=320, out_features=960, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(320, 320, kernel_size=(11,), stride=(1,), groups=320, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=320, out_features=1280, bias=True)
          (w_2): Linear(in_features=1280, out_features=320, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (27): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=320, out_features=320, bias=True)
          (linear_q_k_v): Linear(in_features=320, out_features=960, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(320, 320, kernel_size=(11,), stride=(1,), groups=320, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=320, out_features=1280, bias=True)
          (w_2): Linear(in_features=1280, out_features=320, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (28): EncoderLayerSANM(
        (self_attn): MultiHeadedAttentionSANM(
          (linear_out): Linear(in_features=320, out_features=320, bias=True)
          (linear_q_k_v): Linear(in_features=320, out_features=960, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (fsmn_block): Conv1d(320, 320, kernel_size=(11,), stride=(1,), groups=320, bias=False)
          (pad_fn): ConstantPad1d(padding=(5, 5), value=0.0)
        )
        (feed_forward): PositionwiseFeedForward(
          (w_1): Linear(in_features=320, out_features=1280, bias=True)
          (w_2): Linear(in_features=1280, out_features=320, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): ReLU()
        )
        (norm1): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (norm2): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
    (after_norm): LayerNorm((320,), eps=1e-12, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (predictor): CifPredictorV3(
    (pad): ConstantPad1d(padding=(1, 1), value=0)
    (cif_conv1d): Conv1d(320, 320, kernel_size=(3,), stride=(1,))
    (cif_output): Linear(in_features=320, out_features=1, bias=True)
    (dropout): Dropout(p=0.1, inplace=False)
    (upsample_cnn): ConvTranspose1d(320, 320, kernel_size=(3,), stride=(3,))
    (blstm): LSTM(320, 320, batch_first=True, bidirectional=True)
    (cif_output2): Linear(in_features=640, out_features=1, bias=True)
  )
  (criterion_pre): mae_loss(
    (criterion): L1Loss()
  )
)
"""
