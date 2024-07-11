import time
from deps.SenseVoice.model import SenseVoiceSmall

m: SenseVoiceSmall = None
model_dir = "./models/FunAudioLLM/SenseVoiceSmall"
m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, hub="hf")
print(m, kwargs)


start = time.time()
res = m.inference(
    # data_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav",
    data_in="./records/asr_example_zh.wav",
    language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
    use_itn=False,
    **kwargs,
)

print(res, time.time() - start)
