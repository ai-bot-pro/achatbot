## melo-tts
# - https://weedge.github.io/post/multimoding/voices/open_voice_extra_se_and_convert/
# - https://github.com/myshell-ai/MeloTTS

from melo.api import TTS

# Speed is adjustable
speed = 1.0
device = "cpu"  # or cuda:0

text = "我最近在学习machine learning，希望能够在未来的artificial intelligence领域有所建树。"
model = TTS(language="ZH", device=device)
speaker_ids = model.hps.data.spk2id

output_path = "zh.wav"
model.tts_to_file(text, speaker_ids["ZH"], output_path, speed=speed)
