# 解决方案pipeline

仨种方案:使用本地模型能力,调用远程api,以及混合模式；分别适合研究人员(推理测，建模，对应数据处理，训练微调，推理优化),应用开发人员(产品业务定位能力),全栈开发人员（资源整合能力）:
- 本地模型：(more model's ckpt(.pt) use torch to load firstly, conver to other format)
  - llm: llama2,3,3.1; gemma,2; qwen,1.5,2; etc...
  - speech
    - detector model: waker(picovoice); VAD(silero); detector(pyannote); etc.. 
    - asr model: Whisper; SenseVoice
    - tts model: ChatTTS; CosyVocie
  - vision
    - cv model: YOLOv5, YOLOv8
- 远程api:
  - llm: qianfan, openai, or the same as openai api(together, groq ...), etc...
  - speech: 
    - asr: openai wishper, etc...
    - tts: gtts, edge tts, etc...
  - vision
    - cv: 
- 混合