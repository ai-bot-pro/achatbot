# Multimodal Interaction
## text (generate/chat)
<img width="1040" height="881" alt="image" src="https://github.com/user-attachments/assets/cbe81f83-8bbb-4bf2-9faa-354d14ec757e" />


## audio (voice)

### audio -> txt
- stream-speech-understanding (realtime ASR )
  <img width="1068" height="684" alt="image" src="https://github.com/user-attachments/assets/691675b4-a12f-455c-9cfe-cc9d2ef2cd9b" />

### text -> audio
- tts 文本/音素 → 梅尔谱图 → 波形 (pipeline/e2e)


<img width="1389" height="509" alt="image" src="https://github.com/user-attachments/assets/cc046327-7243-4cb6-ad1b-ef5a9a764812" />

- tts text tokenizer->text tokens + audio codec encoder ->vq code -> AR LLM -> vq code → audio codec decoder → 波形
 (pipeline)
  <img width="1101" height="417" alt="image" src="https://github.com/user-attachments/assets/c5db15fa-45b2-4074-bd44-bdaf4134735a" />

- tts text/speech tokenizer -> text tokens + speech tokens -> AR LLM -> speech tokens -> flow → 梅尔谱图 -> vocoder → 波形
 (pipeline)
  <img width="1088" height="479" alt="image" src="https://github.com/user-attachments/assets/f8635a57-6820-49e5-aaaa-b8f192b870e3" />




- audio-llm (multimode-chat)
  ![pipe](https://github.com/user-attachments/assets/9970cf18-9bbc-4109-a3c5-e3e3c88086af)
  ![queue](https://github.com/user-attachments/assets/30f2e880-f16d-4b62-8668-61bb97c57b2b)


- stream-tts (realtime-(clone)-speaker)
  ![text-audio](https://github.com/user-attachments/assets/676230a0-0a99-475b-9ef5-6afc95f044d8)
  ![audio-text text-audio](https://github.com/user-attachments/assets/cbcabf98-731e-4887-9f37-649ec81e37a0)


## vision (CV)

- stream-ocr (realtime-object-detection)

## more

- Embodied Intelligence: Robots that touch the world, perceive and move
