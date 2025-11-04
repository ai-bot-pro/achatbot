# Multimodal Interaction
## text (generate/chat)

<img width="1014" height="895" alt="image" src="https://github.com/user-attachments/assets/683760c3-6350-4a0e-bb86-edc2c5db8da2" />


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

- tts text + speech 梅尔谱图 -> NAR DDPM+CFM -> 梅尔谱图 -> vocoder → 波形 (pipeline)
  <img width="1336" height="510" alt="image" src="https://github.com/user-attachments/assets/2314a29e-e705-4e62-8d22-43a3296d1143" />



### audio -> tex/audio
- asr+llm+tts (voice-chat)
  <img width="1164" height="842" alt="image" src="https://github.com/user-attachments/assets/0dd3d7c4-fc40-486f-a5b2-1eac458a215c" />





## vision


## more

- Embodied Intelligence: Robots that touch the world, perceive and move
