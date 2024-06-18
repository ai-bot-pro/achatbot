r"""
python -m  demo.tts_chat
"""
import torch
import torchaudio

import deps.ChatTTS.ChatTTS as ChatTTS

if __name__ == "__main__":

    chat = ChatTTS.Chat()
    chat.load_models(compile=True, source="local",
                     local_path="./models/2Noise/ChatTTS")

    texts = ["你好，我是机器人",]
    ###################################
    r"""
    def infer_code(
        models,
        text, 
        spk_emb = None,
        top_P = 0.7, 
        top_K = 20, 
        temperature = 0.3, 
        repetition_penalty = 1.05,
        max_new_token = 2048,
        **kwargs
    ):
    """
    # Sample a speaker from Gaussian.
    rand_spk = chat.sample_random_speaker()
    params_infer_code = {
        'spk_emb': rand_spk,  # add sampled speaker
        'temperature': .3,  # using custom temperature
        'top_P': 0.7,  # top P decode
        'top_K': 20,  # top K decode
    }

    ###################################
    # For sentence level manual control.

    # use oral_(0-9), laugh_(0-2), break_(0-7)
    # to generate special token in text to synthesize.
    params_refine_text = {
        'prompt': '[oral_2][laugh_0][break_6]'
    }
    wav = chat.infer(
        texts,
        params_refine_text=params_refine_text,
        params_infer_code=params_infer_code,
        do_text_normalization=False,
    )
    torchaudio.save("./records/chat_tts_output1.wav",
                    torch.from_numpy(wav[0]), 24000)
    ###################################
    # For word level manual control.
    text = 'What is [uv_break]your favorite english food?[laugh][lbreak]'
    wav = chat.infer(
        text, skip_refine_text=True,
        params_refine_text=params_refine_text,
        params_infer_code=params_infer_code,
        do_text_normalization=False,
    )
    print("wav[0] type", type(wav[0]))
    torchaudio.save("./records/chat_tts_output2.wav",
                    torch.from_numpy(wav[0]), 24000)
