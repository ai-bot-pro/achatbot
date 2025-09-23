import time
from clearvoice import ClearVoice
import typer

app = typer.Typer()


@app.command()
def offline():
    myClearVoice = ClearVoice(task="speech_enhancement", model_names=["MossFormerGAN_SE_16K"])

    # process single wave file
    start = time.time()
    output_wav = myClearVoice(input_path="records/speech_with_noise16k.wav", online_write=False)
    print(f"warmup cost: {time.time() - start} s")
    start = time.time()
    output_wav = myClearVoice(input_path="records/speech_with_noise16k.wav", online_write=False)
    print(f"cost: {time.time() - start} s")
    myClearVoice.write(output_wav, output_path="records/output_MossFormerGAN_SE_16K.wav")


"""
python demo/clearvoice/MossFormerGAN_SE_16K.py
"""
if __name__ == "__main__":
    import torch
    import os

    thread_count = os.cpu_count()
    print(f"thread_count: {thread_count}")

    torch.set_num_threads(min(8, thread_count))
    torch.set_num_interop_threads(min(8, thread_count))

    app()
