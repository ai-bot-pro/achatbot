from clearvoice import ClearVoice
import typer

app = typer.Typer()


@app.command()
def offline():
    myClearVoice = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])

    # process single wave file
    output_wav = myClearVoice(input_path="records/speech_with_noise_48k.wav", online_write=False)
    myClearVoice.write(output_wav, output_path="records/output_MossFormer2_SE_48K.wav")


"""
python demo/clearvoice/MossFormer2_SE_48K.py
"""
if __name__ == "__main__":
    import torch
    import os

    thread_count = os.cpu_count()
    print(f"thread_count: {thread_count}")

    torch.set_num_threads(min(8, thread_count))
    torch.set_num_interop_threads(min(8, thread_count))

    app()
