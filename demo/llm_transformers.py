from threading import Thread
from transformers import pipeline, TextStreamer, TextIteratorStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name_or_path = "./models/Qwen/Qwen2-0.5B-Instruct"


def manual():
    # Now you do not need to add "trust_remote_code=True"
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    print("-------round 1------")
    # Instead of using model.chat(), we directly use model.generate()
    # But you need to use tokenizer.apply_chat_template() to format your inputs as shown below
    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Directly use generate() and tokenizer.decode() to get the output.
    # Use `max_new_tokens` to control the maximum output length.
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
    )
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)

    print("-------round 2------")
    messages.append({"role": "assistant", "content": response})

    prompt = "Tell me more."
    messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Directly use generate() and tokenizer.decode() to get the output.
    # Use `max_new_tokens` to control the maximum output length.
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
    )
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)


def pipe_line():
    pipe = pipeline("text-generation", model_name_or_path, torch_dtype="auto", device_map="auto")

    print("-------round 1------")
    # the default system message will be used
    messages = [
        {"role": "user", "content": "Give me a short introduction to large language model."}
    ]

    response_message = pipe(messages, max_new_tokens=512)[0]["generated_text"][-1]
    print(response_message)

    print("-------round 2------")
    messages.append(response_message)

    prompt = "Tell me more."
    messages.append({"role": "user", "content": prompt})

    response_message = pipe(messages, max_new_tokens=512)[0]["generated_text"][-1]
    print(response_message)


def pipeline_batch():
    pipe = pipeline("text-generation", model_name_or_path, torch_dtype="auto", device_map="auto")

    pipe.tokenizer.padding_side = "left"

    message_batch = [
        [{"role": "user", "content": "Give me a detailed introduction to large language model."}],
        [{"role": "user", "content": "Hello!"}],
    ]

    result_batch = pipe(message_batch, max_new_tokens=512, batch_size=2)
    response_message_batch = [result[0]["generated_text"][-1] for result in result_batch]
    print(response_message_batch)


def pipeline_text_streamer():
    """
    print the stream of text from the pipeline to stdout
    """
    pipe = pipeline("text-generation", model_name_or_path, torch_dtype="auto", device_map="auto")

    streamer = TextStreamer(pipe.tokenizer, skip_prompt=True, skip_special_tokens=True)

    messages = [
        {"role": "user", "content": "Give me a short introduction to large language model."}
    ]

    response_message = pipe(messages, max_new_tokens=512, streamer=streamer)[0]["generated_text"][
        -1
    ]
    print(response_message)


def pipeline_text_iter_streamer():
    pipe = pipeline("text-generation", model_name_or_path, torch_dtype="auto", device_map="auto")

    messages = [
        {"role": "user", "content": "Give me a short introduction to large language model."}
    ]

    streamer = TextIteratorStreamer(pipe.tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Use Thread to run generation in background
    # Otherwise, the process is blocked until generation is complete
    # and no streaming effect can be observed.
    generation_kwargs = dict(text_inputs=messages, max_new_tokens=512, streamer=streamer)
    thread = Thread(target=pipe, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    for new_text in streamer:
        print(new_text, end="", flush=True)
        generated_text += new_text

    print("\n\ngenerated_text:\n", generated_text)


if __name__ == "__main__":
    # manual()
    # pipe_line()
    # pipeline_text_streamer()
    pipeline_text_iter_streamer()
