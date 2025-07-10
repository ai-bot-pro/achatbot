import os
import openai

"""
FASTDEPLOY_SERVE_URL https://weege126--fastdeploy-api-server-serve-dev.modal.run python src/llm/fastdeploy/cli.py
"""
URL = os.getenv("FASTDEPLOY_SERVE_URL", "https://weedge--fastdeploy-api-server-serve-dev.modal.run")

client = openai.Client(base_url=f"{URL}/v1", api_key="null")

response = client.chat.completions.create(
    model="null",
    messages=[
        {"role": "system", "content": "I'm a helpful AI assistant."},
        # {"role": "user", "content": "Rewrite Li Bai's 'Quiet Night Thought' as a modern poem"},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg"
                        #"url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"
                    },
                },
                #{"type": "text", "text": "What era does this artifact belong to?"},
                {"type": "text", "text": "请描述下图片内容"},
            ],
        },
    ],
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta:
        print(chunk.choices[0].delta.content, end="")
print("\n")
