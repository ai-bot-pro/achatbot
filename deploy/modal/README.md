# intro
the Starter plan with $30.00 included compute credits per month, for development, so nice~

# run

```shell
pip install modal 

modal setup

# create worksapce achatbot environment
modal environment create achatbot

# serve
IMAGE_NAME=test modal serve -e achatbot src/fastapi_serve.py
IMAGE_NAME=prod modal serve -e achatbot src/fastapi_serve.py

# run bot e.g.:
# webrtc_audio_bot on test pip image
IMAGE_NAME=test modal serve -e achatbot src/fastapi_webrtc_audio_bot_serve.py
# webrtc_audio_bot on prod pip image
IMAGE_NAME=test modal serve -e achatbot src/fastapi_webrtc_audio_bot_serve.py

```

# references (nice docs) 👍
- https://modal.com/docs/guide
- https://modal.com/docs/guide/gpu
- https://modal.com/docs/guide/cuda
- https://modal.com/docs/examples/basic_web
- https://github.com/modal-labs/modal-examples