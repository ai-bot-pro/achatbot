# comfyui
depoly comfyui to modal

## Download Models(ckpt)
- Download models from huggingface
```shell
# download flux1.0 schnell fp8 cpkt weights
modal run src/download_models.py --repo-ids "Comfy-Org/flux1-schnell" --allow-patterns "flux1-schnell-fp8.safetensors"
```

## DIY Workflow
- DIY workflow and export workflow json (need Nvidia GPU / AMD / MAC ARM Silicon) 
```shell
# default use L4 GPU on modal
modal serve src/comfyui/ui.py

# use L40s GPU on modal
IMAGE_GPU=L40S modal serve src/comfyui/ui.py 
```

## Deploy API Server
- deploy comfyui api server (need Nvidia GPU / AMD / MAC ARM Silicon)
```shell
# default use L4 GPU on modal
modal serve src/comfyui/server.py

# use L40s GPU on modal
IMAGE_GPU=L40S modal serve src/comfyui/server.py 
```

## Client
- generate image
```shell
python deploy/modal/src/comfyui/client.py --modal-workspace "weedge" --prompt "Spider-Man visits Yosemite, rendered by Blender, trending on artstation" --dev

python deploy/modal/src/comfyui/client.py --modal-workspace $(modal profile current) --prompt "Surreal dreamscape with floating islands, upside-down waterfalls, and impossible geometric structures, all bathed in a soft, ethereal light" --dev

python client.py --modal-workspace $(modal profile current) --prompt "Surreal dreamscape with floating islands, upside-down waterfalls, and impossible geometric structures, all bathed in a soft, ethereal light" --size 512x512 --dev
```