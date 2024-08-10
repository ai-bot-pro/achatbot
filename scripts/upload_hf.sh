#/bin/bash

set -e


#pip install huggingface-hub

find ./videos -type f -print0 \
    | xargs -0 -I {} python scripts/upload_hf.py upload_file -rt dataset -r weege007/youtube_videos -f {} -p {}
