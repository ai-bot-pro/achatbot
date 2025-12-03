import uuid
import json
from pathlib import Path

from . import model_dir, MODEL_VOL, comfyui_out_dir, COMFYUI_OUT_VOL, clear_comfyui_output_dir


def link_comfyui_dir():
    """
    ymlink the model to the right ComfyUI directory
    NOTE: need link ckpt file, don't to link models dir
    """
    clear_comfyui_output_dir()


def change_workflow_conf(file_path: Path, **kwargs) -> str:
    workflow_data = json.loads(file_path.read_text())
    client_id = uuid.uuid4().hex
    if kwargs.get("prompt"):
        workflow_data["1"]["inputs"]["prompt"] = kwargs.get("prompt")

    if kwargs.get("width"):
        workflow_data["2"]["inputs"]["width"] = kwargs.get("width")
    if kwargs.get("height"):
        workflow_data["2"]["inputs"]["height"] = kwargs.get("height")
    if kwargs.get("length"):
        workflow_data["2"]["inputs"]["num_frames"] = kwargs.get("length")
    if kwargs.get("steps"):
        workflow_data["2"]["inputs"]["num_inference_steps"] = kwargs.get("steps")

    images = kwargs.get("images")
    workflow_data["2"]["inputs"]["image_path"] = -99999
    if images is not None and isinstance(images, list) and len(images) > 0:
        # now just support one image input
        if isinstance(images[0], bytes):
            input_image_filename = f"input_{client_id}.jpeg"
            input_image_path = f"{comfyui_out_dir}/{input_image_filename}"
            with open(input_image_path, "wb") as f:
                f.write(images[0])
            workflow_data["7"]["inputs"]["image"] = input_image_path
    else:
        del workflow_data["7"]

    print(file_path, workflow_data)
    new_workflow_file = f"{client_id}.json"
    json.dump(workflow_data, Path(new_workflow_file).open("w", encoding="utf-8"))

    return client_id
