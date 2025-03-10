import os
from pathlib import PosixPath

import modal

app = modal.App("train-nano-gpt-monitor")

volume = modal.Volume.from_name("train-slm-volume", create_if_missing=True)
volume_path = PosixPath("/data")
tb_log_path = volume_path / "tb_logs"


tensorboard_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "tensorboard==2.17.1"
)


with tensorboard_image.imports():
    import os
    import tensorboard

# ### Monitor experiments with TensorBoard
"""
modal serve src/train/nano_gpt/monitor_tensorboard.py

open the TensorBoard dashboard in your browser: (init load dashboard need some time)
e.g.: https://weedge--train-nano-gpt-monitor-monitor-training-dev.modal.run
"""


@app.function(
    image=tensorboard_image,
    volumes={volume_path: volume},
    allow_concurrent_inputs=1000,
)
@modal.wsgi_app()
def monitor_training():
    import time

    print("📈 TensorBoard: Waiting for logs...")
    ct = 0
    while not tb_log_path.exists():
        ct += 1
        if ct > 10:
            raise Exception("No logs found after 10 seconds.")
        volume.reload()  # make sure we have the latest data.
        time.sleep(1)

    # start TensorBoard server looking at all experiments
    board = tensorboard.program.TensorBoard()
    board.configure(logdir=str(tb_log_path))
    (data_provider, deprecated_multiplexer) = board._make_data_provider()
    wsgi_app = tensorboard.backend.application.TensorBoardWSGIApp(
        board.flags,
        board.plugin_loaders,
        data_provider,
        board.assets_zip_provider,
        deprecated_multiplexer,
    )
    return wsgi_app
