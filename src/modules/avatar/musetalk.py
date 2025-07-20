import argparse
import copy
import glob
import pickle
import queue
import threading
import time
import logging
import os
import shutil
import sys
import json
from typing import Tuple

import cv2
import librosa
import numpy as np
from tqdm import tqdm

from src.common.time_utils import timeit
from src.common.types import RESOURCES_DIR, MODELS_DIR, TEST_DIR
from src.common.factory import EngineClass
from src.modules.avatar.interface import IFaceAvatar
from src.types.avatar import SpeechAudio
from src.common.logger import Logger

Logger.init(os.getenv("LOG_LEVEL", "info").upper(), is_file=False, is_console=True)

try:
    cur_dir = os.path.dirname(__file__)
    musetalk_dir = os.path.join(cur_dir, "../../../deps/MuseTalk")
    if bool(os.getenv("ACHATBOT_PKG", "")):
        musetalk_dir = os.path.join(cur_dir, "../../MuseTalk")
    sys.path.insert(1, musetalk_dir)

    import torch
    from transformers import WhisperModel

    from mmpose.apis import inference_topdown, init_model
    from mmpose.structures import merge_data_samples

    from deps.MuseTalk.musetalk.utils.face_parsing import FaceParsing
    from deps.MuseTalk.musetalk.utils.utils import datagen, load_all_model
    from deps.MuseTalk.musetalk.utils.blending import get_image_prepare_material, get_image_blending
    from deps.MuseTalk.musetalk.utils.audio_processor import AudioProcessor
    from deps.MuseTalk.musetalk.utils.face_detection import FaceAlignment, LandmarksType


except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use lite-avatar, you need to `pip install achatbot[musetalk_avatar]`. "
    )
    raise Exception(f"Missing module: {e}")


def video2imgs(vid_path, save_path, ext=".png", cut_frame=10000000):
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break


def read_imgs(img_list):
    frames = []
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames


def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None


class FaceAlignmentLandmark:
    def __init__(self, model_dir: str = MODELS_DIR):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # initialize the mmpose model
        config_file = os.path.join(
            musetalk_dir,
            "musetalk",
            "utils",
            "dwpose",
            "rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py",
        )
        checkpoint_file = os.path.join(model_dir, "dwpose", "dw-ll_ucoco_384.pth")
        self.model = init_model(config_file, checkpoint_file, device=torch.device(device))

        # initialize the face detection model
        self.fa = FaceAlignment(LandmarksType._2D, flip_input=False, device=device)

        # maker if the bbox is not sufficient
        # coord_placeholder is used to mark invalid bounding boxes
        self.coord_placeholder = (0.0, 0.0, 0.0, 0.0)

    def get_landmark_and_bbox(self, img_list, upperbondrange=0):
        frames = read_imgs(img_list)
        batch_size_fa = 1
        batches = [frames[i : i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
        coords_list = []
        if upperbondrange != 0:
            logging.debug(
                f"get key_landmark and face bounding boxes with the bbox_shift: {upperbondrange}"
            )
        else:
            logging.debug("get key_landmark and face bounding boxes with the default value")
        average_range_minus = []
        average_range_plus = []
        for fb in tqdm(batches):
            results = inference_topdown(self.model, np.asarray(fb)[0])
            results = merge_data_samples(results)
            keypoints = results.pred_instances.keypoints
            face_land_mark = keypoints[0][23:91]
            face_land_mark = face_land_mark.astype(np.int32)

            # get bounding boxes by face detetion
            bbox = self.fa.get_detections_for_batch(np.asarray(fb))

            # adjust the bounding box refer to landmark
            # Add the bounding box to a tuple and append it to the coordinates list
            for j, f in enumerate(bbox):
                if f is None:  # no face in the image
                    coords_list += [self.coord_placeholder]
                    continue

                half_face_coord = face_land_mark[
                    29
                ]  # np.mean([face_land_mark[28], face_land_mark[29]], axis=0)
                range_minus = (face_land_mark[30] - face_land_mark[29])[1]
                range_plus = (face_land_mark[29] - face_land_mark[28])[1]
                average_range_minus.append(range_minus)
                average_range_plus.append(range_plus)
                if upperbondrange != 0:
                    half_face_coord[1] = (
                        upperbondrange + half_face_coord[1]
                    )  # 手动调整  + 向下（偏29）  - 向上（偏28）
                half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]
                upper_bond = half_face_coord[1] - half_face_dist

                f_landmark = (
                    np.min(face_land_mark[:, 0]),
                    int(upper_bond),
                    np.max(face_land_mark[:, 0]),
                    np.max(face_land_mark[:, 1]),
                )
                x1, y1, x2, y2 = f_landmark

                if (
                    y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0
                ):  # if the landmark bbox is not suitable, reuse the bbox
                    coords_list += [f]
                    w, h = f[2] - f[0], f[3] - f[1]
                    print("error bbox:", f)
                else:
                    coords_list += [f_landmark]

        logging.info(
            f"bbox_shift parameter adjustment, Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , the current value: {upperbondrange}"
        )
        return coords_list, frames


class MusetalkAvatar(EngineClass):
    TAG = "musetalk_avatar"

    def __init__(
        self,
        avatar_id="avatar_0",
        material_video_path="./deps/MuseTalk/data/video/sun.mp4",
        bbox_shift=0,
        batch_size=20,  # for warmup / inference
        gen_batch_size=5,  # for frames generation
        force_preparation=False,
        parsing_mode="jaw",
        left_cheek_width=90,
        right_cheek_width=90,
        audio_padding_length_left=2,
        audio_padding_length_right=2,
        fps=25,
        version="v15",
        result_dir="./results",
        extra_margin=10,
        model_dir=os.path.join(MODELS_DIR, "weege007/musetalk"),
        gpu_id=0,
        debug=False,
    ):
        """Initialize MuseAvatarV15

        Args:
            avatar_id (str): Avatar ID
            material_video_path (str): Material Video path
            bbox_shift (int): Face bounding box offset
            batch_size (int): Batch size
            force_preparation (bool): Whether to force data preparation
            parsing_mode (str): Face parsing mode, default 'jaw'
            left_cheek_width (int): Left cheek width
            right_cheek_width (int): Right cheek width
            audio_padding_length_left (int): Audio left padding length
            audio_padding_length_right (int): Audio right padding length
            fps (int): Video frame rate
            version (str): MuseTalk version
            result_dir (str): Output directory for results
            extra_margin (int): Extra margin
            gpu_id (int): GPU device ID
        """
        self.avatar_id = avatar_id
        self.material_video_path = material_video_path
        self.bbox_shift = bbox_shift
        self.batch_size = batch_size
        self.gen_batch_size = gen_batch_size
        self.force_preparation = force_preparation
        self.parsing_mode = parsing_mode
        self.left_cheek_width = left_cheek_width
        self.right_cheek_width = right_cheek_width
        self.audio_padding_length_left = audio_padding_length_left
        self.audio_padding_length_right = audio_padding_length_right
        self.fps = fps
        self.version = version
        self.result_dir = result_dir
        self.extra_margin = extra_margin

        self.model_dir = model_dir
        self.vae_dir = os.path.join(model_dir, "sd-vae")
        self.unet_dir = os.path.join(model_dir, "musetalkV15")
        self.whisper_dir = os.path.join(model_dir, "whisper")
        self.face_parse_dir = os.path.join(model_dir, "face-parse-bisent")

        self.gpu_id = gpu_id
        self.debug = debug

        # Set paths
        if self.version == "v15":
            self.base_path = os.path.join(self.result_dir, self.version, "avatars", avatar_id)
        else:  # v1
            self.base_path = os.path.join(self.result_dir, "avatars", avatar_id)

        self.avatar_path = self.base_path
        self.full_imgs_path = os.path.join(self.avatar_path, "full_imgs")
        self.coords_path = os.path.join(self.avatar_path, "coords.pkl")
        self.latents_out_path = os.path.join(self.avatar_path, "latents.pt")
        self.video_out_path = os.path.join(self.avatar_path, "vid_output")
        self.mask_out_path = os.path.join(self.avatar_path, "mask")
        self.mask_coords_path = os.path.join(self.avatar_path, "mask_coords.pkl")
        self.avatar_info_path = os.path.join(self.avatar_path, "avator_info.json")
        self.frames_path = os.path.join(self.avatar_path, "frames.pkl")
        self.masks_path = os.path.join(self.avatar_path, "masks.pkl")

        self.avatar_info = {
            "avatar_id": avatar_id,
            "material_video_path": material_video_path,
            "bbox_shift": bbox_shift,
            "version": self.version,
        }

        # Model related
        self.device = None
        self.vae = None
        self.unet = None
        self.pe = None
        self.whisper = None
        self.fp = None
        self.audio_processor = None
        self.weight_dtype = None
        self.timesteps = None

        # Data related
        self.input_latent_list_cycle = None
        self.coord_list_cycle = None
        self.frame_list_cycle = None
        self.mask_coords_list_cycle = None
        self.mask_list_cycle = None

        self.fa_landmark = None

    def load(self):
        """Initialize digital avatar and Load model weights

        Automatically determine whether to regenerate data by checking the integrity of files in the avatar directory.
        If force_preparation is True, force regeneration.
        Files to check include:
        1. latents.pt - latent features file
        2. coords.pkl - face coordinates file
        3. mask_coords.pkl - mask coordinates file
        4. avator_info.json - config info file
        5. frames.pkl - frame data file
        6. masks.pkl - mask data file
        """
        # 1. Check if data preparation is needed
        required_files = [
            self.latents_out_path,  # latent features file
            self.coords_path,  # face coordinates file
            self.mask_coords_path,  # mask coordinates file
            self.avatar_info_path,  # config info file
            self.frames_path,  # frame data file
            self.masks_path,  # mask data file
        ]

        # Check if data needs to be generated
        need_preparation = self.force_preparation  # If force regeneration, set to True

        if not need_preparation and os.path.exists(self.avatar_path):
            # Check if all required files exist
            for file_path in required_files:
                if not os.path.exists(file_path):
                    need_preparation = True
                    break

            # If config file exists, check if bbox_shift has changed
            if os.path.exists(self.avatar_info_path):
                with open(self.avatar_info_path, "r") as f:
                    avatar_info = json.load(f)
                if avatar_info["bbox_shift"] != self.avatar_info["bbox_shift"]:
                    logging.error(
                        f"bbox_shift changed from {avatar_info['bbox_shift']} to {self.avatar_info['bbox_shift']}, need re-preparation"
                    )
                    need_preparation = True
        else:
            need_preparation = True

        # 2. Initialize device and models
        self.device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")
        self.timesteps = torch.tensor([0], device=self.device)

        # Load models
        self.vae, self.unet, self.pe = load_all_model(
            vae_dir=self.vae_dir,
            unet_dir=self.unet_dir,
            device=self.device,
        )

        # Convert to half precision
        self.pe = self.pe.half().to(self.device)
        self.vae.vae = self.vae.vae.half().to(self.device)
        self.unet.model = self.unet.model.half().to(self.device)
        self.weight_dtype = self.unet.model.dtype

        self.fa_landmark = FaceAlignmentLandmark(model_dir=self.model_dir)

        # Initialize audio processor and Whisper model
        self.audio_processor = AudioProcessor(feature_extractor_path=self.whisper_dir)
        self.whisper = WhisperModel.from_pretrained(self.whisper_dir)
        self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
        self.whisper.requires_grad_(False)

        # Initialize face parser
        if self.version == "v15":
            self.fp = FaceParsing(
                left_cheek_width=self.left_cheek_width,
                right_cheek_width=self.right_cheek_width,
                model_dir=self.face_parse_dir,
            )
        else:
            self.fp = FaceParsing(model_dir=self.face_parse_dir)

        # 3. Prepare or load data
        if need_preparation:
            if self.force_preparation:
                logging.info(f"  force creating avatar: {self.avatar_id}")
            else:
                logging.info(f"  creating avatar: {self.avatar_id}")
            # If directory exists but needs regeneration, delete it first
            if os.path.exists(self.avatar_path):
                shutil.rmtree(self.avatar_path)
            # Create required directories
            osmakedirs(
                [self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path]
            )
            # Generate data
            self.prepare_material()
        else:
            logging.info(
                f"Avatar {self.avatar_id} exists and is complete, loading existing data..."
            )
            # Load existing data
            self.input_latent_list_cycle = torch.load(self.latents_out_path)
            with open(self.coords_path, "rb") as f:
                self.coord_list_cycle = pickle.load(f)
            with open(self.frames_path, "rb") as f:
                self.frame_list_cycle = pickle.load(f)
            with open(self.mask_coords_path, "rb") as f:
                self.mask_coords_list_cycle = pickle.load(f)
            with open(self.masks_path, "rb") as f:
                self.mask_list_cycle = pickle.load(f)

        logging.info(f"load is done!")

        # Warm up models is only needed in current thread
        logging.info("Warming up models...")
        self._warmup_models()
        logging.info("Warmup complete")

    def _warmup_models(self):
        """
        Warm up all models and feature extraction pipeline to avoid first-frame delay.
        """
        import time

        t_warmup_start = time.time()
        whisper_warmup_time = 0
        generate_frames_warmup_time = 0
        whisper_warmup_ok = False
        generate_frames_warmup_ok = False

        try:
            t0 = time.time()
            self._warmup_whisper_feature()
            whisper_warmup_time = time.time() - t0
            whisper_warmup_ok = True
        except Exception as e:
            logging.error(f"extract_whisper_feature warmup error: {str(e)}", exc_info=True)

        try:
            t0 = time.time()
            dummy_whisper = torch.zeros(
                self.batch_size, 50, 384, device=self.device, dtype=self.weight_dtype
            )
            _ = self.generate_frames(dummy_whisper, 0, self.batch_size)
            generate_frames_warmup_time = time.time() - t0
            generate_frames_warmup_ok = True
        except Exception as e:
            logging.error(f"generate_frames warmup error: {str(e)}", exc_info=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_warmup_end = time.time()
        logging.info(
            f"All models warmed up via generate_frames pipeline (batch_size={self.batch_size}, zeros) | "
            f"extract_whisper_feature: {whisper_warmup_time * 1000:.1f} ms ({'OK' if whisper_warmup_ok else 'FAIL'}), "
            f"generate_frames: {generate_frames_warmup_time * 1000:.1f} ms ({'OK' if generate_frames_warmup_ok else 'FAIL'}), "
            f"total: {(t_warmup_end - t_warmup_start) * 1000:.1f} ms (with CUDA sync)"
        )

    def _warmup_whisper_feature(self):
        warmup_sr = 16000
        dummy_audio = np.zeros(warmup_sr, dtype=np.float32)
        _ = self.extract_whisper_feature(dummy_audio, warmup_sr)

    def prepare_material(self):
        """Prepare all materials needed for the digital avatar

        This method is the core of the first stage, mainly completes the following tasks:
        1. Save basic avatar info
        2. Process input video/image sequence
        3. Extract face features and bounding boxes
        4. Generate face masks
        5. Save all processed data
        """
        logging.info("preparing data materials ... ...")

        # Step 1: Save basic avatar config info
        with open(self.avatar_info_path, "w") as f:
            json.dump(self.avatar_info, f)

        # Step 2: Process input source (support video file or image sequence)
        if os.path.isfile(self.material_video_path):
            # If input is a video file, use video2imgs to extract frames
            video2imgs(self.material_video_path, self.full_imgs_path, ext="png")
        else:
            # If input is an image directory, copy all png images directly
            logging.info(f"copy files in {self.material_video_path}")
            files = os.listdir(self.material_video_path)
            files.sort()
            files = [file for file in files if file.split(".")[-1] == "png"]
            for filename in files:
                shutil.copyfile(
                    f"{self.material_video_path}/{filename}", f"{self.full_imgs_path}/{filename}"
                )

        # Get all input image paths and sort
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, "*.[jpJP][pnPN]*[gG]")))

        # Step 3: Extract face landmarks and bounding boxes
        logging.info("extracting landmarks...")
        coord_list, frame_list = self.fa_landmark.get_landmark_and_bbox(
            input_img_list, self.bbox_shift
        )

        # Step 4: Extract latent features
        input_latent_list = []
        idx = -1
        for bbox, frame in zip(coord_list, frame_list):
            idx = idx + 1
            # coord_placeholder is used to mark invalid bounding boxes
            if bbox == self.fa_landmark.coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox

            # Extra margin handling for v15 version
            if self.version == "v15":
                y2 = y2 + self.extra_margin  # Add extra chin area
                y2 = min(y2, frame.shape[0])  # Ensure not out of image boundary
                y1 = max(y1, 0)  # Ensure not out of image boundary
                coord_list[idx] = [x1, y1, x2, y2]  # Update bbox in coord_list

            # Crop face region and resize to 256x256
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(
                crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4
            )

            # Use VAE to extract latent features
            latents = self.vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)

        # Step 5: Build cycle sequence (by forward + reverse order)
        self.frame_list_cycle = frame_list + frame_list[::-1]
        self.coord_list_cycle = coord_list + coord_list[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []

        # Step 6: Generate and save masks
        for i, frame in enumerate(tqdm(self.frame_list_cycle)):
            # Save processed frame
            cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.png", frame)

            # Get current frame's face bbox
            x1, y1, x2, y2 = self.coord_list_cycle[i]

            # Select face parsing mode by version
            if self.version == "v15":
                mode = self.parsing_mode  # v15 supports different parsing modes
            else:
                mode = "raw"  # v1 only supports raw mode

            # Generate mask and crop box
            mask, crop_box = get_image_prepare_material(
                frame, [x1, y1, x2, y2], fp=self.fp, mode=mode
            )

            # Save mask and related info
            cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png", mask)
            self.mask_coords_list_cycle += [crop_box]
            self.mask_list_cycle.append(mask)

        # Step 7: Save all processed data
        # Save mask coordinates
        with open(self.mask_coords_path, "wb") as f:
            pickle.dump(self.mask_coords_list_cycle, f)

        # Save face coordinates
        with open(self.coords_path, "wb") as f:
            pickle.dump(self.coord_list_cycle, f)

        # Save latent features
        torch.save(self.input_latent_list_cycle, self.latents_out_path)

        # Save frame data
        with open(self.frames_path, "wb") as f:
            pickle.dump(self.frame_list_cycle, f)

        # Save mask data
        with open(self.masks_path, "wb") as f:
            pickle.dump(self.mask_list_cycle, f)

    def res2combined(self, res_frame, idx):
        """Blend the generated frame with the original frame
        Args:
            res_frame: Generated frame (numpy array)
            idx: Current frame index
        Returns:
            numpy.ndarray: Blended full frame
        """
        t0 = time.time()
        # Get the face bbox and original frame for the current frame
        bbox = self.coord_list_cycle[idx % len(self.coord_list_cycle)]
        ori_frame = copy.deepcopy(self.frame_list_cycle[idx % len(self.frame_list_cycle)])
        t1 = time.time()
        x1, y1, x2, y2 = bbox
        try:
            # Resize the generated frame to face region size
            res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
        except Exception as e:
            logging.error(f"res2combined error: {str(e)}", exc_info=True)
            return ori_frame
        t2 = time.time()
        # Add protection: if res_frame is all zeros, return original frame directly
        if np.all(res_frame == 0):
            # if self.debug:
            logging.warning(f"res2combined: res_frame is all zero, return ori_frame, idx={idx}")
            return ori_frame
        # Get the corresponding mask and crop box
        mask = self.mask_list_cycle[idx % len(self.mask_list_cycle)]
        mask_crop_box = self.mask_coords_list_cycle[idx % len(self.mask_coords_list_cycle)]
        t3 = time.time()
        # Blend the generated facial expression with the original frame
        combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
        t4 = time.time()
        if self.debug:
            logging.info(
                f"[PROFILE] res2combined: idx={idx}, ori_copy={t1 - t0:.4f}s, resize={t2 - t1:.4f}s, mask_fetch={t3 - t2:.4f}s, blend={t4 - t3:.4f}s, total={t4 - t0:.4f}s"
            )
        return combine_frame

    def extract_whisper_feature(self, segment: np.ndarray, sampling_rate: int) -> torch.Tensor:
        """
        Extract whisper features for a single audio segment
        """
        t0 = time.time()
        audio_feature = self.audio_processor.feature_extractor(
            segment, return_tensors="pt", sampling_rate=sampling_rate
        ).input_features
        if self.weight_dtype is not None:
            audio_feature = audio_feature.to(dtype=self.weight_dtype)
        whisper_chunks = self.audio_processor.get_whisper_chunk(
            [audio_feature],
            self.device,
            self.weight_dtype,
            self.whisper,
            len(segment),
            fps=self.fps,
            audio_padding_length_left=self.audio_padding_length_left,
            audio_padding_length_right=self.audio_padding_length_right,
        )
        t1 = time.time()
        if self.debug:
            logging.info(
                f"[PROFILE] extract_whisper_feature: duration={t1 - t0:.4f}s, segment_len={len(segment)}, sampling_rate={sampling_rate}"
            )
        return whisper_chunks  # shape: [num_frames, 50, 384]

    @torch.no_grad()
    def generate_frame(self, whisper_chunk: torch.Tensor, idx: int) -> np.ndarray:
        """
        Generate a frame based on whisper features and frame index
        """
        import time

        t0 = time.time()
        # Ensure whisper_chunk shape is (B, 50, 384)
        if whisper_chunk.ndim == 2:
            whisper_chunk = whisper_chunk.unsqueeze(0)
        t1 = time.time()
        latent = self.input_latent_list_cycle[idx % len(self.input_latent_list_cycle)]
        if latent.dim() == 3:
            latent = latent.unsqueeze(0)
        t2 = time.time()
        audio_feature = self.pe(whisper_chunk.to(self.device))
        t3 = time.time()
        latent = latent.to(device=self.device, dtype=self.unet.model.dtype)
        t4 = time.time()
        pred_latents = self.unet.model(
            latent, self.timesteps, encoder_hidden_states=audio_feature
        ).sample

        t5 = time.time()
        pred_latents = pred_latents.to(device=self.device, dtype=self.vae.vae.dtype)
        recon = self.vae.decode_latents(pred_latents)
        t6 = time.time()
        res_frame = recon[0]  # Only one frame, take the first
        combined_frame = self.res2combined(res_frame, idx)
        t7 = time.time()

        # Profile statistics, print average every 1 second
        if self.debug:
            if not hasattr(self, "_profile_stat"):
                self._profile_stat = {
                    "count": 0,
                    "sum": [0.0] * 7,  # 7 stages
                    "last_time": time.time(),
                }
            self._profile_stat["count"] += 1
            self._profile_stat["sum"][0] += t1 - t0
            self._profile_stat["sum"][1] += t2 - t1
            self._profile_stat["sum"][2] += t3 - t2
            self._profile_stat["sum"][3] += t4 - t3
            self._profile_stat["sum"][4] += t5 - t4
            self._profile_stat["sum"][5] += t6 - t5
            self._profile_stat["sum"][6] += t7 - t0
            now = time.time()
            if now - self._profile_stat["last_time"] >= 1.0:
                cnt = self._profile_stat["count"]
                avg = [s / cnt for s in self._profile_stat["sum"]]
                logging.info(
                    f"[PROFILE_AVG] count={cnt} "
                    f"prep_whisper={avg[0]:.4f}s, "
                    f"prep_latent={avg[1]:.4f}s, "
                    f"pe={avg[2]:.4f}s, "
                    f"latent_to={avg[3]:.4f}s, "
                    f"unet={avg[4]:.4f}s, "
                    f"vae={avg[5]:.4f}s, "
                    f"total={avg[6]:.4f}s"
                )
                self._profile_stat["count"] = 0
                self._profile_stat["sum"] = [0.0] * 7
                self._profile_stat["last_time"] = now
        return combined_frame

    def generate_idle_frame(self, idx: int) -> np.ndarray:
        """
        Generate an idle static frame (no inference, for avatar idle/no audio)
        """
        # Directly return a frame from the original frame cycle
        frame = self.frame_list_cycle[idx % len(self.frame_list_cycle)]
        return frame

    @torch.no_grad()
    def generate_frames(
        self, whisper_chunks: torch.Tensor, start_idx: int, batch_size: int
    ) -> list:
        """
        Batch generate multiple frames based on whisper features and frame index
        whisper_chunks: [B, 50, 384]
        start_idx: start frame index
        batch_size: batch size
        Return: List of (recon, idx) tuples, length is batch_size
        """
        t0 = time.time()
        # Ensure whisper_chunks shape is (B, 50, 384)
        if whisper_chunks.ndim == 2:
            whisper_chunks = whisper_chunks.unsqueeze(0)
        elif whisper_chunks.ndim == 3 and whisper_chunks.shape[0] == 1:
            pass
        B = whisper_chunks.shape[0]
        assert B == batch_size, f"whisper_chunks.shape[0] ({B}) != batch_size ({batch_size})"
        idx_list = [start_idx + i for i in range(batch_size)]
        latent_list = []
        t1 = time.time()
        for idx in idx_list:
            latent = self.input_latent_list_cycle[idx % len(self.input_latent_list_cycle)]
            if latent.dim() == 3:
                latent = latent.unsqueeze(0)
            latent_list.append(latent)
        latent_batch = torch.cat(latent_list, dim=0)  # [B, ...]
        t2 = time.time()
        audio_feature = self.pe(whisper_chunks.to(self.device))
        t3 = time.time()
        latent_batch = latent_batch.to(device=self.device, dtype=self.unet.model.dtype)
        t4 = time.time()
        pred_latents = self.unet.model(
            latent_batch, self.timesteps, encoder_hidden_states=audio_feature
        ).sample
        # # Force set pred_latents to all nan for debugging： unet get nan value
        # pred_latents[:] = float('nan')
        t5 = time.time()
        pred_latents = pred_latents.to(device=self.device, dtype=self.vae.vae.dtype)
        recon = self.vae.decode_latents(pred_latents)
        t6 = time.time()
        avg_time = (t6 - t0) / B if B > 0 else 0.0
        if self.debug:
            logging.info(
                f"[PROFILE] generate_frames: start_idx={start_idx}, batch_size={batch_size}, "
                f"prep_whisper={t1 - t0:.4f}s, prep_latent={t2 - t1:.4f}s, pe={t3 - t2:.4f}s, "
                f"latent_to={t4 - t3:.4f}s, unet={t5 - t4:.4f}s, vae={t6 - t5:.4f}s, total={t6 - t0:.4f}s, total_per_frame={avg_time:.4f}s"
            )
            # debug for nan value
            logging.info(
                f"latent_batch stats: min={latent_batch.min().item()}, max={latent_batch.max().item()}, mean={latent_batch.mean().item()}, nan_count={(torch.isnan(latent_batch).sum().item() if torch.isnan(latent_batch).any() else 0)}"
            )
            logging.info(
                f"pred_latents stats: min={pred_latents.min().item()}, max={pred_latents.max().item()}, mean={pred_latents.mean().item()}, nan_count={(torch.isnan(pred_latents).sum().item() if torch.isnan(pred_latents).any() else 0)}"
            )
            if isinstance(recon, np.ndarray):
                logging.info(
                    f"recon stats: min={recon.min()}, max={recon.max()}, mean={recon.mean()}, nan_count={np.isnan(recon).sum()}"
                )
            elif isinstance(recon, torch.Tensor):
                logging.info(
                    f"recon stats: min={recon.min().item()}, max={recon.max().item()}, mean={recon.mean().item()}, nan_count={(torch.isnan(recon).sum().item() if torch.isnan(recon).is_floating_point() else 0)}"
                )
            else:
                logging.info(f"recon type: {type(recon)}")
        return [(recon[i], idx_list[i]) for i in range(B)]

    @torch.no_grad()
    def inference(self, audio_path, out_vid_name, fps, skip_save_images):
        """Inference to generate talking avatar video

        Args:
            audio_path: Input audio file path
            out_vid_name: Output video name (based on audio file name)
            fps: Video frame rate
            skip_save_images: Whether to skip saving intermediate frame images
        """
        # Create temp directory for generated frames
        tmp_dir = os.path.join(self.avatar_path, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        logging.info("start inference")

        # Stage 1: Audio feature extraction #
        start_time = time.time()
        # Use Whisper to extract audio features
        start_time = time.perf_counter()
        whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(
            audio_path, weight_dtype=self.weight_dtype
        )
        get_audio_feature_time = time.perf_counter() - start_time
        print(f"{get_audio_feature_time=:.4f}")
        # Chunk audio features
        start_time = time.perf_counter()
        whisper_chunks = self.audio_processor.get_whisper_chunk(
            whisper_input_features,
            self.device,
            self.weight_dtype,
            self.whisper,
            librosa_length,
            fps=fps,
            audio_padding_length_left=self.audio_padding_length_left,
            audio_padding_length_right=self.audio_padding_length_right,
        )
        get_whisper_chunk_time = time.perf_counter() - start_time
        print(f"{get_whisper_chunk_time=:.4f}")
        logging.info(f"processing audio:{audio_path} costs {(time.time() - start_time) * 1000}ms")

        # Stage 2: Batch generation #
        # Calculate total number of frames to generate
        video_num = len(whisper_chunks)
        # Create result frame queue for multithreaded processing
        res_frame_queue = queue.Queue()
        self.idx = 0

        # Create processing thread
        process_thread = threading.Thread(
            target=self.process_frames, args=(res_frame_queue, video_num, skip_save_images)
        )
        process_thread.start()

        # Create data generator for batch processing
        gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)

        start_time = time.time()

        # Batch generate facial expressions
        for i, (whisper_batch, latent_batch) in enumerate(
            tqdm(gen, total=int(np.ceil(float(video_num) / self.batch_size)))
        ):
            start_time = time.perf_counter()
            # 1. Process audio features
            audio_feature_batch = self.pe(whisper_batch.to(self.device))
            pe_cost = time.perf_counter() - start_time
            print(f"{pe_cost=:.4f}")
            # 2. Prepare latent features
            start_time = time.perf_counter()
            latent_batch = latent_batch.to(device=self.device, dtype=self.unet.model.dtype)
            # 3. Use UNet to generate facial expressions
            pred_latents = self.unet.model(
                latent_batch, self.timesteps, encoder_hidden_states=audio_feature_batch
            ).sample
            pred_latents_cost = time.perf_counter() - start_time
            print(f"{pred_latents_cost=:.4f}")

            # 4. Decode generated latent features
            start_time = time.perf_counter()
            pred_latents = pred_latents.to(device=self.device, dtype=self.vae.vae.dtype)
            recon = self.vae.decode_latents(pred_latents)
            recon_cost = time.perf_counter() - start_time
            print(f"{recon_cost=:.4f}")

            # 5. Put generated frames into queue
            for res_frame in recon:
                res_frame_queue.put(res_frame)

        # Wait for processing thread to finish
        process_thread.join()

        # Stage 3: Post-processing #
        # Output processing time statistics
        if skip_save_images:
            logging.info(
                "Total process time of {} frames without saving images = {}s".format(
                    video_num, time.time() - start_time
                )
            )
        else:
            logging.info(
                "Total process time of {} frames including saving images = {}s".format(
                    video_num, time.time() - start_time
                )
            )

        # Save video if needed
        if out_vid_name is not None and not skip_save_images:
            # 1. Convert image sequence to video
            temp_video = os.path.join(self.avatar_path, "temp.mp4")
            cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {tmp_dir}/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 {temp_video}"
            logging.info(cmd_img2video)
            os.system(cmd_img2video)

            # 2. Combine audio into video
            os.makedirs(self.video_out_path, exist_ok=True)
            output_vid = os.path.join(self.video_out_path, f"{out_vid_name}.mp4")
            cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {temp_video} {output_vid}"
            logging.info(cmd_combine_audio)
            os.system(cmd_combine_audio)

            # 3. Clean up temp files
            os.remove(temp_video)
            shutil.rmtree(tmp_dir)
            print(f"Result saved to: {output_vid}")
        logging.info("\n")

    def process_frames(self, res_frame_queue, video_len, skip_save_images):
        """Process generated video frames

        This method runs in a separate thread and is responsible for processing generated video frames, including:
        1. Get generated frames from the queue
        2. Resize frames to match the original video
        3. Blend generated facial expressions with the original frame
        4. Save processed frames (if needed)

        Args:
            res_frame_queue: Queue for generated frames
            video_len: Total number of frames to process
            skip_save_images: Whether to skip saving intermediate frame images
        """
        logging.info(video_len)
        while True:
            # Exit if all frames have been processed
            if self.idx >= video_len - 1:
                break

            try:
                # Get generated frame from queue, 1s timeout
                res_frame = res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            # Get the face bbox and original frame for the current frame
            bbox = self.coord_list_cycle[self.idx % (len(self.coord_list_cycle))]
            ori_frame = copy.deepcopy(
                self.frame_list_cycle[self.idx % (len(self.frame_list_cycle))]
            )
            x1, y1, x2, y2 = bbox

            try:
                # Resize the generated frame to face region size
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except Exception:
                continue

            # Get the corresponding mask and crop box
            mask = self.mask_list_cycle[self.idx % (len(self.mask_list_cycle))]
            mask_crop_box = self.mask_coords_list_cycle[
                self.idx % (len(self.mask_coords_list_cycle))
            ]

            # Blend the generated facial expression with the original frame
            combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)

            # Save processed frame if needed
            if skip_save_images is False:
                cv2.imwrite(f"{self.avatar_path}/tmp/{str(self.idx).zfill(8)}.png", combine_frame)

            self.idx = self.idx + 1


def run_batch_test(args):
    """Run batch audio test"""
    # Initialize digital avatar
    avatar = MusetalkAvatar(
        avatar_id=args.avatar_id,
        material_video_path=args.material_video_path,
        bbox_shift=args.bbox_shift,
        batch_size=args.batch_size,
        force_preparation=args.force_preparation,
        parsing_mode=args.parsing_mode,
        left_cheek_width=args.left_cheek_width,
        right_cheek_width=args.right_cheek_width,
        audio_padding_length_left=args.audio_padding_length_left,
        audio_padding_length_right=args.audio_padding_length_right,
        fps=args.fps,
        version=args.version,
        result_dir=args.result_dir,
        extra_margin=args.extra_margin,
        model_dir=args.model_dir,
        gpu_id=args.gpu_id,
    )
    avatar.load()

    # Get all audio files in the audio directory
    audio_files = []
    for ext in ["*.wav", "*.mp3"]:
        audio_files.extend(glob.glob(os.path.join(args.audio_dir, ext)))
    audio_files.sort()

    audio_files = [
        os.path.join(TEST_DIR, "audio_files", "asr_example_zh.wav"),
        os.path.join(TEST_DIR, "audio_files", "eng_speech.wav"),
    ]
    # Process each audio file
    for audio_path in audio_files:
        # Use audio file name as output video name
        audio_name = os.path.splitext(os.path.basename(audio_path))[0]

        logging.info(f"\nProcessing audio: {audio_path}")

        # Run inference
        avatar.inference(
            audio_path=audio_path,
            out_vid_name=audio_name,  # Use audio file name directly
            fps=args.fps,
            skip_save_images=args.skip_save_images,
        )


"""
python -m src.modules.avatar.musetalk

python -m src.modules.avatar.musetalk --model_dir ./models/weege007/musetalk

python -m src.modules.avatar.musetalk --version v1 --model_dir ./models/weege007/musetalk
"""

# Run main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version", type=str, default="v15", choices=["v1", "v15"], help="MuseTalk version"
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id")
    parser.add_argument("--model_dir", type=str, default=MODELS_DIR, help="model weights directory")
    parser.add_argument(
        "--result_dir", type=str, default="./results", help="Result output directory"
    )
    parser.add_argument("--extra_margin", type=int, default=10, help="Face crop extra margin")
    parser.add_argument("--fps", type=int, default=25, help="Video frame rate")
    parser.add_argument("--bbox_shift", type=int, default=0, help="Face bbox offset")
    parser.add_argument(
        "--audio_padding_length_left", type=int, default=2, help="Audio left padding"
    )
    parser.add_argument(
        "--audio_padding_length_right", type=int, default=2, help="Audio right padding"
    )
    parser.add_argument("--batch_size", type=int, default=20, help="Inference batch size")
    parser.add_argument("--parsing_mode", type=str, default="jaw", help="Face fusion mode")
    parser.add_argument("--left_cheek_width", type=int, default=90, help="Left cheek width")
    parser.add_argument("--right_cheek_width", type=int, default=90, help="Right cheek width")
    parser.add_argument(
        "--skip_save_images", action="store_true", help="Whether to skip saving images"
    )
    parser.add_argument("--avatar_id", type=str, default="avator_2", help="Avatar ID")
    parser.add_argument(
        "--force_preparation",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Whether to force data regeneration (True/False)",
    )
    parser.add_argument(
        "--material_video_path",
        type=str,
        default=os.path.join("deps/MuseTalk", "data", "video", "sun.mp4"),
        help="Video path",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default=os.path.join("deps/MuseTalk", "data", "audio"),
        help="Audio directory path",
    )

    args = parser.parse_args()

    logging.info("Current config:")
    for arg in vars(args):
        logging.info(f"  {arg}: {getattr(args, arg)}")

    run_batch_test(args)
