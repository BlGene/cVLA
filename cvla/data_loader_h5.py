import glob
import time
import json
import random
from pathlib import Path

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from mani_skill.utils.structs import Pose
from cvla.utils_trajectory import DummyCamera
from cvla.utils_traj_tokens import to_prefix_suffix
from cvla.utils_traj_tokens import getActionEncInstance
from cvla.data_augmentations import depth_to_color


class H5Dataset(Dataset):
    def __init__(self, h5_file_or_dir, return_depth=False, augment_depth=None, depth_to_color=True,
                 augment_rgbds=None, augment_rgb=None, augment_text=None, return_only_prefix=False,
                 action_encoder="xyzrotvec-cam-1024xy", limit_samples=None, augment_rgb_forced=None,
                 return_robot_pose=False, return_format="paligemma"):
        """
        The augment functions are applied in order same order as the order of arguments.
        """
        h5_file_or_dir = Path(h5_file_or_dir)
        if h5_file_or_dir.is_dir():
            h5_files = sorted(glob.glob(f"{h5_file_or_dir}/*.h5"))
            h5_file_path = Path(h5_file_or_dir) / h5_files[-1]
            if len(h5_files) > 1:
                print(f"Warning, multiple h5 files, choosing {h5_file_path}")
        elif h5_file_or_dir.is_file():
            h5_file_path = h5_file_or_dir
        else:
            raise ValueError(f"dataset neither file nor dir: {h5_file_or_dir}")
        self.h5_file = h5py.File(h5_file_path, "r")
        self.return_depth = return_depth
        if limit_samples is not None:
            self.h5_file_len = limit_samples
        else:
            self.h5_file_len = len(self.h5_file)

        self.action_encoder_name = action_encoder
        self.action_encoder = getActionEncInstance(action_encoder)
        self.return_depth = return_depth
        self.augment_rgb = augment_rgb
        self.augment_rgbds = augment_rgbds
        self.augment_text = augment_text
        self.augment_depth = augment_depth
        self.depth_to_color = depth_to_color
        self.augment_rgb_forced = augment_rgb_forced        # only for copy-pasting when needed

        self.return_robot_pose = return_robot_pose
        self.return_only_prefix = return_only_prefix        # used only for paired dataset for setup
        self.return_format = return_format

    def __len__(self):
        return self.h5_file_len

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
            
        max_retries = 300  # 15 minutes = 900 seconds / 3 seconds per retry
        for retry_count in range(max_retries):
            try:
                return self.getitem_func(idx)
            except OSError as e:
                print(f"OSError encountered for index {idx}: {e}. Retrying in 3 seconds...")
                time.sleep(3)
    
        # If all retries fail, load a random sample
        print(f"Failed to load index {idx} after 15 minutes. Loading a random sample instead.")
        random_idx = random.randint(0, len(self) - 1)
        return self.getitem_func(random_idx)
    
    @staticmethod
    def format_molmo(image, entry):
        image_np = np.asarray(image)
        #from pdb import set_trace
        #set_trace()
        return {
            "image": image_np,
            "prompt": entry["prefix"],
            "text": entry["suffix"],
            "style": "pointing",
            "metadata": {
                "image": image_np,
                "prompt": entry["prefix"],
                "target_action": entry["suffix"],
                "episode_id" : "unknown",
                "house_id": "unknown",
                "frame_idx": -1,
                "task_type": False,
                "include_scene": False,
                "is_done_frame": False,
                #"target_box": None,
                "camera": entry["camera"]
            }
        }
    
    def get(self, idx, rnd=None):
        return self.__getitem__(idx)

    def getitem_func(self, idx: int, force_augs=False):
        action_text = None
        try:
            raw_data = self.h5_file[f"traj_{idx}/obs_scene"][()]
            decoded = json.loads(raw_data.decode("utf-8"))
            action_text = decoded["text"]
        except KeyError:
            # Old style data for datasets version -8 and older.
            for x in self.h5_file[f'traj_{idx}/obs/extra'].keys():
                if x.startswith("action_text_"):
                    action_text = str(x).replace("action_text_", "")

        frame_idx = slice(0, 1)
        obj_start = Pose(torch.tensor(self.h5_file[f"traj_{idx}/obs/extra/obj_start"][frame_idx]))
        obj_end = Pose(torch.tensor(self.h5_file[f"traj_{idx}/obs/extra/obj_end"][frame_idx]))
        grasp_pose = Pose(torch.tensor(self.h5_file[f"traj_{idx}/obs/extra/grasp_pose"][frame_idx]))
        tcp_pose = Pose(torch.tensor(self.h5_file[f"traj_{idx}/obs/extra/tcp_pose"][frame_idx]))
        robot_pose = Pose(torch.tensor(self.h5_file[f"traj_{idx}/obs/extra/robot_pose"][frame_idx]))

        camera_intrinsic = self.h5_file[f"traj_{idx}/obs/sensor_param/render_camera/intrinsic_cv"][frame_idx]
        camera_extrinsic = self.h5_file[f"traj_{idx}/obs/sensor_param/render_camera/extrinsic_cv"][frame_idx]
        
        image = self.h5_file[f'traj_{idx}/obs/sensor_data/render_camera/rgb'][0]
        width, height, c = image.shape
        camera = DummyCamera(camera_intrinsic, camera_extrinsic, width, height)

        if self.augment_text is not None:
            action_text = self.augment_text(action_text)

        enc_traj = self.action_encoder.encode_trajectory
        prefix, token_str, curve_3d, orns_3d, info = to_prefix_suffix(obj_start, obj_end,
                                                                       camera, grasp_pose, tcp_pose,
                                                                       action_text, enc_traj, robot_pose=robot_pose)
        entry = dict(prefix=prefix, suffix=token_str, camera=camera)

        if self.return_only_prefix:     # used only for paired dataset for setup
            entry["camera_intrinsic"] = camera_intrinsic
            entry["camera_extrinsic"] = camera_extrinsic
            return entry

        if self.return_robot_pose:
            entry["robot_pose"] = robot_pose

        depth = None
        seg = None
        if self.augment_rgbds is not None or force_augs:
            depth = self.h5_file[f'traj_{idx}/obs/sensor_data/render_camera/depth'][0][:,:,0]
            depth = np.clip(depth, 0, 1023)
            seg = self.h5_file[f'traj_{idx}/obs/sensor_data/render_camera/segmentation'][0]
            if force_augs:
                image = self.augment_rgb_forced(image, depth, seg)
            else:
                image = self.augment_rgbds(image, depth, seg)

        if self.augment_rgb is not None:
            image = self.augment_rgb(image)

        if self.return_depth:
            if depth is None:
                depth = self.h5_file[f'traj_{idx}/obs/sensor_data/render_camera/depth'][0][:,:,0]
                depth = np.clip(depth, 0, 1023)  # depth im [mm]

            if self.augment_depth is not  None:
                depth, suffix = self.augment_depth(depth, entry["suffix"], self.action_encoder, camera)
                entry["suffix"] = suffix
            
            if self.depth_to_color:
                depth = depth_to_color(depth)
                depth = np.clip((depth * 255).round(), 0, 255).astype(np.uint8)
            else:
                depth = depth[:, :, 0] # depth im [mm]

            assert self.return_format == "paligemma"
            return [depth, image], entry
        
        if self.return_format == "molmo":
            return self.format_molmo(image, entry)
        else:
            assert self.return_format == "paligemma"
        return image, entry