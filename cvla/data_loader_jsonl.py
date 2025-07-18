import os
import json
import random
import numpy as np

import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R

from mani_skill.utils.structs.pose import Pose
from cvla.utils_trajectory import DummyCamera
from cvla.utils_traj_tokens import getActionEncInstance, to_prefix_suffix
from cvla.data_augmentations import depth_to_color as depth_to_color_func



class JSONLDataset(Dataset):
    def __init__(self, jsonl_file_path: str, image_directory_path=None, return_depth=False,
                 augment_rgb=None, augment_text=None, augment_depth=None, depth_to_color=True,
                 augment_crop=None, limit_samples=None, action_encoder="xyzrotvec-cam-1024xy",
                 train_ratio: float = 0.8, seed: int = 42, split="train", return_format="paligemma"):
        jsonl_file_path = Path(jsonl_file_path)
        if jsonl_file_path.is_file():
            dataset_path = jsonl_file_path.parent
            jsonl_file_path = jsonl_file_path
        elif jsonl_file_path.is_dir():
            dataset_path = jsonl_file_path
            jsonl_file_path = jsonl_file_path / "_annotations.all.jsonl"
        else:
            raise ValueError(f"didn't find {jsonl_file_path}")
        if image_directory_path is None:
            image_directory_path = Path(dataset_path) / "dataset"

        self.dataset_path = dataset_path
        self.jsonl_file_path = jsonl_file_path
        self.image_directory_path = image_directory_path
        self.entries = self._load_entries(jsonl_file_path)

        if "-droid-" in str(self.dataset_path):
            dataset_orn = R.from_euler("xyz", (180,0, 90), degrees=True)
            self.dataset_orn_offset = Pose.create_from_pq(q=dataset_orn.as_quat(scalar_first=True))
        else:
            raise NotImplementedError(f"dataset_orn_offset not implemented for {self.dataset_path}, double check with dataset_compare.ipynb")

        random.seed(seed)
        indexes_all = np.arange(len(self.entries))
        random.shuffle(indexes_all)
        split_index = int(len(indexes_all) * train_ratio)
        assert split in ["train", "valid"], f"split must be 'train' or 'valid', got {split}"
        indices_split = sorted(indexes_all[:split_index]) if split == "train" else sorted(indexes_all[split_index:])
        if limit_samples is not None:
            assert limit_samples <= len(indices_split), f"limit_samples was {limit_samples}, must be less than {len(indices_split)}"
            indices_split = np.array(indices_split)[np.linspace(0, len(indices_split) - 1, limit_samples, dtype=int)]
        
        self.action_encoder = getActionEncInstance(action_encoder)
        self.all_entries = self.entries
        entries_new = []
        for i in indices_split:
            entry = self.entries[i]
            new_entry = self.encode_actions(entry)
            new_entry["line_idx"] = i
            entries_new.append(new_entry)
        self.entries = entries_new

        self.augment_text = augment_text
        self.augment_rgb = augment_rgb
        self.augment_depth = augment_depth
        self.return_depth = return_depth
        self.depth_to_color = depth_to_color
        self.augment_crop = augment_crop
        self.return_only_prefix = False
        self.return_format = return_format

    def _load_entries(self, json_path: str):
        entries = []
        with open(json_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                entries.append(data)
        return entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int):
        return self.getitem_func(idx)

    def get(self, idx, rnd=None):
        return self.__getitem__(idx)

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
    
    def encode_actions(self, label):
        """
        This code should be similar to that in gen_dataset.py
        """
        img_path = self.image_directory_path / label["image"]
        width, height = Image.open(img_path).size
        camera = DummyCamera(label["camera_intrinsic"], label["camera_extrinsic"], width, height)
        obj_start_pose = Pose(raw_pose=torch.tensor(label["obj_start_pose"]))
        obj_end_pose = Pose(raw_pose=torch.tensor(label["obj_end_pose"]))
        grasp_pose = Pose(raw_pose=torch.tensor(label["grasp_pose"])) * self.dataset_orn_offset
        tcp_pose = Pose(raw_pose=torch.tensor(label["tcp_start_pose"])) * self.dataset_orn_offset
        robot_pose = Pose(raw_pose=torch.tensor(label["robot_pose"])) * self.dataset_orn_offset
        action_text = label["action_text"]
        enc_traj = self.action_encoder.encode_trajectory
        prefix, suffix, _, _, info = to_prefix_suffix(obj_start_pose, obj_end_pose, camera, grasp_pose, tcp_pose, action_text, enc_traj, robot_pose=robot_pose)
        return dict(image=img_path, prefix=prefix, suffix=suffix, camera=camera, enc_info=info)


    def getitem_func(self, idx: int, force_augs=False):
        if idx < 0 or idx >= len(self.entries):
            raise IndexError("Index out of range")
        
        entry = self.entries[idx].copy()
        
        if self.return_only_prefix: # used only for paired dataset for setup, no augmentation
            return entry
        
        if self.augment_text:
            assert entry["prefix"] == " ".join((entry["enc_info"]["action_text"], entry["enc_info"]["robot_state"]))
            entry["prefix"] = " ".join((self.augment_text(entry["enc_info"]["action_text"]), entry["enc_info"]["robot_state"]))
                
        image_path = os.path.join(self.image_directory_path, entry['image'])
        image = Image.open(image_path)

        if self.augment_rgb is not None:
            image = self.augment_rgb(image)

        if self.return_depth:
            import numpy as np
            depth_path = os.path.join(self.image_directory_path, str(entry['image']).replace("_first.jpg", "_depth0.png"))
            depth_png = Image.open(depth_path)
            depth_png = np.array(depth_png)
            depth_png[depth_png==3000] = 0
            scaling_png_to_mm = 3 / 65535 * 1000  # for real exports v1 and v2
            #scaling_png_to_mm = 1
            depth = depth_png * scaling_png_to_mm
            if self.augment_depth is not None:
                depth, suffix = self.augment_depth(depth, entry["suffix"])
                entry["suffix"] = suffix
            if self.depth_to_color:
                depth = depth_to_color_func(depth)
                depth = np.clip((depth * 255).round(), 0, 255).astype(np.uint8)

            # TODO(max): remove this code duplication
            if self.augment_crop:
                assert self.depth_to_color
                orig_entry = entry["suffix"]
                image, suffix_new, camera_new = self.augment_crop(image, orig_entry, self.action_encoder, orig_entry["camera"])  # adjust suffix
                depth = torch.tensor(depth)
                depth, _ = self.augment_crop(depth, orig_entry, self.action_encoder)
                entry["suffix"] = suffix_new
                entry["camera"] = camera_new

            assert self.return_format == "molmo"
            return (depth, image), entry
        
        if self.augment_crop:
            assert self.return_depth == False
            image, entry["suffix"], entry["camera"] = self.augment_crop(image, entry["suffix"], self.action_encoder, entry["camera"])  # adjust suffix

        if self.return_format == "molmo":
            return self.format_molmo(image, entry)
        else:
            assert self.return_format == "paligemma"
        return image, entry



class ValidDataset:
    """
    Some JSONL files don't have depth files, create a wrapper to ignore those.
    """
    def __init__(self, dataset):
        self.valid_idxs = []
        for i in range(len(dataset)):
            try:
                dataset[i]
                self.valid_idxs.append(i)
            except FileNotFoundError:
                pass
        self.dataset = dataset
        
    def __len__(self):
        return len(self.valid_idxs)
    
    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self.valid_idxs):
            raise IndexError("Index out of range")
        valid_idx = self.valid_idxs[idx]
        return self.dataset[valid_idx]
