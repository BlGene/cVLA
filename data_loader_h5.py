import glob
import time
import h5py
import random
from pathlib import Path
import torch
from PIL import Image
from collections import defaultdict
import numpy as np
import pickle
from mani_skill.utils.structs import Pose
from mani_skill.examples.utils_traj_tokens import to_prefix_suffix
from mani_skill.examples.utils_traj_tokens import getActionEncInstance
from utils_trajectory import DummyCamera
from torch.utils.data import Dataset
from data_augmentations import depth_to_color



class H5Dataset(Dataset):
    def __init__(self, h5_file_or_dir, return_depth=False, augment_rgbds=None, augment_rgb=None,
                 augment_depth=None, depth_to_color=True, augment_text=None, return_only_prefix=False,
                 action_encoder="xyzrotvec-cam-1024xy", limit_samples=None):
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

        self.return_only_prefix = return_only_prefix        # used only for paired dataset for setup

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
    
        
    def getitem_func(self, idx: int):
        action_text = None
        for x in self.h5_file[f'traj_{idx}/obs/extra'].keys():
            if x.startswith("action_text_"):
                action_text = str(x).replace("action_text_", "")

        frame_idx = slice(0,1)
        
        obj_start = Pose(torch.tensor(self.h5_file[f"traj_{idx}/obs/extra/obj_start"][frame_idx]))
        obj_end = Pose(torch.tensor(self.h5_file[f"traj_{idx}/obs/extra/obj_end"][frame_idx]))
        grasp_pose = Pose(torch.tensor(self.h5_file[f"traj_{idx}/obs/extra/grasp_pose"][frame_idx]))
        tcp_pose = Pose(torch.tensor(self.h5_file[f"traj_{idx}/obs/extra/tcp_pose"][frame_idx]))
        camera_intrinsic = self.h5_file[f"traj_{idx}/obs/sensor_param/render_camera/intrinsic_cv"][frame_idx]
        camera_extrinsic = self.h5_file[f"traj_{idx}/obs/sensor_param/render_camera/extrinsic_cv"][frame_idx]
        #print("XXX", obj_start.shape)
        
        image = self.h5_file[f'traj_{idx}/obs/sensor_data/render_camera/rgb'][0]
        width, height, c = image.shape
        camera = DummyCamera(camera_intrinsic, camera_extrinsic, width, height)

        if self.augment_text is not None:
            action_text = self.augment_text(action_text)

        enc = self.action_encoder.encode_trajectory
        prefix, token_str, curve_3d, orns_3d, info = to_prefix_suffix(obj_start, obj_end,
                                                                       camera, grasp_pose, tcp_pose,
                                                                       action_text, enc, robot_pose=None)
        entry = dict(prefix=prefix, suffix=token_str, camera=camera)

        if self.return_only_prefix:     # used only for paired dataset for setup
            entry["camera_intrinsic"] = camera_intrinsic
            entry["camera_extrinsic"] = camera_extrinsic
            return entry

        depth = None
        seg = None
        if self.augment_rgbds is not None:
            depth = self.h5_file[f'traj_{idx}/obs/sensor_data/render_camera/depth'][0][:,:,0]
            depth = np.clip(depth, 0, 1023)
            seg = self.h5_file[f'traj_{idx}/obs/sensor_data/render_camera/segmentation'][0]
            image = self.augment_rgbds(image, depth, seg)

        if self.augment_rgb is not None:
            image = self.augment_rgb(image)

        if self.return_depth:
            if depth is None:
                depth = self.h5_file[f'traj_{idx}/obs/sensor_data/render_camera/depth'][0][:,:,0]
                depth = np.clip(depth, 0, 1023)

            if self.augment_depth is not  None:
                depth, suffix = self.augment_depth(depth, entry["suffix"])
                entry["suffix"] = suffix
            
            if self.depth_to_color:
                depth = depth_to_color(depth)
                depth = np.clip((depth * 255).round(), 0, 255).astype(np.uint8)

            #else:
            #    depth = depth[:,:,0]
            return [depth, image], entry
        
        return Image.fromarray(image), entry



class PairedH5Dataset(Dataset):
    def __init__(self, h5_dataset, num_images_in_context=1, image_order="interleaved", load_presampled_pairs_path=None):
        self.h5_dataset = h5_dataset
        self.num_images_in_context = num_images_in_context
        self.image_order = image_order
        self.load_presampled_pairs_path = load_presampled_pairs_path
        assert self.image_order in ["interleaved", "images_first"]  # interleaved is image, text, image..., images_first is image, image, text,...

        if self.load_presampled_pairs_path is not None and load_presampled_pairs_path.exists():
            print(f"Loading pre-sampled pairs from {load_presampled_pairs_path}")
            # load presampled pickle and save it to self.task_lookup
            with open(load_presampled_pairs_path, "rb") as f:
                self.task_lookup = pickle.load(f)
        else:

            # setup - define a lookup table for image idx and tasks they are performing
            self.h5_dataset.return_only_prefix = True
            self.task_lookup = defaultdict(list)
            for i in range(len(self.h5_dataset)):
                entry = self.h5_dataset[i]
                prefix = entry["prefix"].split("<")[0].strip()
                self.task_lookup[prefix].append(i)

            self.h5_dataset.return_only_prefix = False

            # if load_presampled_pairs_path is not None and does not exist, save the pre-sampled pairs
            if self.load_presampled_pairs_path is not None and not load_presampled_pairs_path.exists():
                print(f"Saving pre-sampled pairs to {load_presampled_pairs_path}")
                with open(load_presampled_pairs_path, "wb") as f:
                    pickle.dump(self.task_lookup, f)

        # if there are tasks with only one image, we need to remove them
        self.task_lookup = {k:v for k,v in self.task_lookup.items() if len(v) > 1}

        self.paired_len = sum([len(v) for v in self.task_lookup.values()])  # number of possible pairs

        # print statistics about the dataset
        print("Statistics about the paired dataset:")
        print(f"Number of tasks with more than one image: {len(self.task_lookup)}, total number of pairs: {self.paired_len}")
        # print(f"Tasks and number of images: {[(k, len(v)) for k,v in self.task_lookup.items()]}")

    def __len__(self):
        return self.paired_len
            
    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        
        # sample a task                 # TODO: Change to weighted sampling, now it is only uniform
        task = np.random.choice(list(self.task_lookup.keys()))

        # sample random images to be put into context
        context_idx = np.random.choice(self.task_lookup[task], self.num_images_in_context + 1, replace=False)
        images, entries = [], []
        for i in context_idx:
            image, entry = self.h5_dataset[i]
            images.append(image)
            entries.append(entry)
        
        return images, entries



