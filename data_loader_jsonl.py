import os
import json
import pickle
import numpy as np

from PIL import Image
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset

from utils_trajectory import DummyCamera
from utils_traj_tokens import getActionEncInstance
from data_augmentations import depth_to_color as depth_to_color_func

def clean_prompt(prompt_text):
    return prompt_text.lower().replace("\n","").replace(".","").replace("  "," ")

class JSONLDataset(Dataset):
    def __init__(self, jsonl_file_path: str, image_directory_path=None, return_depth=False, return_camera=True, 
                 augment_rgb=None, clean_prompt=True, augment_text=None, augment_depth=None, depth_to_color=True,
                 augment_crop=None, limit_samples=None):
        jsonl_file_path = Path(jsonl_file_path)
        if jsonl_file_path.is_file():
            dataset_path = jsonl_file_path.parent
            jsonl_file_path = jsonl_file_path
        elif jsonl_file_path.is_dir():
            dataset_path = jsonl_file_path
            jsonl_file_path = jsonl_file_path / "_annotations.valid.jsonl"
        else:
            raise ValueError(f"didn't find {jsonl_file_path}")
        if image_directory_path is None:
            image_directory_path = Path(dataset_path) / "dataset"

        self.jsonl_file_path = jsonl_file_path
        self.image_directory_path = image_directory_path
        self.entries = self._load_entries(jsonl_file_path)
        if limit_samples:
            keep_indices = np.linspace(0, len(self.entries) - 1, limit_samples, dtype=int)
            self.entries = [self.entries[i] for i in keep_indices]
            
        self.clean_promt = clean_prompt
        self.return_camera = return_camera
        self.augment_rgb = augment_rgb
        self.augment_text = augment_text
        self.augment_depth = augment_depth
        self.return_depth = return_depth
        self.depth_to_color = depth_to_color
        self.augment_crop = augment_crop
        self.return_only_prefix = False

        self.action_encoder = getActionEncInstance("xyzrotvec-cam-1024xy")

        if self.return_camera:
            jsonl_all_path = Path(dataset_path) / "_annotations.all.jsonl"
            self.all_entries = self._load_entries(jsonl_all_path)


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
        if idx < 0 or idx >= len(self.entries):
            raise IndexError("Index out of range")
        
        entry = self.entries[idx].copy()

        if self.return_only_prefix:     # used only for paired dataset for setup
            entry["prefix"] = clean_prompt(entry["prefix"])
            all_entry = self.all_entries[idx]
            entry["camera_intrinsic"] = all_entry["camera_intrinsic"]
            entry["camera_extrinsic"] = all_entry["camera_extrinsic"]
            return entry

        image_path = os.path.join(self.image_directory_path, entry['image'])
        image = Image.open(image_path)
        
        if self.augment_rgb is not None:
            image = self.augment_rgb(image)

        if self.clean_promt:
            entry["prefix"] = clean_prompt(entry["prefix"])

        if self.augment_text:
            entry["prefix"] = self.augment_text(entry["prefix"])

        if self.augment_crop:
            assert self.return_depth == False
            # Achtung! Incompatible with depth!
            # Suffix was encoded with original image size, must be decoded also with original.
            image, entry["suffix"] = self.augment_crop(image, entry["suffix"])
        
        if self.return_camera:
            image_width, image_height = image.size # must be after crop
            all_entry = self.all_entries[idx]
            camera_extrinsic = all_entry["camera_extrinsic"]
            camera_intrinsic = all_entry["camera_intrinsic"]
            camera = DummyCamera(camera_intrinsic, camera_extrinsic, width=image_width, height=image_height)
            entry["camera"] = camera

        if self.return_depth:
            import numpy as np
            depth_path = os.path.join(self.image_directory_path, entry['image'].replace("_first.jpg", "_depth0.png"))
            depth_png = Image.open(depth_path)
            depth_png = np.array(depth_png)
            depth_png[depth_png==3000] = 0
            #scaling_png_to_mm = 3 / 65535 * 1000  # for real exports v1 and v2
            scaling_png_to_mm = 1
            depth = depth_png * scaling_png_to_mm
            if self.augment_depth is not None:
                depth, suffix = self.augment_depth(depth, entry["suffix"])
                entry["suffix"] = suffix
            if self.depth_to_color:
                depth = depth_to_color_func(depth)
                depth = np.clip((depth * 255).round(), 0, 255).astype(np.uint8)
            return (depth, image), entry
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


class PairedJSONLDataset(Dataset):
    def __init__(self, json_dataset, num_images_in_context=1, image_order="interleaved", load_presampled_pairs_path=None):
        self.json_dataset = json_dataset
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
            self.json_dataset.return_only_prefix = True
            self.task_lookup = defaultdict(list)
            for i in range(len(self.json_dataset)):
                prefix = self.json_dataset[i]
                prefix = prefix.split("<")[0].strip()
                self.task_lookup[prefix].append(i)

            self.json_dataset.return_only_prefix = False

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
            image, entry = self.json_dataset[i]
            images.append(image)
            entries.append(entry)
        
        return images, entries