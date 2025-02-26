import os
import json
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from utils_trajectory import DummyCamera
from data_augmentations import depth_to_color

def clean_prompt(prompt_text):
    return prompt_text.lower().replace("\n","").replace(".","").replace("  "," ")

class JSONLDataset(Dataset):
    def __init__(self, jsonl_file_path: str, image_directory_path=None, return_depth=False, return_camera=True, augment_rgb=None, clean_prompt=True, augment_text=None, depth_to_color=True):
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
        self.entries = self._load_entries()
        self.clean_promt = clean_prompt
        self.return_camera = return_camera
        self.augment_rgb = augment_rgb
        self.augment_text = augment_text
        self.return_depth = return_depth
        self.depth_to_color = depth_to_color

        # TODO(maxim):
        # if self.return_camera:
        #    load _annotations.all.jsonl -> get camera 
        #    dict[index] -> camera


    def _load_entries(self):
        entries = []
        with open(self.jsonl_file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                entries.append(data)
        return entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self.entries):
            raise IndexError("Index out of range")
        
        entry = self.entries[idx]

        image_path = os.path.join(self.image_directory_path, entry['image'])
        image = Image.open(image_path)

        if self.return_camera:
            image_width, image_height = image.size
            camera_extrinsic = [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]]
            camera_intrinsic = [[[410.029, 0.0, 224.0], [0.0, 410.029, 224.0], [0.0, 0.0, 1.0]]]
            camera = DummyCamera(camera_intrinsic, camera_extrinsic, width=image_width, height=image_height)
            entry["camera"] = camera
        
        if self.augment_rgb is not None:
            image = self.augment_rgb(image)

        if self.clean_promt:
            entry["prefix"] = clean_prompt(entry["prefix"])

        if self.augment_text:
            entry["prefix"] = self.augment_text(entry["prefix"])

        if self.return_depth:
            import numpy as np
            depth_path = os.path.join(self.image_directory_path, entry['image'].replace("_first.jpg", "_depth0.png"))
            depth_xs = Image.open(depth_path)
            depth_xs = np.array(depth_xs)
            depth_xs[depth_xs==65535] = 0
            scaling_png_to_mm = 3 / 65535 * 1000
            depth_mm = depth_xs * scaling_png_to_mm
            if self.depth_to_color:
                depth_mm = depth_to_color(np.expand_dims(depth_mm, axis=2))
            return (depth_mm, image), entry

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
