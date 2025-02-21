import os
import json
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional

def clean_prompt(prompt_text):
    return prompt_text.lower().replace("\n","").replace(".","").replace("  "," ")

class JSONLDataset(Dataset):
    def __init__(self, jsonl_file_path: str, image_directory_path=None, clean_prompt=True):
        jsonl_file_path = Path(jsonl_file_path)
        if jsonl_file_path.is_file():
            dataset_path = jsonl_file_path.parent
            jsonl_file_path = jsonl_file_path
        elif jsonl_file_path.is_dir():
            dataset_path = jsonl_file_path
            jsonl_file_path = jsonl_file_path / "_annotations_train.jsonl"
        else:
            raise ValueError(f"didn't find {jsonl_file_path}")
        if image_directory_path is None:
            image_directory_path = Path(dataset_path) / "dataset"
        self.jsonl_file_path = jsonl_file_path
        self.image_directory_path = image_directory_path
        self.entries = self._load_entries()
        self.clean_promt = clean_prompt

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
        if self.clean_promt:
            entry["prefix"] = clean_prompt(entry["prefix"])
            
        image_path = os.path.join(self.image_directory_path, entry['image'])
        image = Image.open(image_path)
        return image, entry
