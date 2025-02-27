from torch.utils.data import Dataset
from PIL import Image
import os

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the root directory containing subdirectories of images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        # Collect all image paths and labels
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        # Walk through subdirectories
        for class_idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):  # Only process directories
                self.class_to_idx[class_name] = class_idx
                for file_name in os.listdir(class_path):
                    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):  # Check valid image extensions
                        self.image_paths.append(os.path.join(class_path, file_name))
                        self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")  # Ensure RGB format

        # Apply transformations if given
        if self.transform:
            image = self.transform(image)

        return image, label