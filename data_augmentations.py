import os
import glob
from pathlib import Path
import numpy as np
from PIL import Image
import random
import torch
import torchvision.transforms
from torchvision.transforms import Compose, Normalize


# RGB augmentation -----------------------------------------------------------
list_of_transformations = [
            torchvision.transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05
            ),
            torchvision.transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2)),
            torchvision.transforms.RandomPosterize(bits=7, p=0.2),
            torchvision.transforms.RandomPosterize(bits=6, p=0.2),
            torchvision.transforms.RandomPosterize(bits=5, p=0.2),
            torchvision.transforms.RandomPosterize(bits=4, p=0.2),
            torchvision.transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        ]
transform = torchvision.transforms.Compose(list_of_transformations)

def augment_image_rgb(image_np):
    image_pil = Image.fromarray(image_np)
    image_tensor = transform(image_pil)
    image_np_transformed = np.asarray(image_tensor)  # Convert to (H, W, C)
    return image_np_transformed

# Random Background ----------------------------------------------------------
class RandomizeBackgrounds:
    def __init__(self, p=0.2, background_images_path = "/data/lmbraid19/argusm/datasets/indoorCVPR/Images", mask_idxs=(0, 16, 17)):
        self.background_image_files = glob.glob(os.path.join(background_images_path, "**", "*.jpg"), recursive=True)
        print("Number of bg images:", len(self.background_image_files))
        self.mask_idxs = mask_idxs
        self.prob = p


    def __call__(self, image, depth, seg, size=(448,448)):
        if random.random() > self.prob:
            return image
        
        seg_mask = np.isin(seg[:, :, 0], self.mask_idxs)
        random_image_path = random.choice(self.background_image_files)
        random_image_pil = Image.open(random_image_path).resize(size).convert('RGB')
        random_image = np.asarray(random_image_pil)
        image_aug = image.copy()  # Create a copy to preserve the original
        image_aug[seg_mask] = random_image[seg_mask]
        return image_aug