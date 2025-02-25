import os
import glob
from pathlib import Path
import numpy as np
from PIL import Image
import random
import torch
import torchvision.transforms
from torchvision.transforms import Compose, Normalize
from torch.utils.data import Dataset
import re

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
    def __init__(self, p=0.2, background_images = "/data/lmbraid19/argusm/datasets/indoorCVPR/Images", mask_idxs=(0, 16, 17)):

        if isinstance(background_images, (str, Path)):
            self.background_images = glob.glob(os.path.join(background_images, "**", "*.jpg"), recursive=True)
            print("Number of bg images:", len(self.background_images))
            if len(self.background_images) == 0:
                raise ValueError(f"No backgound images found in path {background_images}")
        
        elif isinstance(background_images, Dataset):
            self.background_images = background_images
            self.background_images_len = len(background_images)
        else:
            return ValueError
        
        self.mask_idxs = mask_idxs
        self.prob = p

    def get_bg_image(self):
        if isinstance(self.background_images, list):
            random_image_path = random.choice(self.background_images)
            random_image_pil = Image.open(random_image_path)
        elif isinstance(self.background_images, Dataset):
            random_idx =  random.randrange(self.background_images_len)
            random_image_pil = self.background_images[random_idx][0]
        else:
            raise ValueError
        return random_image_pil

    def __call__(self, image, depth, seg, size=(448,448)):
        if random.random() > self.prob:
            return image
        
        seg_mask = np.isin(seg[:, :, 0], self.mask_idxs)
        random_image_pil = self.get_bg_image().resize(size).convert('RGB')
        random_image = np.asarray(random_image_pil)
        image_aug = image.copy()  # Create a copy to preserve the original
        image_aug[seg_mask] = random_image[seg_mask]
        return image_aug
    

# Text ----------------------------------------------------------

# Dictionary where values will be replaced with their corresponding keys
clevr_move_replacement_dict = {
    "pick up the": "move",
    "place the": "move",
    "get the": "move",
    "put the": "move",
    "pick the": "move",
    "building block": "cube",
    "block": "cube",
    "thing": "cube",
    "and put it in the": "onto",
    "inside the": "onto",
    "and place it in the": "onto",
    "in the": "onto",
    "into the": "onto",
    "inside": "onto",
}
# Create regex pattern to match all dictionary keys
pattern = re.compile(r'\b(' + '|'.join(map(re.escape, clevr_move_replacement_dict.keys())) + r')\b')
def simplyify_text(text):
    return pattern.sub(lambda match: clevr_move_replacement_dict[match.group(0)], text)

real_block_sent = {'put the {} in the {}': 92, 'put the {} inside the {}': 30, 'place the {} inside the {}': 13,
                   'pick the {} and put it in the {}': 7, 'pick up the {} and put it in the {}': 7, 'put the {} into the {}': 5,
                   'place the {} in the {}': 3, 'pick up the {} and put it inside the {}': 1, 'put the {} inside {}': 1,
                   'get the {} and place it in the {}': 1}

real_block_word = {'cube': {'block': 118, 'cube': 21, 'box': 2}}

def complexify_text(text):
    _, size_1, color_1, shape_1, _, size_2, color_2, shape_2 = text.strip().split(" ")
    sampled_key = random.choices(list(real_block_sent.keys()), weights=list(real_block_sent.values()))[0]
    if shape_1 in real_block_word:
        shape_1 = random.choices(list(real_block_word[shape_1].keys()), weights=list(real_block_word[shape_1].values()))[0]
    if shape_2 in real_block_word:
        shape_2 = random.choices(list(real_block_word[shape_2].keys()), weights=list(real_block_word[shape_2].values()))[0]
    object_name = " ".join((size_1, color_1, shape_1))
    container_name = " ".join((size_2, color_2, shape_2))
    new_text = sampled_key.format(object_name, container_name)
    return new_text

