import os
import glob
from pathlib import Path
import numpy as np
from PIL import Image
import random
import torch
import torchvision.transforms
from torchvision.transforms import Compose, Normalize
from torchvision.transforms import v2
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

# Depth To Image Stuff  ----------------------------------------------------------
from matplotlib.cm import viridis

def depth_to_norm(depth_image, depth_min=0, depth_max=1023):
    depth_norm = (np.clip(depth_image, depth_min, depth_max) - depth_min ) / (depth_max-depth_min)
    return depth_norm

def norm_to_depth(norm_image, depth_min=0, depth_max=1023):
    depth = norm_image * (depth_max-depth_min) + depth_min
    return depth

class ViridisToNorm:
    def __init__(self):
        viridis_values = np.linspace(0, 1, len(viridis.colors))
        # Create LUT dictionary mapping RGB -> value
        self.viridis_lut = {tuple(rgb[:3]): value for rgb, value in zip(viridis.colors, viridis_values)}  # Disel: only the brave.
        #self.viridis_lut = {rgb_to_key(rgb[:3]): value for rgb, value in zip(viridis.colors, viridis_values)}
        
    def __call__(self, array):
        if isinstance(array, np.ndarray):
            old_shape = array.shape
            recovered_flat = np.array([self.viridis_lut.get(tuple(color), 0) for color in array.reshape(-1, 3)])
            return recovered_flat.reshape(old_shape[:-1])
        elif isinstance(array, tuple):
            return self.viridis_lut.get(array, 0)
        else:
            raise ValueError
        
    def color_to_depth(self, depth_as_color):
        assert depth_as_color.ndim == 3
        assert depth_as_color.shape[2] == 3
            
        norm_image = self.__call__(depth_as_color)
        depth_image = norm_to_depth(norm_image)
        return depth_image
    
    # def rgb_to_key(rgb, precision=3):
    #         return tuple(np.round(rgb, precision))  # Round for consistent lookup

color_to_norm = ViridisToNorm()
norm_to_color = lambda norm: viridis(norm)[:,:, :3] ## should take (w,h) not (w,h,1)

color_to_depth = color_to_norm.color_to_depth
def depth_to_color(depth_image):
    """
    Arguments:
        depth_image: in [mm]
    """
    assert depth_image.ndim == 2
    # Normalize the data to the range [0, 1]
    depth_norm = depth_to_norm(depth_image=depth_image)
    depth_rgb = viridis(depth_norm)[:, :, :3]
    return depth_rgb

def test_norm_color_pingpoing():
    random_array = np.random.rand(4, 4)
    viridis_mapped = norm_to_color(random_array)
    recovered_array = color_to_norm(viridis_mapped)
    assert np.all(np.isclose(random_array, recovered_array, atol=.01))


# Depth Augmentation --------------------------------------------------------
class DepthAugmentation:
    def __init__(self, depth_range=(25,100), max_delta_depth=35):
        self.depth_range = depth_range
        self.delta_depth_max = max_delta_depth
        
    def __call__(self, depth_image_mm, suffix):
        """
        Arguments:
            depth_image_mm: depth image in [mm]
            suffix: encoded trajectory
        """
        loc_strings = [int(x) for x in re.findall(r"<(?:loc|seg)(\d+)>", suffix)]
        loc_strings = np.array(loc_strings)
        try:
            loc_strings = loc_strings.reshape(-1, 6)
        except ValueError as e:
            print(suffix, loc_strings)
            raise e
        depth_vals_cm = loc_strings[:,2]
        assert np.all(depth_vals_cm > 0)

        # Compute effective bounds for delta_depth_cm
        min_depth, max_depth = self.depth_range
        lower_bound = np.maximum(min_depth - min(depth_vals_cm), -self.delta_depth_max)
        upper_bound = np.minimum(max_depth - max(depth_vals_cm), self.delta_depth_max)
        
        # Ensure lower_bound does not exceed upper_bound and sample
        lower_bound = np.minimum(lower_bound, upper_bound)
        delta_depth_cm = np.round(np.random.uniform(lower_bound, upper_bound))
        
        # Compute new depth values
        new_depth_vals_cm = depth_vals_cm + delta_depth_cm
        new_depth_vals_cm = new_depth_vals_cm.astype(np.uint16)
        
        # Validate all conditions
        assert np.all(np.abs(delta_depth_cm) <= self.delta_depth_max), "Delta depth exceeds max limit!"
        assert np.min(new_depth_vals_cm) >= min_depth, "Depth below min range!"
        assert np.max(new_depth_vals_cm) <= max_depth, "Depth above max range!"
        
        # Calculate new depth values
        loc_strings[:, 2] = new_depth_vals_cm
        suffix_new = ""
        for x in loc_strings:
            suffix_new += "<loc{a[0]:04d}><loc{a[1]:04d}><loc{a[2]:04d}><seg{a[3]:03d}><seg{a[4]:03d}><seg{a[5]:03d}>".format(a=x)
        depth_image_aug = depth_image_mm.copy()
        depth_image_aug[depth_image_aug != 0] += (delta_depth_cm*10).astype(np.uint16)
        
        return depth_image_aug, suffix_new

# Text Augmentations --------------------------------------------------------

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

real_block_word = {'cube': {'block': 118, 'cube': 21, 'box': 2},'sphere': {'sphere': 50, 'ball': 50}}

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



# Crop ----------------------------------------------------------

def obj_start_crop(image: Image, suffix: str, crop_size=None, object_size=None):
    """
    Arguments:
        image: PIL.Image (h w c)
        suffix (12) = obj_start (h w d rv0 rv1 rv2), obj_end (h w d rv0 rv1 rv2)
        crop_size: int, total width of crop
        object_size: unsused

    Crop image around object start position,
    shift coor center to crop center and set crop size as maximum coordinate.
    """

    image_width, image_height = image.size[:2]

    # convert string tokens to int
    # depth in cm, x and y in [0, 1023]
    loc_strings = [int(x) for x in re.findall(r"<(?:loc|seg)(-?\d+)>", suffix)]
    loc_strings = np.array(loc_strings)
    loc_strings = loc_strings.reshape(-1, 6) # each row contains (h w d r0 r1 r2)

    # convert from maniskill (1024, 1024) to (image_height, image_width)
    loc_h = (loc_strings[:, 0]/(1024-1)*image_height).round().astype(int) # (obj_start_x obj_end_x)
    loc_w = (loc_strings[:, 1]/(1024-1)*image_width).round().astype(int)

    # Image crop at obj_start
    top = loc_h[0]-crop_size//2
    left = loc_w[0]-crop_size//2

    # === valid crop
    top = max(0, top)
    left = max(0, left)
    crop_h = min(image_height - top, crop_size)
    crop_w = min(image_width - left, crop_size)
    box = [top, left, crop_h, crop_w] # (y, x, h, w)
    crop = v2.functional.crop(image, *box)

    # add delta to y x, go from (image_height, image_width) to maniskill (1024, 1024)
    loc_strings[:, 0] = ((loc_h - top)/crop_h*(1024-1)).round().astype(int)
    loc_strings[:, 1] = ((loc_w - left)/crop_w*(1024-1)).round().astype(int)

    suffix_new = ""
    for x in loc_strings:
        suffix_new += "<loc{a[0]:04d}><loc{a[1]:04d}><loc{a[2]:04d}><seg{a[3]:03d}><seg{a[4]:03d}><seg{a[5]:03d}>".format(a=x)

    return crop, suffix_new

def crop_start_end(image: Image, suffix: str, crop_size=None, object_size=None):
    """
    Arguments:
        image: PIL.Image (h w c)
        suffix (12) = obj_start (h w d rv0 rv1 rv2), obj_end (h w d rv0 rv1 rv2)
        crop_size: int, total width of crop
        object_size: int, size of start and end objects in pixels

    Crop at center between start and end objects.
    Shift coor center to crop center and set crop size as maximum coordinate.
    """

    image_width, image_height = image.size[:2]

    # convert string tokens to int
    # depth in cm, x and y in [0, 1023]
    loc_strings = [int(x) for x in re.findall(r"<(?:loc|seg)(-?\d+)>", suffix)]
    loc_strings = np.array(loc_strings)
    loc_strings = loc_strings.reshape(-1, 6) # each row contains (h w d r0 r1 r2)

    # convert from maniskill (1024, 1024) to (image_height, image_width)
    loc_h = (loc_strings[:, 0]/(1024-1)*image_height).round().astype(int) # (obj_start_x obj_end_x)
    loc_w = (loc_strings[:, 1]/(1024-1)*image_width).round().astype(int)

    # Image crop at middle between start and end
    center_h = np.mean(loc_h)
    center_w = np.mean(loc_w)
    min_crop_h = abs(loc_h[0] - loc_h[1]) + object_size
    min_crop_w = abs(loc_w[0] - loc_w[1]) + object_size

    # Make both start and end objects visible
    crop_h = max(min_crop_h, crop_size)
    crop_w = max(min_crop_w, crop_size)
    top = center_h - crop_h // 2
    left = center_w - crop_w // 2
    
    box = (top, left, crop_h, crop_w) # yxhw
    crop = v2.functional.crop(image, *box)

    # add delta to y x, go from (image_height, image_width) to maniskill (1024, 1024)
    loc_strings[:, 0] = ((loc_h - top)/crop_h*(1024-1)).round().astype(int)
    loc_strings[:, 1] = ((loc_w - left)/crop_w*(1024-1)).round().astype(int)

    suffix_new = ""
    for x in loc_strings:
        suffix_new += "<loc{a[0]:04d}><loc{a[1]:04d}><loc{a[2]:04d}><seg{a[3]:03d}><seg{a[4]:03d}><seg{a[5]:03d}>".format(a=x)

    return crop, suffix_new

def test_obj_start_crop():
    """
    Test:
    1. np array (10, 10) -> PIL image
    2. define suffix string
    3. check crop size, crop contents, new suffix
    """

    a = np.zeros((10, 10))
    a[:5, :5] = 1
    image = Image.fromarray(a)
    suffix = "<loc0500><loc0><loc0><seg0><seg0><seg0>  <loc0800><loc0400><loc0><seg0><seg0><seg0>" 
    # begin = (5, 0)
    # end = (8, 4)
    # ..vvv |
    # ..vvv |
    # ..B.. |
    # ..... |
    # ..... |
    # ------E
    crop, suffix = obj_start_crop(image, suffix, 5)
    assert crop.size == (5, 5)
    assert np.allclose(crop, np.array([
        [0., 0., 1., 1., 1.],
        [0., 0., 1., 1., 1.],
        [0., 0., 0., 0., 0.], # y=5, x=0
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
    ]))

    loc_strings = [int(x) for x in re.findall(r"<(?:loc|seg)(\d+)>", suffix)]
    loc_strings = np.array(loc_strings)
    loc_strings = loc_strings.reshape(-1, 6)

    # go from maniskill (1024, 1024) to ((0, 1), (0, 1))
    loc_h = loc_strings[:, 0]/(1024-1) # (obj_start_x obj_end_x)
    loc_w = loc_strings[:, 1]/(1024-1)

    assert np.allclose(loc_h, np.array([0.5, 1.0]), atol=0.2) # (start_y, end_y)
    assert np.allclose(loc_w, np.array([0.5, 1.2]), atol=0.2) # (start_x, end_x)


if __name__ == "__main__":
    test_norm_color_pingpoing()
    test_obj_start_crop()
