import glob
import h5py
from pathlib import Path
import torch
from PIL import Image
from mani_skill.utils.structs import Pose
from mani_skill.examples.utils_traj_tokens import to_prefix_suffix
from mani_skill.examples.utils_traj_tokens import getActionEncDecFunction
from utils_trajectory import DummyCamera
from torch.utils.data import Dataset
from data_augmentations import depth_to_color

enc_func, dec_func = getActionEncDecFunction("xyzrotvec-cam-proj2")


class H5Dataset(Dataset):
    def __init__(self, h5_file_or_dir, return_depth=False, augment_rgbds=None, augment_rgb=None, augment_text=None, depth_to_color=True):
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
        
        self.h5_file_len = len(self.h5_file)

        self.augment_rgb = augment_rgb
        self.augment_rgbds = augment_rgbds
        self.augment_text = augment_text
        self.depth_to_color = depth_to_color

    def __len__(self):
        return self.h5_file_len

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        
        action_text = None
        for x in self.h5_file[f'traj_{idx}/obs/extra'].keys():
            if x.startswith("action_text_"):
                action_text = str(x).replace("action_text_", "")
            
        obj_start = Pose(torch.tensor(self.h5_file[f"traj_{idx}/obs/extra/obj_start"][:]))
        obj_end = Pose(torch.tensor(self.h5_file[f"traj_{idx}/obs/extra/obj_end"][:]))
        grasp_pose = Pose(torch.tensor(self.h5_file[f"traj_{idx}/obs/extra/grasp_pose"][:]))
        tcp_pose = Pose(torch.tensor(self.h5_file[f"traj_{idx}/obs/extra/tcp_pose"][:]))
        camera_intrinsic = self.h5_file[f"traj_{idx}/obs/sensor_param/render_camera/intrinsic_cv"][:]
        camera_extrinsic = self.h5_file[f"traj_{idx}/obs/sensor_param/render_camera/extrinsic_cv"][:]
        
        image = self.h5_file[f'traj_{idx}/obs/sensor_data/render_camera/rgb'][0]
        width, height, c = image.shape
        camera = DummyCamera(camera_intrinsic, camera_extrinsic, width, height)

        if self.augment_text is not None:
            action_text = self.augment_text(action_text)

        prefix, token_str, curve_3d, orns_3d, info = to_prefix_suffix(obj_start, obj_end,
                                                                       camera, grasp_pose, tcp_pose,
                                                                       action_text, enc_func, robot_pose=None)
        entry = dict(prefix=prefix, suffix=token_str, camera=camera)

        depth = None
        seg = None
        if self.augment_rgbds is not None:
            depth = self.h5_file[f'traj_{idx}/obs/sensor_data/render_camera/depth'][0]
            seg = self.h5_file[f'traj_{idx}/obs/sensor_data/render_camera/segmentation'][0]
            image = self.augment_rgbds(image, depth, seg)

        if self.augment_rgb is not None:
            image = self.augment_rgb(image)

        if self.return_depth:
            if depth is None:
                depth = self.h5_file[f'traj_{idx}/obs/sensor_data/render_camera/depth'][0]
            if self.depth_to_color:
                depth = depth_to_color(depth)
            else:
                depth = depth[:,:,0]
            return [depth, image], entry
        
        return Image.fromarray(image), entry
