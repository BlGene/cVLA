import h5py
import torch
import numpy as np
from matplotlib.cm import viridis
from mani_skill.utils.structs import Pose
from mani_skill.examples.utils_traj_tokens import to_prefix_suffix
from mani_skill.examples.utils_traj_tokens import getActionEncDecFunction
from utils_trajectory import DummyCamera
from torch.utils.data import Dataset

enc_func, dec_func = getActionEncDecFunction("xyzrotvec-cam-proj2")


def depth_as_color(depth_image, depth_min=0, depth_max=1023):
    # Normalize the data to the range [0, 1]
    assert depth_image.ndim == 3
    assert depth_image.shape[2] == 1
    depth_norm = (np.clip(depth_image[:,:,0], depth_min, depth_max) - depth_min ) / (depth_max-depth_min)
    depth_rgb = viridis(depth_norm)[:, :, :3]
    return depth_rgb
    
class H5Dataset(Dataset):
    def __init__(self, h5_file_path, return_depth=False):
        self.h5_file = h5py.File(h5_file_path, "r")
        self.return_depth = return_depth
        
    def __len__(self):
        return len(self.h5_file)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        
        image = self.h5_file[f'traj_{idx}/obs/sensor_data/render_camera/rgb'][0]
        
        action_text = None
        for x in self.h5_file[f'traj_{idx}/obs/extra'].keys():
            if x.startswith("action_text_"):
                action_text = str(x).replace("action_text_", "")
            
        obj_start = Pose(torch.tensor(self.h5_file[f"traj_{idx}/obs/extra/obj_start"]))
        obj_end = Pose(torch.tensor(self.h5_file[f"traj_{idx}/obs/extra/obj_end"]))
        grasp_pose = Pose(torch.tensor(self.h5_file[f"traj_{idx}/obs/extra/grasp_pose"]))
        tcp_pose = Pose(torch.tensor(self.h5_file[f"traj_{idx}/obs/extra/tcp_pose"]))
        camera_intrinsic = self.h5_file[f"traj_{idx}/obs/sensor_param/render_camera/intrinsic_cv"]
        camera_extrinsic = self.h5_file[f"traj_{idx}/obs/sensor_param/render_camera/extrinsic_cv"]
        width, height, c = image.shape
        camera = DummyCamera(camera_intrinsic, camera_extrinsic, width, height)
        prefix, token_str, curve_3d, orns_3d, info = to_prefix_suffix(obj_start, obj_end,
                                                                       camera, grasp_pose, tcp_pose,
                                                                       action_text, enc_func, robot_pose=None)
        entry = dict(prefix=prefix, suffix=token_str, camera=camera)

        if self.return_depth:
            depth = self.h5_file[f'traj_{idx}/obs/sensor_data/render_camera/depth'][0]
            depth_image = depth_as_color(depth)
            return [image, depth_image], entry
        
        return image, entry
