# PaliGemma tokenize vocabulary with 1024 entries that represent coordinates in normalized image-space (<loc0000>...<loc1023>)
import numpy as np
import torch
import re
from pdb import set_trace
from scipy.spatial.transform import Rotation as R
from mani_skill.utils.structs import Pose

from cvla.utils_trajectory import project_points, clip_and_interpolate
from cvla.utils_trajectory import unproject_points
from cvla.utils_trajectory import generate_curve_torch

def normalize_imgcoord(traj, resolution_wh, token_num=1024):
    """
    Normalize (xy) pixel positions into (0, norm_max-1) following paligemma detection
    Arguments:
        traj: array with shape (P waypoints, 2) with waypoints in (width, height) px
    Returns:
        traj_norm: array with shape (P waypoints, 2) with waypoints normalized [0,1]
    """
    assert traj.shape[1] == 2
    w, h = resolution_wh
    # Convert from resolution_wh back to 1024x1024 space
    traj_norm = traj / np.array([[w, h]]) * (token_num - 1)
    traj_norm = traj_norm.round().int()
    if torch.any(traj_norm != torch.clamp(traj_norm, 0, token_num - 1)):
        print("Warning: tokens out of range.")
        # Construct the result string using the original pattern
    return traj_norm


def to_prefix_suffix(obj_start, obj_end, camera, grasp_pose, tcp_pose, action_text, enc_func, robot_pose=None, num_points=2):
    """
    This function generates a trajectory that moves the object from obj_start to obj_end,
    assuming we grasp the object at obj_start with grasp_pose. It keeps the same relative position
    between object and grasp pose.
    """
    batch_size = 1
    grasp_start = grasp_pose
    grasp_end = obj_end * obj_start.inv() * grasp_pose  # keep same object->grasp relation
    _, curve_3d = generate_curve_torch(grasp_start.get_p(), grasp_end.get_p(), num_points=2)
    #_, curve_3d = generate_curve_torch(obj_start.get_p(), obj_end.get_p(), num_points=2)
    assert curve_3d.shape == (batch_size, num_points, 3)
    # Always keep rotation at grap pose
    #orns_3d = grasp_start.get_q().clone().detach()  # get rotation
    #orns_3d = orns_3d.expand(curve_3d.shape[0], curve_3d.shape[1], -1)
    if num_points != 2:
        # What you probably want to do here is interpolate between orns.
        raise NotImplementedError("See code comment above for tip here.")
    orns_3d = torch.stack((grasp_start.get_q().clone().detach(),grasp_end.get_q().clone().detach()),axis=1)
    assert orns_3d.shape == (batch_size, num_points, 4)

    #from pdb import set_trace
    #set_trace()
    #print(orns_3d.shape)
    curve_25d, depth, token_str, didclip_traj = enc_func(curve_3d, orns_3d, camera, robot_pose=robot_pose, return_didclip=True)
    # encode tcp position in prompt (prefix)
    _, _, tcp_str, didclip_tcp = enc_func(tcp_pose.get_p().unsqueeze(0), tcp_pose.get_q().unsqueeze(0), camera, robot_pose=robot_pose, return_didclip=True)   
    prefix = action_text+" "+tcp_str
    info = dict(didclip_traj=didclip_traj, didclip_tcp=didclip_tcp, robot_state=tcp_str, action_text=action_text)
    return prefix, token_str, curve_3d, orns_3d, info


#-----------------------------------------------------------------------------

class TrajectoryEncoder_rbt100:
    num_tokens = 6
    DEPTH_SCALE = 100
    ROT_SCALE = (128-1) / (2*np.pi)
    POS_OFFSET = 0.5
    ROT_TOKEN = "<seg{x:03d}>"
    NAME = "xyzrotvec-rbt-100"
    DEPTH_TOKENS_START = 0
    DEPTH_TOKENS_END = 1024

    def encode_trajectory(self, curve_3d, orns_3d, camera, robot_pose=None, return_didclip=False):
        """
        Trajectory encoded as y, x, z with x, y being pixel positions in range (0, 1023) and z being depth in cm
        """
        # In theory this code should handle (N, 7) poses, but for now we only support single envs

        # transform orientation to matrix
        assert curve_3d.ndim == 3 and orns_3d.ndim == 3
        assert curve_3d.shape[:2] == orns_3d.shape[:2]
        assert curve_3d.shape[2] == 3 and orns_3d.shape[2] == 4
        assert robot_pose is not None

        poses = Pose.create_from_pq(p=curve_3d.view(-1, 3), q=orns_3d.view(-1, 4))
        poses_r = robot_pose.inv() * poses
        orn_r_obj = R.from_quat(poses_r.get_q(), scalar_first=True)
        rotvec = torch.tensor(orn_r_obj.as_rotvec())
        rotvec_positive = rotvec + torch.tensor([np.pi,]*3)
        is_ok = (rotvec_positive > 0).all() and (rotvec_positive < 2*np.pi).all()
        if not is_ok:
            print("Warning: Rotvec out of range.")
        xyz_positive = poses_r.get_p() + self.POS_OFFSET
        is_ok = (xyz_positive > 0).all() and (xyz_positive*self.DEPTH_SCALE < 1024).all()
        if not is_ok:
            print("Warning: xyz position out of range.")

        # This is the part that is not parrelized
        env_idx = 0
        traj_1024_o = (xyz_positive*self.DEPTH_SCALE).round().int()  # distance from camera in [cm]
        rotvec_int = (rotvec_positive*self.ROT_SCALE).round().int()
        traj_1024 = np.clip(traj_1024_o, self.DEPTH_TOKENS_START, self.DEPTH_TOKENS_END)
        if not torch.allclose(traj_1024,traj_1024_o):
            print("Warning: depth out of range.", traj_1024)

        result = ""
        for keypoint_xyz, rv in zip(traj_1024, rotvec_int):
            result += f"<loc{keypoint_xyz[1]:04d}><loc{keypoint_xyz[0]:04d}><loc{keypoint_xyz[2]:04d}>"
            result += self.ROT_TOKEN.format(x=rv[0]) + self.ROT_TOKEN.format(x=rv[1]) + self.ROT_TOKEN.format(x=rv[2])
        result = result.strip()  # Remove any trailing newlines
        if return_didclip:
            return None, None, result, False
        return None, None, result

    def decode_caption(self, caption, camera=None, robot_pose=None):
        """
        Takes a trajectory string and converts it into curve_25d, quat_c
        """
        num_tokens = self.num_tokens
        # Pattern to extract numbers inside <loc####> tags
        #loc_strings = re.findall(r"<loc(\d{4})>", caption)
        loc_strings = re.findall(r"<(?:loc|seg)(\d+)>", caption)
        num_position_tokens = len(loc_strings)
        loc_strings_pairs = loc_strings[:(num_position_tokens//num_tokens)*num_tokens]
        loc_numbers = [int(x) for x in loc_strings_pairs]
        loc_x = [x/self.DEPTH_SCALE for x in loc_numbers[::num_tokens]]
        loc_y = [x/self.DEPTH_SCALE for x in loc_numbers[1::num_tokens]]
        loc_z = [x/self.DEPTH_SCALE for x in loc_numbers[2::num_tokens]]  # depth
        loc_r0 = [x/self.ROT_SCALE for x in loc_numbers[3::num_tokens]]  # rotvec[0]
        loc_r1 = [x/self.ROT_SCALE for x in loc_numbers[4::num_tokens]]  # rotvec[1]
        loc_r2 = [x/self.ROT_SCALE for x in loc_numbers[5::num_tokens]]  # rotvec[2]
        
        xyz_positive = torch.tensor((loc_y, loc_x, loc_z)).T
        rotvec_positive = torch.tensor((loc_r0, loc_r1, loc_r2)).T
        pos_r = xyz_positive - self.POS_OFFSET
        rotvec = rotvec_positive - torch.tensor([np.pi,]*3)
        quat_r = torch.tensor(R.from_rotvec(rotvec).as_quat(scalar_first=True)).float()
        return pos_r, quat_r

    def decode_trajectory(self, caption, camera=None, robot_pose=None):
        """
        Takes a caption string and converts it to world coordinates
        """
        assert isinstance(robot_pose, Pose)
        pos_r, quat_r = self.decode_caption(caption, camera=None)
        pose_r = Pose.create_from_pq(p=pos_r, q=quat_r)    
        poses = robot_pose * pose_r
        return poses.get_p().unsqueeze(0), poses.get_q().unsqueeze(0)


#-----------------------------------------------------------------------------

class TrajectoryEncoder_xyzrotvec_1024xy:
    num_tokens = 6
    ROT_SCALE = (128-1) / (2*np.pi)
    XY_TOKENS = 1024  # as in 1024 tokens in total going from 0 to 1023
    DEPTH_SCALE = 100  # set depth scale or (range and tokens) not both!
    DEPTH_RANGE = None # e.g. (.1, 1.0) in [meters], also set depth tokens
    DEPTH_TOKENS = -1  # the nunbers of tokens the depth range should be divided into
    DEPTH_TOKENS_START = 0
    DEPTH_TOKENS_END = 1024
    NAME = "xyzrotvec-cam-1024xy"
    ENCODING_STR = "<loc{kp_xy[1]:04d}><loc{kp_xy[0]:04d}><loc{kp_d:04d}><seg{kp_r[0]:03d}><seg{kp_r[1]:03d}><seg{kp_r[2]:03d}>"

    def __init__(self):
        if self.DEPTH_SCALE:
            assert self.DEPTH_RANGE is None and self.DEPTH_TOKENS == -1
        if self.DEPTH_RANGE:
            assert self.DEPTH_SCALE is None
            assert self.DEPTH_TOKENS > 0 and self.DEPTH_TOKENS < 1024

    def encode_pixel_coords(self, curve_3d, orns_3d, camera, robot_pose=None, return_didclip=False):
        """
        Encode a curve to pixel coordinates
        Arguments:
            curve_3d: shape (N, P, 3) position in [m]
            orns_3d: shape (N, P, 4) pose in quaternion, scalar first

        Returns:
            curve_2d_short: shape (N, P, 3)
            poses_c_quat
            didclip
        """
        assert curve_3d.ndim == 3 and curve_3d.shape[2] == 3
        assert orns_3d.ndim == 3 and orns_3d.shape[2] == 4
        assert curve_3d.shape[:2] == orns_3d.shape[:2]
        assert curve_3d.shape[0] == 1  # for now

        curve_25d = project_points(camera, curve_3d, return_depth=True)
        #curve_2d = curve_25d[..., :2]
        #depth = curve_25d[..., 2]
        curve_25d_short, didclip = clip_and_interpolate(curve_25d, camera, return_didclip=True)
        #print(curve_25d.shape, curve_2d_short.shape, didclip)

        # transform orientation to matrix
        assert curve_3d.shape[:2] == orns_3d.shape[:2]
        assert curve_3d.ndim == 3 and orns_3d.ndim == 3
        assert curve_3d.shape[2] == 3 and orns_3d.shape[2] == 4

        poses = Pose.create_from_pq(p=curve_3d.view(-1, 3), q=orns_3d.view(-1, 4))
        extrinsic_orn = R.from_matrix(camera.get_extrinsic_matrix()[:, :3, :3])
        extrinsic_p = Pose.create_from_pq(p=camera.get_extrinsic_matrix()[:, :3, 3],
                                        q=extrinsic_orn.as_quat(scalar_first=True))
        poses_c = extrinsic_p * poses
        poses_c_quat = R.from_quat(poses_c.get_q(), scalar_first=True)

        return curve_25d_short, poses_c_quat, didclip

    def to_string(self, traj_norm, depth_env, rotvec_int):
        """
        PaliGemma extra special token encoder.
        """
        result = ""
        for kp_xy, kp_d, kp_r in zip(traj_norm, depth_env, rotvec_int):
            result += f"<loc{kp_xy[1]:04d}><loc{kp_xy[0]:04d}><loc{kp_d:04d}><seg{kp_r[0]:03d}><seg{kp_r[1]:03d}><seg{kp_r[2]:03d}>"
        result = result.strip()  # Remove any trailing newlines
        return result

    def encode_pixel_tokens(self, curve_25d_short, poses_c_quat, camera):
        """
        Arguments:
            curve_25d_short: shape (N, P, 3) tensor
            poses_c_quat: shape (N * P) sklear Rotation
        """
        assert curve_25d_short.ndim == 3
        assert len(poses_c_quat) == curve_25d_short.shape[0] * curve_25d_short.shape[1]
        assert curve_25d_short.shape[0] == 1  # for now

        rotvec = torch.tensor(poses_c_quat.as_rotvec())
        rotvec_positive = rotvec + torch.tensor([np.pi,]*3)
        is_ok = (rotvec_positive > 0).all() and (rotvec_positive < 2*np.pi).all()
        if not is_ok:
            print("Warning: Rotvec out of range.")

        # This is the part that is not parrelized
        depth = curve_25d_short[:, :, 2]
        env_idx = 0
        if self.DEPTH_SCALE:
            depth_int = (depth[env_idx]*self.DEPTH_SCALE).round().int()  # distance from camera in [cm]
        elif self.DEPTH_RANGE:
            depth_env = np.clip(depth[env_idx], self.DEPTH_RANGE[0], self.DEPTH_RANGE[1])
            depth_norm = (depth_env-self.DEPTH_RANGE[0]) / (self.DEPTH_RANGE[1]-self.DEPTH_RANGE[0])
            depth_int = (depth_norm * self.DEPTH_TOKENS).round().int()
        else:
            raise ValueError("No depth scale or range provided.")
        
        depth_env_o = depth_int + self.DEPTH_TOKENS_START
        depth_env = np.clip(depth_env_o, self.DEPTH_TOKENS_START, self.DEPTH_TOKENS_END)
        if not torch.allclose(depth_env,depth_env_o):
            print("Warning: depth out of range.", depth_env)
        
        traj_norm = normalize_imgcoord(curve_25d_short[env_idx][:,:2], (camera.width, camera.height), token_num=self.XY_TOKENS)
        rotvec_int = (rotvec_positive * self.ROT_SCALE).round().int()
        result = self.to_string(traj_norm, depth_env, rotvec_int)
        # result = ""
        # for kp_xy, kp_d, kp_r in zip(traj_norm, depth_env, rotvec_int):
        #     result += f"<loc{kp_xy[1]:04d}><loc{kp_xy[0]:04d}><loc{kp_d:04d}><seg{kp_r[0]:03d}><seg{kp_r[1]:03d}><seg{kp_r[2]:03d}>"
        # result = result.strip()  # Remove any trailing newlines
        return result

    def encode_trajectory(self, curve_3d, orns_3d, camera, robot_pose=None, return_didclip=False):
        """
        Encode a curve to pixel coordinates
        Arguments:
            curve_3d: shape (N, P, 3) position in [m]
            orns_3d: shape (N, P, 4) pose in quaternion, scalar first

        Returns:
            curve_2d_short: shape (N, P, 3)
            poses_c_quat
            didclip
        """
        assert curve_3d.ndim == 3 and curve_3d.shape[2] == 3
        assert orns_3d.ndim == 3 and orns_3d.shape[2] == 4
        assert curve_3d.shape[:2] == orns_3d.shape[:2]
        assert curve_3d.shape[0] == 1  # for now

        # In theory this code should handle (N, 7) poses, but for now we only support single envs
        curve_2d_short, poses_c, didclip = self.encode_pixel_coords(curve_3d, orns_3d, camera, robot_pose=robot_pose)
        depth = curve_2d_short[:,:,2]
        result = self.encode_pixel_tokens(curve_2d_short, poses_c, camera)
        if return_didclip:
            return curve_2d_short, depth, result, didclip 
        return curve_2d_short, depth, result

    def from_string(self, caption):
        loc_strings = re.findall(r"<(?:loc|seg)(\d+)>", caption)
        num_position_tokens = len(loc_strings)
        loc_strings_pairs = loc_strings[:(num_position_tokens//self.num_tokens)*self.num_tokens]
        loc_numbers = [int(x) for x in loc_strings_pairs]
        return loc_numbers

    def decode_caption(self, caption: str, camera=None, robot_pose=None):
        """
        Takes a trajectory string and converts it into curve_25d, quat_c
        
        Arguments:
            caption: see encode_trajectory

        Returns:
            curve_3d: position in [px, px, m] in camera coordinates shape (P, 3)
            orns_3d: pose in quaternion, scalar first in camera coordinates, shape (P, 4)
        """
        assert camera is not None
        # Pattern to extract numbers inside <loc####> tags
        #loc_strings = re.findall(r"(<loc(\d{4})>|<seg(\d{3})>)", caption)
        # loc_strings = re.findall(r"<(?:loc|seg)(\d+)>", caption)
        # num_position_tokens = len(loc_strings)
        # loc_strings_pairs = loc_strings[:(num_position_tokens//self.num_tokens)*self.num_tokens]
        # loc_numbers = [int(x) for x in loc_strings_pairs]
        loc_numbers = self.from_string(caption)
        loc_h = [x/(self.XY_TOKENS-1)*camera.height for x in loc_numbers[::self.num_tokens]]
        loc_w = [x/(self.XY_TOKENS-1)*camera.width for x in loc_numbers[1::self.num_tokens]]
        loc_d = []
        for x in loc_numbers[2::self.num_tokens]:
            x_c = np.clip(x, self.DEPTH_TOKENS_START, self.DEPTH_TOKENS_END)
            if x_c != x:
                print(f"Warning: depth token was {x} range was [{self.DEPTH_TOKENS_START}, {self.DEPTH_TOKENS_END}]")

            depth_int = x_c - self.DEPTH_TOKENS_START
            if self.DEPTH_SCALE:
                depth = depth_int / self.DEPTH_SCALE  # Undo scaling
            elif self.DEPTH_RANGE:
                depth_norm = depth_int / self.DEPTH_TOKENS
                depth = depth_norm * (self.DEPTH_RANGE[1] - self.DEPTH_RANGE[0]) + self.DEPTH_RANGE[0]
            else:
                raise ValueError("No depth scale or range provided.")
            loc_d.append(float(depth))
        #loc_d = [x/self.DEPTH_SCALE for x in loc_numbers[2::self.num_tokens]]  # depth
        loc_r0 = [x/self.ROT_SCALE for x in loc_numbers[3::self.num_tokens]]  # rotvec[0]
        loc_r1 = [x/self.ROT_SCALE for x in loc_numbers[4::self.num_tokens]]  # rotvec[1]
        loc_r2 = [x/self.ROT_SCALE for x in loc_numbers[5::self.num_tokens]]  # rotvec[2]
        
        curve_25d = torch.tensor((loc_w, loc_h, loc_d)).T
        rotvec_positive = torch.tensor((loc_r0, loc_r1, loc_r2)).T
        rotvec = rotvec_positive - torch.tensor([np.pi,]*3)
        try:
            quat_c = torch.tensor(R.from_rotvec(rotvec).as_quat(scalar_first=True)).float()
        except ValueError as e:
            raise ValueError(f"Failed to decode caption {caption}")
        
        return curve_25d, quat_c

    def decode_trajectory(self, caption: str, camera=None, robot_pose=None):
        """
        Takes a caption string and converts it to world coordinates
        Returns:
            curve_w in world/robot coordinates, shape (P, 3)
            quat_w in world/robot coordinates, shape (P, 4)
        """
        assert camera is not None
        curve_25d, quat_c = self.decode_caption(caption, camera)
        # from camera to world coordinates
        extrinsic_orn = R.from_matrix(camera.get_extrinsic_matrix()[:, :3, :3])
        extrinsic = Pose.create_from_pq(p=camera.get_extrinsic_matrix()[:, :3, 3],
                                        q=extrinsic_orn.as_quat(scalar_first=True))
        quat_w = extrinsic.inv() * Pose.create_from_pq(q=quat_c)
        curve_w = unproject_points(camera, curve_25d) 

        return curve_w, quat_w.get_q().unsqueeze(0)  # shape (P, 3 = u, v, d)

class TrajectoryEncoder_rbt100loc(TrajectoryEncoder_rbt100):
    DEPTH_TOKEN = "loc"
    NAME = "xyzrotvec-rbt-128"

class TrajectoryEncoder_rbt128(TrajectoryEncoder_rbt100):
    DEPTH_SCALE = 128
    NAME = "xyzrotvec-rbt-128"

class TrajectoryEncoder_rbt256(TrajectoryEncoder_rbt100):
    DEPTH_SCALE = 256
    NAME = "xyzrotvec-rbt-256"

class TrajectoryEncoder_rbt512(TrajectoryEncoder_rbt100):
    DEPTH_SCALE = 512
    NAME = "xyzrotvec-rbt-512"

class TrajectoryEncoder_rbt1024(TrajectoryEncoder_rbt100):
    DEPTH_SCALE = 1024
    NAME = "xyzrotvec-rbt-1024"


class TrajectoryEncoder_xyzrotvec_512xy(TrajectoryEncoder_xyzrotvec_1024xy):
    XY_TOKENS = 512  # as in 1024 tokens in total going from 0 to 511
    NAME = "xyzrotvec-cam-512xy"

class TrajectoryEncoder_xyzrotvec_256xy(TrajectoryEncoder_xyzrotvec_1024xy):
    XY_TOKENS = 256  # as in 1024 tokens in total going from 0 to 255
    NAME = "xyzrotvec-cam-256xy"

class TrajectoryEncoder_xyzrotvec_128xy(TrajectoryEncoder_xyzrotvec_1024xy):
    XY_TOKENS = 128  # as in 1024 tokens in total going from 0 to 255
    NAME = "xyzrotvec-cam-128xy"

class TrajectoryEncoder_xyzrotvec_512xy128d(TrajectoryEncoder_xyzrotvec_1024xy):
    XY_TOKENS = 512  # as in 1024 tokens in total going from 0 to 511
    DEPTH_SCALE = None
    DEPTH_RANGE = (.1, 1.0)  # [meters]
    DEPTH_TOKENS = 128
    remaining_tokens = 1024 - XY_TOKENS - DEPTH_TOKENS
    DEPTH_TOKENS_START = XY_TOKENS + (remaining_tokens)//2  # 704
    DEPTH_TOKENS_END = DEPTH_TOKENS_START + DEPTH_TOKENS # 832
    NAME = "xyzrotvec-cam-512xy128d"

class TrajectoryEncoder_xyzrotvec_512xy256d(TrajectoryEncoder_xyzrotvec_1024xy):
    XY_TOKENS = 512  # as in 1024 tokens in total going from 0 to 511
    DEPTH_SCALE = None
    DEPTH_RANGE = (.1, 1.0)  # [meters]
    DEPTH_TOKENS = 256
    remaining_tokens = 1024 - XY_TOKENS - DEPTH_TOKENS
    DEPTH_TOKENS_START = XY_TOKENS + (remaining_tokens)//2  
    DEPTH_TOKENS_END = DEPTH_TOKENS_START + DEPTH_TOKENS
    NAME = "xyzrotvec-cam-512xy256d"

class TrajectoryEncoder_molmo_1000(TrajectoryEncoder_xyzrotvec_1024xy):
    XY_TOKENS = 1000 # as in 1024 tokens in total going from 0 to 511
    DEPTH_SCALE = 100
    #DEPTH_RANGE = (.1, 1.0)  # [meters]
    #DEPTH_TOKENS = 256
    ROT_SCALE = (1000-1) / (2*np.pi)
    #remaining_tokens = 1024 - XY_TOKENS - DEPTH_TOKENS
    #DEPTH_TOKENS_START = XY_TOKENS + (remaining_tokens)//2  
    #DEPTH_TOKENS_END = DEPTH_TOKENS_START + DEPTH_TOKENS
    NAME = "molmo-1000"

    def to_string(self, traj_norm, depth_env, rotvec_int):
        """
        Molmo style encoder
        """
        traj_norm = traj_norm.float()
        rotvec_int = rotvec_int.float()
        traj_norm /= 10
        rotvec_int /= 10
        result = "<eetraj "
        for i, (kp_xy, kp_d, kp_r) in enumerate(zip(traj_norm, depth_env, rotvec_int), start=1):
            result += f'x{i}="{kp_xy[0]:.1f}" y{i}="{kp_xy[1]:.1f}" d{i}="{kp_d:.1f}" a{i}="{kp_r[0]:.1f}" b{i}="{kp_r[1]:.1f}" c{i}="{kp_r[2]:.1f}"'
            result += " "
        result = result.strip()
        result += "></eetraj>"
        return result
    
    def from_string(self, caption):
        attributes = ["x", "y", "d", "a", "b", "c"]
        pattern_parts = [rf'{attr}\d*="\s*([0-9]+(?:\.[0-9]+)?)"' for attr in attributes]
        full_pattern = r"\s+".join(pattern_parts)
        matches = re.findall(full_pattern, caption)
        if not matches:
            raise ValueError(f"No trajectory data found in string. {caption}")

        coords = [[float(val) for val in match] for match in matches]
        coords = torch.tensor(coords)
        # TODO(max): reverse the order here, why?
        # paligemma uses h,w convention and decode_caption flips it back to xy
        # molmo uses w, h  so we flip once here, decode_caption undoes it to obtain xy
        # re-write this so paligemma from_string flips.
        coords[:, :2] = coords[:, [1, 0]]
        coords[:, :2] *= 10
        coords[:, 3:] *= 10
        return coords.flatten().numpy().tolist()
    

def getActionEncInstance(name):
    if name == "xyzrotvec-rbt-100":
        return TrajectoryEncoder_rbt100()
    elif name == "xyzrotvec-rbt-128":
        return TrajectoryEncoder_rbt128()
    elif name == "xyzrotvec-rbt-256":
        return TrajectoryEncoder_rbt256()
    elif name == "xyzrotvec-rbt-512":
        return TrajectoryEncoder_rbt512()
    elif name == "xyzrotvec-rbt-1024":
        return TrajectoryEncoder_rbt1024()
    elif name == "xyzrotvec-cam-1024xy":
        return TrajectoryEncoder_xyzrotvec_1024xy()
    elif name == "xyzrotvec-cam-512xy":
        return TrajectoryEncoder_xyzrotvec_512xy()
    elif name == "xyzrotvec-cam-256xy":
        return TrajectoryEncoder_xyzrotvec_256xy()
    elif name == "xyzrotvec-cam-128xy":
        return TrajectoryEncoder_xyzrotvec_128xy()
    elif name == "xyzrotvec-cam-512xy128d":
        return TrajectoryEncoder_xyzrotvec_512xy128d()
    elif name == "xyzrotvec-cam-512xy256d":
        return TrajectoryEncoder_xyzrotvec_512xy256d()
    elif name == "molmo-1000":
        return TrajectoryEncoder_molmo_1000()
    else:
        raise ValueError(f"unknown encoder name {name}")


def check_encode_decode():
    """
    Check that encoding a trajectory to tokens and back does not produce large errors.
    """
    from utils_trajectory import DummyCamera
    from utils_trajectory import are_orns_close

    camera_extrinsic = [[[-0.759, 0.651, 0.0, 0.0], [0.301, 0.351, -0.887, 0.106], [-0.577, -0.673, -0.462, 0.575]]]
    camera_intrinsic = [[[410.029, 0.0, 224.0], [0.0, 410.029, 224.0], [0.0, 0.0, 1.0]]]

    camera = DummyCamera(camera_intrinsic, camera_extrinsic, width=448, height=448)
    points_3d = torch.tensor([[[-0.1689,  0.0338,  0.0350],
                               [-0.1137,  0.1394,  0.0700]]])
    
    orns_3d = torch.tensor([[[0.0000, 0.4484, 0.8939, 0.0000],
                             [0.0000, 0.4484, 0.8939, 0.0000]]])

    robot_pose = Pose.create_from_pq(p=[[.1,.2,.3]], q=[[1,0,0,0]])

    # check xyz-rotvec-rbt
    encoder_name = "xyzrotvec-rbt-100"
    action_encoder = getActionEncInstance(encoder_name)
    curve_25d, depth, token_str = action_encoder.encode_trajectory(points_3d, orns_3d, camera, robot_pose=robot_pose)
    points_3d_est, orns_3d_est = action_encoder.decode_trajectory(token_str, camera, robot_pose=robot_pose)
    #assert torch.allclose(points_3d, points_3d_est, atol=.005), "xyz-rotvec-rbt pose"
    orns_close, orns_delta = are_orns_close(orns_3d, orns_3d_est, return_max_diff=True, tol_degrees=2)
    assert orns_close 

    encoder_name = "xyzrotvec-cam-1024xy"
    action_encoder = getActionEncInstance(encoder_name)
    enc, dec = action_encoder.encode_trajectory, action_encoder.decode_trajectory


    curve_25d, depth, token_str = enc(points_3d, orns_3d, camera, robot_pose=robot_pose)
    points_3d_est, orns_3d_est = dec(token_str, camera, robot_pose=robot_pose)        
    points_delta = torch.abs(points_3d - points_3d_est).max()
    assert torch.allclose(points_3d, points_3d_est, atol=.005), "xyzrotvec-cam-1024xy"
    #assert torch.allclose(orns_3d, orns_3d_est, atol=.005), "xyz-rotvec-rbt orn"
    #assert are_orns_close(orns_3d, orns_3d_est)
    orns_close, orns_delta = are_orns_close(orns_3d, orns_3d_est, return_max_diff=True, tol_degrees=2)
    assert orns_close 
    #print(f"{encoder_name} pos {points_delta*100:0.4f} [cm]   orn {orns_delta:0.4f} [degrees]")

    if encoder_name == "xyzrotvec-cam-1024xy":
        token_str_cached = "<loc0565><loc0733><loc0063><seg042><seg066><seg068><loc0618><loc0834><loc0051><seg042><seg066><seg068>"
        assert token_str == token_str_cached, f"Token mismatch: {token_str} != {token_str_cached}"
        
    encoder_names = ("xyzrotvec-cam-1024xy", "xyzrotvec-cam-512xy", "xyzrotvec-cam-256xy", "xyzrotvec-cam-128xy",
                         "xyzrotvec-cam-512xy128d","xyzrotvec-cam-512xy256d",
                         "xyzrotvec-rbt-100","xyzrotvec-rbt-128","xyzrotvec-rbt-256",
                         "molmo-1000")
    
    encoder_name_length = max([len(name) for name in encoder_names])
    examples = dict()
    for encoder_name in encoder_names:
        enc = getActionEncInstance(encoder_name)

        curve_25d, depth, token_str = enc.encode_trajectory(points_3d, orns_3d, camera, robot_pose=robot_pose)
        points_3d_est, orns_3d_est = enc.decode_trajectory(token_str, camera, robot_pose=robot_pose)        
        points_delta = torch.abs(points_3d - points_3d_est).max()
        assert torch.allclose(points_3d, points_3d_est, atol=.005), f"{encoder_name} {points_3d} {points_3d_est}"
        #assert torch.allclose(orns_3d, orns_3d_est, atol=.005), "xyz-rotvec-rbt orn"
        #assert are_orns_close(orns_3d, orns_3d_est)
        orns_close, orns_delta = are_orns_close(orns_3d, orns_3d_est, return_max_diff=True, tol_degrees=2)
        print(f"{encoder_name:<{encoder_name_length}} pos {points_delta*100:0.4f} [cm]   orn {orns_delta:0.4f} [degrees]")
        examples[encoder_name] = token_str

        if encoder_name == "xyzrotvec-cam-1024xy":
            token_str_cached = "<loc0565><loc0733><loc0063><seg042><seg066><seg068><loc0618><loc0834><loc0051><seg042><seg066><seg068>"
            assert token_str == token_str_cached, f"Token mismatch: {token_str} != {token_str_cached}"
    print("All tests passed!\n")

    for encoder_name in encoder_names:
        print(f"{encoder_name:<{encoder_name_length}} {examples[encoder_name]}")
    print("\nMake sure the encoding looks right.\n")

    # Check that the encoder produces the right number of tokens
    encoder_names_paligemma = [name for name in encoder_names if "xyzrotvec-" in name]
    model_type = "google/paligemma2-3b-pt-224"
    processor = None
    try:
        from transformers import PaliGemmaProcessor
        processor = PaliGemmaProcessor.from_pretrained(model_type, use_fast=True)
    except ImportError:
        print("Warning, could not load preprocessor, couldn't check encoding lenght")
        return
    
    for encoder_name in encoder_names_paligemma:
        encoded = processor.tokenizer.encode(examples[encoder_name])
        print(f"{encoder_name:<{encoder_name_length}} {len(encoded):<8} {encoded}")
        #print(f"{encoder_name:<{encoder_name_length}} {examples[encoder_name]}")
        assert len(encoded) == 12
    
if __name__ == "__main__":
    check_encode_decode()        
    