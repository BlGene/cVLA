"""
This file is supposed to have utils for generating trajectories, meaning geometric stuff.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
from pdb import set_trace


def batch_multiply(tensor_A, tensor_B):
    # Reshape tensor_B to (N, 4, 1) to enable batch matrix multiplication
    tensor_B = tensor_B.unsqueeze(-1)
    # Perform batch matrix multiplication
    result = torch.bmm(tensor_A, tensor_B)
    # Remove the last dimension to get shape (N, 3)
    result = result.squeeze(-1)
    return result


def project_point(camera, point_3d):
    """
    Projects a 3D point onto the 2D image plane.

    Parameters:
    - camera: RenderCamera instance
    - point_3d: pytorch tensor of shape (N, 3), I think N batches envs
    
    Returns:
    - A tuple (x, y) representing the 2D image coordinates.
    """
    extrinsic_matrix = camera.get_extrinsic_matrix()
    # Convert the point to homogeneous coordinates
    point_3d_h = torch.cat([point_3d, torch.ones((len(point_3d), 1))], dim=1)    
    # Transform to camera coordinates
    point_camera = batch_multiply(extrinsic_matrix, point_3d_h)
    # Step 2: Get the intrinsic matrix for projection
    intrinsic_matrix = camera.get_intrinsic_matrix()
    # Project onto the image plane
    point_image_h = batch_multiply(intrinsic_matrix, point_camera[:,:3])
    # Normalize by the z-coordinate to get 2D image coordinates
    point_image_n = point_image_h[:, :2] / point_image_h[:, 2]
    return point_image_n

def convert_to_tensor(matrix):
    if isinstance(matrix, torch.Tensor):
        return matrix.clone().detach()
    else:
        return torch.tensor(matrix, dtype=torch.float32)
        
class DummyCamera:
    def __init__(self, intrinsic_matrix=None, extrinsic_matrix=None, width=None, height=None):
        self.intrinsic_matrix = None
        self.extrinsic_matrix = None
        if intrinsic_matrix is not None:
            self.intrinsic_matrix = convert_to_tensor(intrinsic_matrix)
        if extrinsic_matrix is not None:
            self.extrinsic_matrix = convert_to_tensor(extrinsic_matrix)
        self.width = width
        self.height = height

    def get_extrinsic_matrix(self):
        return self.extrinsic_matrix
    
    def get_intrinsic_matrix(self): 
        return self.intrinsic_matrix


def project_points(camera, points_3d, return_depth=False):
    """
    Projects a batch of 3D points onto the 2D image plane.
    
    Parameters:
    - camera: An instance of the Camera class with necessary methods.
    - points_3d: A tensor of shape (N, P, 3) representing the 3D points in world coordinates.
    
    Returns:
    - xy: a tensor of shape (N, P, 2) representing the 2D image coordinates.
    - depth: a tensor of shape (N, P) representing the depth of each point.

    """
    
    # Step 1: Get the extrinsic matrix and expand for batch processing
    extrinsic_matrix = camera.get_extrinsic_matrix()  # Shape (3, 4)
    extrinsic_matrix = extrinsic_matrix.unsqueeze(0).expand(-1, points_3d.shape[1], -1, -1) # shape (N, P, 3, 4)
    # Convert points to homogeneous coordinates by adding a fourth dimension of ones
    ones = torch.ones((*points_3d.shape[:2], 1))  # Shape (N, P, 1)
    points_3d_h = torch.cat([points_3d, ones], dim=-1)  # Shape (N, P, 4)
    # Transform to camera coordinates using the extrinsic matrix
    points_camera = batch_multiply(extrinsic_matrix.view(-1, 3, 4), points_3d_h.view(-1, 4))
    # Step 2: Get the intrinsic matrix and apply it for projection
    intrinsic_matrix = camera.get_intrinsic_matrix()  # Shape (N, 3, 3)
    intrinsic_matrix = intrinsic_matrix.unsqueeze(0).expand(-1, points_3d.shape[1], -1, -1) # shape (N, P, 3, 4)    
    # Project onto the image plane by applying the intrinsic matrix
    points_image_h = batch_multiply(intrinsic_matrix.view(-1, 3, 3), points_camera.view(-1, 3))
    points_image_h = points_image_h.view(points_3d.shape[0], points_3d.shape[1], 3)  # Shape (N, P, 3)
    # Normalize by the z-coordinate to get 2D image coordinates
    x = points_image_h[..., 0] / points_image_h[..., 2]
    y = points_image_h[..., 1] / points_image_h[..., 2]
    points25 = torch.stack((x,y, points_image_h[..., 2]), dim=-1)
    return points25

def unproject_points_cam(camera, points_25d):
    """
    Unprojects 2D image points back into 3D camera coordinates.

    Parameters:
    - camera: An instance of the Camera class with necessary methods.
    - points_2d: A tensor of shape (N, P, 3) representing the 2D image coordinates.

    Returns:
    - points_3d: A tensor of shape (N, P, 3) representing the 3D points in world coordinates.
    """
    points_25d = points_25d.clone()
    if points_25d.ndim == 2:
        points_25d = points_25d.unsqueeze(0)  # shape (N, P, 3)
    
    points_25d[..., 0] *= points_25d[..., 2] # undo normalization by z
    points_25d[..., 1] *= points_25d[..., 2] # undo normalization by z
    # Step 1: Get the intrinsic matrix and its inverse
    intrinsic_matrix = camera.get_intrinsic_matrix()  # Shape (N, 3, 3)
    intrinsic_matrix_inv = torch.linalg.inv(intrinsic_matrix)  # Inverse intrinsic matrix
    intrinsic_matrix_inv = intrinsic_matrix_inv.unsqueeze(0).expand(-1, points_25d.shape[1], -1, -1) # shape (N, P, 3, 4)    

    # Step 2: Convert from image plane to camera coordinates
    points_camera = batch_multiply(intrinsic_matrix_inv.view(-1, 3, 3), points_25d.view(-1, 3))  # Shape (N * P, 3)
    return points_camera, points_25d

def unproject_points(camera, points_25d):
    """
    Unprojects 2D image points back into 3D world coordinates.

    Parameters:
    - camera: An instance of the Camera class with necessary methods.
    - points_2d: A tensor of shape (N, P, 3) representing the 2D image coordinates.

    Returns:
    - points_3d: A tensor of shape (N, P, 3) representing the 3D points in world coordinates.
    """
    points_camera, points_25d = unproject_points_cam(camera, points_25d)
    points_camera_h = torch.cat((points_camera, torch.ones(len(points_camera), 1)), axis=1)

    # Step 3: Get the extrinsic matrix and its inverse
    extrinsic_matrix = camera.get_extrinsic_matrix()  # shape (N, 3, 4)
    extrinsic_matrix_h = torch.cat((extrinsic_matrix, torch.zeros((1, 1, 4))), dim=1)  # Shape (1, 4, 4)
    extrinsic_matrix_h[:, 3, 3] = 1  # Set the last element to 1
    extrinsic_matrix_inv = torch.linalg.inv(extrinsic_matrix_h)  # Inverse extrinsic matrix
    extrinsic_matrix_inv_h = extrinsic_matrix_inv.unsqueeze(0).expand(-1, points_camera_h.shape[0], -1, -1) # shape (N, P, 4, 4)

    # Step 4: Transform back to world coordinates
    points_3d_h_est = batch_multiply(extrinsic_matrix_inv_h.view(-1, 4, 4), points_camera_h.view(-1, 4))
    # recover shape (N, P, 3)
    return points_3d_h_est.view(points_25d.shape[0], points_25d.shape[1], 4)[...,:3]



def generate_curve_torch(points_a, points_b, up_dir=(0, 0, 1), height_scale=1, num_points=20):
    """
    Generate a Bézier going from points_a to points_b via points_mid
    Arguments:
        points_a: tensor shape (N, 3)
    """
    assert points_a.ndim == 2 and points_b.ndim == 2
    assert points_a.shape[1] == 3 and points_a.shape[1] == 3
    assert up_dir == (0, 0, 1)
    assert torch.is_tensor(points_a)
    assert torch.is_tensor(points_b)
    distance_between = torch.linalg.norm(points_a - points_b, axis=1)
    mid_point = (points_a + points_b)/2
    points_mid = mid_point + torch.tensor(up_dir)*distance_between*height_scale
    t = torch.linspace(0, 1, num_points).view(1, -1, 1)  # gives shape (1, num_points, 1)
    # # Calculate the Bézier curve points
    curve = (1-t)**2*points_a.unsqueeze(1) + 2*(1-t)*t*points_mid.unsqueeze(1) + t**2*points_b.unsqueeze(1)
    assert curve.shape[1:] == (num_points, 3)
    return (points_a, points_mid, points_b), curve


def subsample_trajectory(trajectory_old, points_new=8):
    """
    Subsamples a trajectory of shape (N, 20, 2) to a new shape (B, points_new, 2).

    Parameters:
    - trajectory_old: A NumPy array of shape (N, 20, 2) representing the old trajectories.
    - N: The number of points to sample from each trajectory (default is 8).

    Returns:
    - A NumPy array of shape (N, points_new, 2) representing the subsampled trajectories.
    """
    N, points_old, _ = trajectory_old.shape
    x_old = np.linspace(0, points_old - 1, points_old)
    x_new = np.linspace(0, points_old - 1, points_new)
    interpolated_array = np.zeros((N, points_new, 2))
    for i in range(trajectory_old.shape[0]):
        for j in range(trajectory_old.shape[2]):
            interpolated_array[i, :, j] = np.interp(x_new, x_old, trajectory_old[i, :, j])
    return interpolated_array

# Plotting
def plot_gradient_curve(axs, x, y, colormap='viridis'):
    """
    Plots a curve with colors progressing along its length.

    Parameters:
    - axs: The Matplotlib axis object to plot on.
    - x: The x-coordinates of the curve.
    - y: The y-coordinates of the curve.
    - colormap: The colormap to use for the progression (default is 'viridis').
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0, 1)
    lc = LineCollection(segments, cmap=colormap, norm=norm)
    lc.set_array(np.linspace(0, 1, len(segments)))  # Gradient progression along the curve
    axs.add_collection(lc)


def clip_and_interpolate(curve_2d, camera, num_points=None, return_didclip=False):
    """
    Arguments:
        num_points: interpolate to num_points
        return_didclip: returns True if we did a clip operation
    """
    curve_2d_clip = curve_2d.clone()
    curve_2d_clip[:, :, 0] = torch.clip(curve_2d_clip[:, :, 0], 0+1, camera.width)
    curve_2d_clip[:, :, 1] = torch.clip(curve_2d_clip[:, :, 1], 0+1, camera.height)
    didclip = not torch.allclose(curve_2d_clip, curve_2d)

    if num_points:
        curve2d_short = subsample_trajectory(curve_2d_clip, points_new=num_points)
        curve_2d_clip = torch.tensor(curve2d_short)

    if return_didclip:
        return curve_2d_clip, didclip
    
    return curve_2d_clip

from scipy.spatial.transform import Rotation as R


def are_orns_close(orns_3d, orns_3d_est, tol_degrees=0.4, return_max_diff=False):
    orns_R = R.from_quat(orns_3d.view(-1, 4), scalar_first=True)
    orns_est_R = R.from_quat(orns_3d_est.view(-1, 4), scalar_first=True)
    magnitude_radians = torch.tensor((orns_est_R * orns_R.inv()).magnitude()).float()
    angle_degrees = magnitude_radians * (180.0 / torch.pi)
    all_close = torch.allclose(angle_degrees, torch.zeros_like(angle_degrees), atol=tol_degrees)
    if return_max_diff:
        return all_close, angle_degrees.max()
    return all_close

def check_project_unproject():
    camera_extrinsic = [[[-0.759, 0.651, 0.0, 0.0], [0.301, 0.351, -0.887, 0.106], [-0.577, -0.673, -0.462, 0.575]]]
    camera_intrinsic = [[[410.029, 0.0, 224.0], [0.0, 410.029, 224.0], [0.0, 0.0, 1.0]]]

    camera = DummyCamera(camera_intrinsic, camera_extrinsic)
    points_3d = torch.tensor([[[-0.1689,  0.0338,  0.0350],
                               [-0.1137,  0.1394,  0.0700]]])
    
    points25 = project_points(camera, points_3d)
    points_3d_est = unproject_points(camera, points25)

    is_close = torch.allclose(points_3d, points_3d_est, atol=1e-5, rtol=1e-5)
    assert is_close
    
if __name__ == "__main__":
    check_project_unproject()
    print("All tests passed!")