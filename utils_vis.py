import io
import html
import base64
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from utils_traj_tokens import decode_caption_xyzrotvec2, decode_trajectory_xyzrotvec2
from utils_trajectory import DummyCamera, project_points, convert_to_tensor
from PIL import Image
import torch


def get_standard_camera(image_height, image_width):
    camera_extrinsic = [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]]
    camera_intrinsic = [[[410.029, 0.0, 224.0], [0.0, 410.029, 224.0], [0.0, 0.0, 1.0]]]
    camera = DummyCamera(camera_intrinsic, camera_extrinsic, width=image_width, height=image_height)
    return camera

def extract_extrinsics(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Takes a vector [tx ty tz qw qx qy qz]
    return: translation vector (3) and rotation matrix (3, 3)
    """
    translation = pose[:3]
    rotation = Rotation.from_quat(np.array(pose[3:]), scalar_first=True).as_matrix()
    return (translation, rotation)

def local_to_world(t, rot):
    """
    Extrinsic matrix C from local coordinates to world coordinates.
    First apply rotation, then translation.
    t: (3) translation, world coordinates of local center
    rot: (3,3) rotation
    return: (4,4)
    """
    ext = np.eye(4)
    ext[0:3, 0:3] = rot
    ext[0:3, 3] = t
    return ext

def transform_3d(point_3d: np.ndarray, mat: np.ndarray):
    """ 
    Multiply point by transformation matrix in 3D in homogenous coordinates.
    point_3d: (3)
    mat: (4, 4)
    return: (3)
    """
    p3d = np.ones((4))
    p3d[:3] = point_3d # (x y z 1)
    res = mat @ p3d # (4, 4) x (4, 1)
    res = res / res[3] # remove homogeneous coordinate
    return res[:3]


def render_example(image, label, prediction=None, text=None, camera=None):
    """render examples, for use in notebook:
    
        from IPython.display import display, HTML
        display(HTML(html_imgs))
    """
    if isinstance(image, Image.Image):
        image_width, image_height = image.size
    elif isinstance(image, np.ndarray):
        image_width, image_height, _ = image.shape
    else:
        raise ValueError(f"image was {type(image)}")
        
    if camera is None:
        camera = get_standard_camera(image_width, image_height)
        
    plot_width, plot_height = 448, 448
    dpi = 100
    figsize = (plot_width / dpi, plot_height / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(image)
    ax.axis('off')
    if camera:
        try:
            # Pixels + depth
            # curve shape (N points, u v d), quaternion shape (N, qw qx qy qz)
            curve_25d, quat_c = decode_caption_xyzrotvec2(label, camera) 
            curve_2d  = curve_25d[:, :2]
            ax.plot(curve_2d[:, 0], curve_2d[:, 1],'.-', color='green')

            # === draw axis frame
            # curve shape (B, N, X Y Z) in camera 3D space
            # quat shape (B, N, qw qx qy qz)
            curve_w, quat_w = decode_trajectory_xyzrotvec2(label, camera)

            # coordinate frame
            dist = 0.1
            c = np.array([0, 0, 0])
            x = np.array([dist, 0, 0])
            y = np.array([0, dist, 0])
            z = np.array([0, 0, dist])

            # get transform from TCP coors to camera 3D
            pose = np.array([
                curve_w[0, 0, 0], # x
                curve_w[0, 0, 1], # y
                curve_w[0, 0, 2], # z
                quat_w[0, 0, 0], # qw
                quat_w[0, 0, 1], # qx
                quat_w[0, 0, 2], # qy
                quat_w[0, 0, 3] # qz
            ])
            t, rot = extract_extrinsics(pose) # (3), (3, 3)
            tcp_transform = local_to_world(t, rot) # (4, 4)

            # apply 3D transform
            c = transform_3d(c, tcp_transform) # (xyz)
            x = transform_3d(x, tcp_transform)
            y = transform_3d(y, tcp_transform)
            z = transform_3d(z, tcp_transform)

            # project with intrinsic
            points_3d = np.stack((c, x, y, z))[None, :, :] # (xyz) -> (1, 4, xyz)
            points_2d = project_points(camera, convert_to_tensor(points_3d))
            points_2d = points_2d[0, :, :2]

            c = points_2d[0]
            x = points_2d[1]
            y = points_2d[2]
            z = points_2d[3]
            ax.plot((c[0], x[0]), (c[1], x[1]), '.-', color='red')
            ax.plot((c[0], y[0]), (c[1], y[1]), '.-', color='green')
            ax.plot((c[0], z[0]), (c[1], z[1]), '.-', color='blue')
        except ValueError:
            pass

    html_text = ""
    if text:
       html_text = f'{html.escape("text: "+text)}'
    html_text += f'</br></br>{html.escape("label: "+label)}'

    if prediction:
        html_text += f'</br></br>{html.escape("pred: "+prediction)}'
        try:
            curve_2d_gt, quat_c = decode_caption_xyzrotvec2(prediction, camera)
            ax.plot(curve_2d_gt[:, 0], curve_2d_gt[:, 1],'.-', color='lime')
        except ValueError:
            pass

    with io.BytesIO() as buffer:
        fig.savefig(buffer, format='jpeg',bbox_inches='tight', dpi=dpi)
        image_b64 = str(base64.b64encode(buffer.getvalue()), "utf-8")
        res_str =  f"data:image/jpeg;base64,{image_b64}"
    plt.close(fig)
    return f"""
<div style="display: inline-flex; align-items: center; justify-content: center;">
    <img style="width:224px; height:224px;" src="{res_str}" />
    <p style="width:256px; margin:10px; font-size:small;">{html_text}</p>
</div>
"""
