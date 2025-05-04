import io
import html
import base64
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from cvla.utils_traj_tokens import getActionEncInstance
from cvla.utils_trajectory import DummyCamera, project_points, convert_to_tensor


def get_standard_camera(image_height, image_width):
    camera_extrinsic = [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]]
    camera_intrinsic = [[[410.029, 0.0, 224.0], [0.0, 410.029, 224.0], [0.0, 0.0, 1.0]]]
    camera = DummyCamera(camera_intrinsic, camera_extrinsic, width=image_width, height=image_height)
    return camera

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

def draw_coordinate_frame(label, camera, enc, ax, axis_length=0.05):
    """
    Draws a coordinate frame on a Matplotlib axis for each point in a trajectory.

    Parameters:
        label: Input label for decoding.
        camera: Camera model with intrinsics.
        enc: Encoder/decoder that provides trajectory poses.
        ax: Matplotlib axis to draw on.
        axis_length: Length of the axis lines (default is 0.1 meters).
    """
    # TODO(max): this is maxims bad code, should use same functions as utils_trajectory.py
    # Decode 3D trajectory (positions and quaternions)
    curve_w, quat_w = enc.decode_trajectory(label, camera)

    # Define local coordinate frame axes
    origin = np.array([0, 0, 0])
    x_axis = np.array([axis_length, 0, 0])
    y_axis = np.array([0, axis_length, 0])
    z_axis = np.array([0, 0, axis_length])

    num_keypoints = curve_w.shape[1]
    for i in range(num_keypoints):
        # Extract pose as [x, y, z, qw, qx, qy, qz]
        pose = np.hstack([curve_w[0, i], quat_w[0, i]])
        transform = np.eye(4)
        transform[0:3, 3] = pose[:3]
        transform[0:3, 0:3] = Rotation.from_quat(np.array(pose[3:]), scalar_first=True).as_matrix()
        points_world = np.stack([
            transform_3d(origin, transform),
            transform_3d(x_axis, transform),
            transform_3d(y_axis, transform),
            transform_3d(z_axis, transform)
        ])[None]  # Shape (1, 4, 3)
        points_2d = project_points(camera, convert_to_tensor(points_world))[0, :, :2]
        c, x, y, z = points_2d
        ax.plot([c[0], x[0]], [c[1], x[1]], '.-', color='red')
        ax.plot([c[0], y[0]], [c[1], y[1]], '.-', color='green')
        ax.plot([c[0], z[0]], [c[1], z[1]], '.-', color='blue')

def draw_probas_edge(image, pred_scores):
    # pred_scores: see render_example documentation
    image_height, image_width, _ = image.shape

    # === convert token positions and probabilities into histogram
    loc_h = pred_scores["loc_h"]/(1024-1)*image_height
    loc_w = pred_scores["loc_w"]/(1024-1)*image_width
    probas_y, _ = np.histogram(loc_h, bins=image_height, range=(0, image_height), weights=pred_scores["scores_h"])
    probas_x, _ = np.histogram(loc_w, bins=image_width, range=(0, image_width), weights=pred_scores["scores_w"])
    eps = 0.1
    probas_y = probas_y / (probas_y.max() + eps) * 255.0 # make colors visible
    probas_x = probas_x / (probas_x.max() + eps) * 255.0 # make colors visible

    # === draw the strip
    stripe_width = 30 # in pixels
    color_y = np.array([1.0, 1.0, 0.0])
    color_strip = probas_y[:, None] * color_y[None, :] # (image_height, 3)
    image[:, 0:stripe_width] = color_strip[:image_height, None, :3] # (height, stripe_width (broadcast), 3)

    color_x = np.array([0.0, 1.0, 0.5])
    color_strip = probas_x[:, None] * color_x[None, :] # (image_width, 3)
    image[0:stripe_width, :] = color_strip # (stripe_width (brooadcast), width, 3)

    return image

def render_example(image, label: str=None, prediction: str=None, camera=None, enc=None, enc_pred=None, text: str=None, i=0, extra_text: str=None):
    """
    Render an example image to HTML, e.g., for use in notebooks.
    Arguments:
        image: rgb array or (depth, rgb) tuple
        label: the suffix to be predicted
        prediction: the model prediction
        camera: the camera used to record the image
        enc: the default encoder used to decode the both label and prediction
        enc_pred: the encoder that overwrites the default encoder for the prediction
        text: the prefix
    """

    enc_label = enc
    if enc_label is None:
        print("Warning: default action encoder is used")
        enc_label = getActionEncInstance("xyzrotvec-cam-1024xy")
        
    if enc_pred is None:
        enc_pred = enc_label

    if isinstance(image, Image.Image):
        image_width, image_height = image.size
    elif isinstance(image, np.ndarray):
        image_width, image_height, _ = image.shape
    else:
        raise ValueError(f"image was {type(image)}")
        
    if camera is None:
        print("Warning: getting standard camera")
        camera = get_standard_camera(image_width, image_height)

    # logic for when to draw coords
    draw_state_coords = True
    draw_pred_coords = True
    draw_label_coords = True
    if prediction is not None:
        draw_label_coords = False

    plot_width, plot_height = 448, 448

    dpi = 100
    figsize = (plot_width / dpi, plot_height / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(image)
    ax.axis('off')

    if draw_state_coords:
        #robot_state = text.split(" ")[-1]
        robot_state =text
        draw_coordinate_frame(robot_state, camera, enc_label, ax)        
    
    curve_25d, quat_c = None, None
    try:
        curve_25d, quat_c = enc_label.decode_caption(label, camera) 
    except (ValueError, TypeError):
        pass
    if curve_25d is not None:
        curve_2d  = curve_25d[:, :2]
        ax.plot(curve_2d[:, 0], curve_2d[:, 1],'.-', color='green', linewidth=2)
        if draw_label_coords:
            draw_coordinate_frame(label, camera, enc_label, ax)            
    
    html_text = ""
    if text:
        html_text = f'{html.escape("text: "+text)}'
    if extra_text:
        html_text += f'</br></br>{html.escape("info: "+extra_text)}'
    html_text += f'</br></br>{html.escape("label: "+str(label))}'

    if prediction:
        html_text += f'</br></br>{html.escape("pred: "+prediction)}'
        try:
            curve_2d_gt, quat_c = enc_pred.decode_caption(prediction, camera)
            ax.plot(curve_2d_gt[:, 0], curve_2d_gt[:, 1],'.-', color='magenta', linewidth=2)
            ax.scatter(curve_2d_gt[0, 0], curve_2d_gt[0, 1], color='red')
            if draw_pred_coords:
                draw_coordinate_frame(prediction, camera, enc_pred, ax)
        except (ValueError, IndexError):
            pass

    # save fig for visualization 
    # fig.savefig(f"visuals/tmp_{i}.jpg", format='jpeg',bbox_inches='tight', dpi=300)

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
