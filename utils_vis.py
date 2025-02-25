import io
import html
import base64
import numpy as np
import matplotlib.pyplot as plt
from utils_traj_tokens import decode_caption_xyzrotvec2
from utils_trajectory import DummyCamera
from PIL import Image


def get_standard_camera(image_height, image_width):
    camera_extrinsic = [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]]
    camera_intrinsic = [[[410.029, 0.0, 224.0], [0.0, 410.029, 224.0], [0.0, 0.0, 1.0]]]
    camera = DummyCamera(camera_intrinsic, camera_extrinsic, width=image_width, height=image_height)
    return camera

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
        raise ValueError
        
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
            curve_25d, quat_c = decode_caption_xyzrotvec2(label, camera)
            curve_2d  = curve_25d[:, :2]
            ax.plot(curve_2d[:, 0], curve_2d[:, 1],'.-', color='green')
        except ValueError:
            pass

    html_text = ""
    if text:
       html_text = f"{html.escape("text: "+text)}"
    html_text += f"</br></br>{html.escape("label: "+label)}"

    if prediction:
        html_text += f"</br></br>{html.escape("pred: "+prediction)}"
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

