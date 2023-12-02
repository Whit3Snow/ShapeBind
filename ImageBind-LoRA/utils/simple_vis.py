import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def visualize_point_clouds_3d_v2(pcl_lst, title_lst=None, vis_axis_order=[0, 2, 1], fig_title=None):
    if title_lst is None:
        title_lst = [""] * len(pcl_lst)
    
    fig = plt.figure(figsize=(3 * len(pcl_lst), 3))
    if fig_title is not None:
        plt.title(fig_title)
        
    for idx, (pts, title) in enumerate(zip(pcl_lst, title_lst)):
        ax1 = fig.add_subplot(1, len(pcl_lst), 1 + idx, projection='3d')
        ax1.set_title(title)
        ax1.scatter(pts[:, vis_axis_order[0]], pts[:, vis_axis_order[1]], pts[:, vis_axis_order[2]], s=2)
    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    res = np.array(fig.canvas.renderer._renderer)
    res = np.transpose(res, (2, 0, 1))

    plt.close()
    res = Image.fromarray(res[:3].transpose(1,2,0))
    return res

