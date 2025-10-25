
import numpy as np
from typing import Tuple
def draw_cube_3D(ax, Q):
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=Q[0], ys=Q[1], zs=Q[2], s=20)  # draw vertices
    
    # draw lines joining the vertices
    lines = [[0,1,5,6], [1,2,6,7], [2,3,7,4], [3,0,4,5]]
    #ax.set_xlim3d(-2, 3); ax.set_ylim3d(0, 5); ax.set_zlim3d(0, 5);
    #ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z');
    for line in lines:
        ax.plot([Q[0,i] for i in line], [Q[1,i] for i in line], [Q[2,i] for i in line])

def show_as_image(ax2, proj_xy):
    ax2.scatter(proj_xy[0], proj_xy[1], c='k')
    for i  in range(proj_xy.shape[1]):
        ax2.text(proj_xy[0,i] + 0.5, proj_xy[1,i] + 0.5, str(i), fontsize=10, color='blue')

    lines = [[0,1,5,6], [1,2,6,7], [2,3,7,4], [3,0,4,5]]
    #ax.set_xlim3d(-2, 3); ax.set_ylim3d(0, 5); ax.set_zlim3d(0, 5);
    #ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z');
    
    for line in lines:
        ax2.plot([proj_xy[0,i] for i in line], [proj_xy[1,i] for i in line])


##### Task1: Implement your image projection function which takes 3D points P and the focal length
## and returns the projections and the depth values to be used for error and interaction matrix computation
# NOTE: for this assignment we will use direct projections in the camera (not pixel) frame
def project_onto_im(P: np.ndarray, f: float) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D points in the camera frame onto the image plane.

    Parameters
    - P: 3xN array of 3D points expressed in the camera frame
    - f: focal length (scalar)

    Returns
    - proj_xy: 2xN array of image-plane coordinates (in same units as f)
    - depths: length-N array of Z depths
    """
    proj = P / P[2]
    proj_xy = f * proj[0:2, :]
    return proj_xy, P[2]