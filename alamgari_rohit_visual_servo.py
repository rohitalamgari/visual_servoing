import numpy as np
from spatialmath import *

import matplotlib.pyplot as plt
import imageio
from scipy.linalg import expm

from config import DELTA_T, WORLD_CUBE, WORLD_LIMIT, IMAGE_SIZE, FOCAL_LENGTH, DELTA_ERROR_THRESHOLD, NUM_ITER, DIFFICULTY, DIFFICULTY_LEVELS
from misc_utils import draw_cube_3D, show_as_image
from geometry_utils import skew
from typing import Tuple



#### UT Austin CS 395T -- Visual Servoing Assignment 
#### Complete the three tasks (search for 'TASK') below.
#### See the readme file for more details.  
#### AI Usage Policy: you can get help with definitions, the math or python libraries from an AI agent. You can not ask it to generate code for you. 



world_lim = WORLD_LIMIT
im_size = IMAGE_SIZE
f = FOCAL_LENGTH
delta_error_threshold = DELTA_ERROR_THRESHOLD
num_iter = NUM_ITER




# make a skew-symmetric matrix out of 3D vector. Used for rotation computation
def skew(omega):
    return np.array([
        [0, -omega[2], omega[1]],
        [omega[2], 0, -omega[0]],
        [-omega[1], omega[0], 0]
    ])


########### TO DO  ##################################################################


##### TASK 1: Implement your image projection function which takes 3D points P and the focal length
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
    
    #NOTE: This method is already implemented in misc_utils
    #using the same formula from there

    depths = P[2]

    #projection is just fx/z & fy/z    
    proj_xy =  f * P[0:2] / depths
    
    return proj_xy, depths

##################
### this function generates an figure with two subplots. The left one shows, in 3D, the cube and the camera frame in world coordinate frames
### the right one shows the image from the camera point of view. 
def visualize_state(ax1, ax2, w_cube, gt_pose, f, draw_camera_frame=False, world_str="W", frame_str="C", world_lim=10):
    ax1.set_xlim3d(-world_lim, world_lim); ax1.set_ylim3d(-world_lim, world_lim); ax1.set_zlim3d(-world_lim, world_lim);
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z');

    #put the cube in the camera frame and take picture
    cam_cube = gt_pose.inv() * WORLD_CUBE
    proj_xy, gt_Zs = project_onto_im(cam_cube, f)
    
    #draw the world
    plt.sca(ax1)
    draw_cube_3D(ax1, WORLD_CUBE)
    #the plot function of spatialmath seems to be running into some issues when called repeatedly in a loop
    #so I am displaying the frame only in the beginning.
    if draw_camera_frame: 
        gt_pose.plot(frame=frame_str, color="green") 
    
    
    ax2.set_aspect('equal') 
    ax2.set_xlim(-im_size, im_size)
    ax2.set_ylim(-im_size, im_size)

    plt.sca(ax2)            
    show_as_image(ax2,proj_xy)

###################### 
## TASK 2: Write a function which takes feature image locations p, true depth values Zs, and the focal length
## and returns the interaction matrix used for servoing
def compute_interaction_matrix(p, Zs, f):
    """Compute the image-based visual servoing interaction matrix.

    Parameters
    - p: 2xN array of feature coordinates in the image (same units as f)
    - Zs: length-N array of corresponding depths
    - f: focal length (scalar)

    Returns
    - L: (2N)x6 interaction matrix
    """
    L = []

    N = p.shape[1]

    for i in range(N):
        x = p[0, i]
        y = p[1, i]
        z = Zs[i]

        L.append([
            [-f/z, 0, x/z, (x * y) / f, -(f ** 2 + x ** 2) / f, y],
            [0, -f/z, y/z, (f ** 2 + y ** 2) / f, -(x * y) / f, -x]
        ])
    

    L = np.vstack(L)
    return L

if __name__ == "__main__":
    #setup a new figure
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2) 
    title1_text = ax1.set_title("World (Goal Pose)")
    title2_text = ax2.set_title("Cam (Goal Image)")

    
    
    # desired goal image and its pose    
    # note: since we want the optical axis to be Z, we rotate the frame around y
    gt_pose = SE3(4,0.0,0.0)*SE3.Ry(-90, 'deg') 
    #initial camera pose (where the camera is before moving)

    if DIFFICULTY == 0:
        initial_pose = SE3(5, 1.0, 1.0)*SE3.Ry(-90, 'deg')
    elif DIFFICULTY == 1:
        initial_pose = SE3(5, 1.0, 2.0)*SE3.Ry(-90, 'deg')*SE3.Rz(-15, 'deg')
    elif DIFFICULTY == 2:
        initial_pose = SE3(5, 1.0, 2.0)*SE3.Ry(-90, 'deg')*SE3.Rz(-15, 'deg')*SE3.Rx(-5, 'deg')*SE3.Ry(-5, 'deg')
    else:
        initial_pose = SE3(-4, 1.0, 1.0) * SE3.Ry(90, "deg") * SE3.Rz(-15, "deg")



    # visualize the goal state
    #put the cube in the camera frame and take picture
    cam_cube = gt_pose.inv() * WORLD_CUBE
    gt_xy, gt_Zs = project_onto_im(cam_cube, f)
    visualize_state(ax1, ax2, WORLD_CUBE, gt_pose, f=f, draw_camera_frame=True, world_lim=world_lim, frame_str="G")
    plt.show()
    
    # visualize the initial state      
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2) 
    title1_text = ax1.set_title("World (Initial Pose)")
    title2_text = ax2.set_title("Cam (Initial Image)")

    visualize_state(ax1, ax2, WORLD_CUBE, initial_pose, f=f, draw_camera_frame=True, world_lim=world_lim, frame_str="I")
    plt.show()
    ##########################################

    fig_anim = plt.figure(figsize=(8, 4), dpi=100)  # choose a fixed size
    ax_anim = fig_anim.add_subplot(111)
    ax_anim.set_aspect('equal') 
    ax_anim.set_xlim(-im_size, im_size)
    ax_anim.set_ylim(-im_size, im_size)
    frames = []
    plt.ion()
    plt.show(block=False)


    ##################################################  
    ##################################################
    ### begin the servo loop
    cur_pose = initial_pose
    prev_error = 100000.0 #initialize to inf
    cur_error = 100000.0 #initialize to inf

    for i in range(num_iter):
        ax_anim.cla() 
    
        cam_cube = cur_pose.inv() * WORLD_CUBE
        cur_xy, cur_Zs = project_onto_im(cam_cube, f)

        
        #### TASK 3: compute the inteaction matrix L, use its inverse to compute volocities v
        # regular pseudo-inverse       
        # 
        # 
        error =  (gt_xy - cur_xy).flatten(order='F')#compute error. Make sure that it is consistent with your image coordinates
        print(error.shape)
        L = compute_interaction_matrix(cur_xy, cur_Zs, f)
        L = np.linalg.pinv(L)
        print(L.shape)

        v =  L @ error #compute the velocities from L


        ### end of Task 3
        #### use matrix exponential to convert the angular velocities to a rotation matrix.

        vx, vy, vz = v[0:3]
        wx, wy, wz = v[3:]

        dt = DELTA_T
        # a heuristic: reducing dt (i.e. update more frequently) as you approach the goal

        #speed up at the start, adjust based on difficulty
        dt *= .45 ** ( (i / 50) - 3 - (DIFFICULTY * .8)) #a exponential decay function
        # dt *= 20 / ((i ** 2) + 1)
        # dt *= (10 ** (DIFFICULTY))

        # sometimes it helped
        # you can use more sophisticated methods such as a learning rate scheduler 
        if cur_error < 0.1:
            dt /= 5
        elif cur_error < 0.01:
            dt /= 10

        # I ended up using my own skew function and the matrix exponential from scipy
        # because I couldn't get the exponential from mathspatial to work. Leaving it below in comments if 
        # anyone wants to play with it

        omega = [wx, wy, wz]
        R_delta = expm(skew(omega) * dt)
        T_motion = SE3.Rt(R_delta, np.array([vx, vy, vz]) * dt)
        print("T_motion")
        print(T_motion)
        print(v)
        
        ### I was hoping to use this but it didn't work for some reason. Can be removed
        # using SE3.Exp didn't work -- probably something to do with the ordering of the input
        # leaving it here in case anyone wants to play with it
        # would have been nice ot use the exponential map directly to compute the full motion

        # using the full exponential map
        # cur_twist = np.array([wx, wy, wz, vx, vy, vz])
        # T_motion = SE3.Exp(cur_twist)
        # print(T_motion)
        # #input("press enter")


        ## update the pose, record errors, update the animation
        cur_pose = cur_pose * T_motion 
        cur_error = np.linalg.norm(error, np.inf) 
        error_str = "%.3f" % cur_error
        print("error norm: " + error_str)
        
        T_str = "(%.2f,%.2f,%.2f)" % (cur_pose.t[0], cur_pose.t[1], cur_pose.t[2])

        ax_anim.set_title(DIFFICULTY_LEVELS[DIFFICULTY]+ ": t=" + str(i) + ", T=" + T_str + " Err="+error_str)
        
        cam_cube = cur_pose.inv() * WORLD_CUBE
        proj_xy1, _ = project_onto_im(cam_cube, f)        
        show_as_image(ax_anim,proj_xy1)      

        #put the gt image for comparison
        show_goal_image = True
        if show_goal_image:
            cam_cube = gt_pose.inv() * WORLD_CUBE
            proj_xy2, _ = project_onto_im(cam_cube, f)
            plt.sca(ax_anim)            
            show_as_image(ax_anim,proj_xy2)      

        ax_anim.set_aspect('equal') 
        ax_anim.set_xlim(-im_size, im_size)
        ax_anim.set_ylim(-im_size, im_size)
        fig_anim.canvas.draw()

        plt.pause(0.1) # Pause to allow for visual update
        
        if i%5 == 0: #skipping some frames to avoid animation crash
        
            buf = fig_anim.canvas.buffer_rgba()
            frame = np.asarray(buf)
            frames.append(frame.copy())

        delta_error = abs(prev_error-cur_error)
        prev_error = cur_error
        print("delta error:%f" % delta_error)
        if delta_error < delta_error_threshold:
            print("converged. Exiting..")
            break


    ## if you resize the figure during the animation, image io gives an error
    ## this trick (thanks CoPilot!) fixes that
    # Determine the target shape (e.g., shape of the first frame)
    target_shape = frames[0].shape

    # Resize or pad all frames to match the target shape
    uniform_frames = []
    for frame in frames:
        if frame.shape != target_shape:
            # Resize using PIL or OpenCV if needed
            from PIL import Image
            frame = Image.fromarray(frame)
            frame = frame.resize((target_shape[1], target_shape[0]))  # width, height
            frame = np.array(frame)
        uniform_frames.append(frame)

    #### draw a final figure to show the initial, final and goal poses in one place
    fig2 = plt.figure()
    fig2ax = fig2.add_subplot(111, projection='3d')
    title1_text = fig2ax.set_title("Inital, Final and Goal Poses")
    fig2ax.set_xlim3d(-world_lim, world_lim); fig2ax.set_ylim3d(-world_lim, world_lim); fig2ax.set_zlim3d(-world_lim, world_lim);
    fig2ax.set_xlabel('X'); fig2ax.set_ylabel('Y'); fig2ax.set_zlabel('Z');

    draw_cube_3D(fig2ax, WORLD_CUBE)
    plt.sca(fig2ax)
    gt_pose.plot(frame="G", color="black") 
    initial_pose.plot(frame="I", color="green") 
    cur_pose.plot(frame="F", color="red") 

    plt.show()

    input("Done. press enter to save the animation and exit.")

    # Don't forget to change the file name below!
    imageio.mimsave("alamgari_rohit_"+DIFFICULTY_LEVELS[DIFFICULTY]+".gif", uniform_frames, fps=10)

    # That's it. You made it! -Volkan Isler 09/30/2025


   


