# cs395T Fall2025 Visual Servoing Assignment
## Due: Oct 24, 2025 midnight.
## You can earn up to 15% + 5% + 5% of the course grade from this assignment. 

The goals of this assignment are  
* to reinforce what we learned in class about camera models, motion and visual servoing.
* understand the practical use and limitations of a standard iterative method based on a first order approximation.

  
Specifically, we will use the corners of a cube as features for moving the camera to a desired pose --described in terms of a desired goal image. 
We will generate a view from a desired pose and write a controller to drive camera pose toward that pose using image information only. 
To do so, we will minimize the error (defined as the L2 distance between the current and desired features in the image).

The homework has three components. The first one is required. You can take on the other ones for bonus points. 

## Part 1: Standard Image Based Visual-Servoing (15 points)
Take a look at `visual_servo.py`. You will see three tasks marked as Task 1, Task 2 and Task 3 in the code. They all require you to complete missing code segments. 

```
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
    
    proj_xy = 
    depths = 
    return proj_xy, depths
```
```
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
    
    return L
```
```
        #### TASK 3: compute the inteaction matrix L, use its inverse to compute volocities v
        # regular pseudo-inverse       
        # 
        # 
        error =  #compute error. Make sure that it is consistent with your image coordinates
        L = compute_interaction_matrix(cur_xy, cur_Zs, f)
        v =   #compute the velocities from L


        ### end of Task 3
```
The required python modules can be found in `requirements.txt`. The code uses [spatialmath python library](https://bdaiinstitute.github.io/spatialmath-python/). There are also some helper functions to visualize the camera poses, images and to store the main servo loop as an animation. 

Relevant parameters can be found in `config.py`

**What to submit** 

After you run finalize your code, run it on the three difficulty levels by setting the `DIFFICULTY` variable in the config file. Submit 
1. The three animations (make sure to change the prefix of the file name to your `lastname_firstname_animation_" ...
2. Your code (just submit the visual servo file but rename it to `lastname_firstname_visualservo.py`)
3. A short 1 page reflection where you talk about if and how you need to tweak your code to run across different instances. Did you have to change any of the parameters such as the DELTA_T? If so, specifiy which values you used (your animation should be reproducable!) and how you come up with them. How about number of iterations, convergence behavior? 

## Part 2: Practical Issues (up to 5 points)

In part 1, we took many shortcuts. For example, we used the true depth values when computing the interaction matrix. For bonus, you can remove the assumption about known depth and add triangulation to compute depth assuming known correspondences (2 points). You can also add a real camera model using pixels and shiften image center given as an intrinsics matrix (2 points). Perform experiments with added noise to assess the performance of your method in comparison to the version in part 1 (1 point). Submit your new code as `your_name_visual_servo_bonus1.py` along with other necessary files in a self contained directory. 

## Part 3: Beyond the Neighborhood (up to 5 points)

As you will observe, the first order servo solution works only in the region where the error behaves linearly. This is an open ended question to address this limitation using your favorite approach: hand-designed heuristics, RL-based strategies, supervised learning e.g. from expert generated trajectories etc. and comeup with a servoing strategy which works in more general settings. You can for example start with that policy and switch to the servo policy when you are "close enough." Submit 1) your new code as `your_name_visual_servo_bonus2.py` along with other necessary files in a self contained directory, and 2) a 1-2 page description of your approach and results. 

All files should be zipped in a single folder `your_name_visual_servo_submission.zip` and submitted through Canvas. 


#### AI Usage Policy: you can get help with definitions, the math or python libraries from an AI agent. You can not ask it to generate code for you. 
