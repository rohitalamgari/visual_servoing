import numpy as np

# Global constants
WORLD_LIMIT = 10
IMAGE_SIZE = 100
FOCAL_LENGTH = 100.0
DELTA_ERROR_THRESHOLD = 0.01 / FOCAL_LENGTH
NUM_ITER = 1000

# Geometry configuration
THICKNESS = 0.5

# World cube in world coordinates (square prism with given thickness)
WORLD_CUBE = np.array([
    [-THICKNESS, THICKNESS, THICKNESS, -THICKNESS, -THICKNESS, THICKNESS, THICKNESS, -THICKNESS],
    [-1, -1, 1, 1, -1, -1, 1, 1],
    [-1, -1, -1, -1, 1, 1, 1, 1],
])

# set the difficulty level to test your code
DIFFICULTY = 3
DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard", "Extreme"]

# you mau need to play with this for convergence
# it effecticly acts as step size
DELTA_T = 0.005