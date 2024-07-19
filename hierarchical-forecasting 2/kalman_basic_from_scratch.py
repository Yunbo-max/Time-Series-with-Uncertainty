import numpy as np
from numpy.linalg import inv

pos_observations = np.array([4000, 4260, 4550, 4860, 5110])
vel_observations = np.array([280, 282, 285, 286, 290])

x = np.concatenate(
    (pos_observations[:, np.newaxis], vel_observations[:, np.newaxis]), axis=1
)

## initial conditions
a = 2
v = 280
t = 1

## Process / estimation errors
error_est_x = 20
error_est_v = 5

# Observation errors
error_obs_x = 25
error_obs_v = 6

def prediction2d(x,v,t,a):