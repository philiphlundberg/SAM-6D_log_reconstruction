# Load an .npz file

import numpy as np

# Load the .npz file
data = np.load('/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction/SAM-6D/SAM-6D/Data/Example/1_log_no_obst/pile_00000/log_data.npz') 

# Access the data in the .npz file
direction = data['dir']

print(direction)

position = data['pos']

print(position)