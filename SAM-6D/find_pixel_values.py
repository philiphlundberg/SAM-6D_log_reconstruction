from PIL import Image
import numpy as np

# Load the depth image
depth_image_path = "/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction/snapshot_depth.png"
# Smallest number 0, largest number 65535
depth_image = Image.open(depth_image_path)

# Convert image to numpy array
depth_array = np.array(depth_image)

# Get min, max, and unique values in the depth image
min_val = depth_array.min()
max_val = depth_array.max()
unique_vals = np.unique(depth_array)

# Return summary statistics
min_val, max_val, unique_vals[:10]  # Show first 10 unique values for reference

print(min_val, max_val, unique_vals[:20])