import numpy as np

def print_array(array):
    """
    Prints the contents of a NumPy array.

    Parameters:
    array (np.ndarray): The NumPy array to be printed.
    """
    if isinstance(array, np.ndarray):
        print("Array contents:")
        print(array)
    else:
        print("The provided input is not a NumPy array.")

# Example usage
if __name__ == "__main__":
    # Creating a sample NumPy array
    sample_array = np.load("/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction/SAM-6D/SAM-6D/Data/Example/outputs/templates/xyz_1.npy")
    
    # Printing the array
    print_array(sample_array)
