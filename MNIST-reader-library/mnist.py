# An MNIST Loader for Python
import gzip
import struct
import numpy as np

def load_images(filename):
    # Open and unzip the file of images
    with gzip.open(filename, 'rb') as f:
        # Read the header information into a bunch of variables
        _ignored, n_images, columns, rows = struct.unpack('>IIII', f.read(16))
        # Read all tthe pixels into a NumPy array of bytes
        all_pixels = np.frombuffer(f.read(), dtype=np.uint8)
        # Reshape the pixels into  matrix where each line is an image:
        return all_pixels.reshape(n_images, columns * rows)
    
def prepend_bias(X):
    # Insert a column of 1s in the postion of X
    # ("axis=1" stands for: "inserrt a column, not a row")
    return np.insert(X, 0, 1, axis=1)