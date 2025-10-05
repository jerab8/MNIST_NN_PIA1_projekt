import struct
import numpy as np
import matplotlib.pyplot as plt

def load_idx3_ubyte(file_path):
    with open(file_path, 'rb') as f:
        # Read metadata
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        # Load the image data
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num_images, rows, cols)
    return data

# Example usage
images = load_idx3_ubyte("train-images.idx3-ubyte")

# Show first image
print(images[49])
plt.imshow(images[5020], cmap="gray")
plt.show()

def load_idx1_ubyte(file_path):
    with open(file_path, 'rb') as f:
        # Read magic number and number of labels
        magic, num_labels = struct.unpack(">II", f.read(8))
        # Read the labels
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Example usage
labels = load_idx1_ubyte("train-labels.idx1-ubyte")

print("Label:", labels[5020])

