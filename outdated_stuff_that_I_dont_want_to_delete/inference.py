import model
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
IMAGE_SHAPE = (256,256)

def show_image_and_mask(image_array: np.array, mask: np.array):
    image_array = image_array.squeeze()
    mask = mask.squeeze()
    plt.subplot(1,2,1)
    plt.imshow(image_array)
    plt.subplot(1,2,2)
    plt.imshow(mask)
    plt.show()

def load_image(path) -> np.array:
    image_array = Image.open(path)
    image_array = np.asarray(image_array.resize(IMAGE_SHAPE)) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

kernel_dir = sys.argv[1]
model = model.ConvModel()
model.load_kernels(kernel_dir)

image_path = sys.argv[2]
image = load_image(image_path)

print(image.shape)
output = model.model_fn(image, *model.kernels)

show_image_and_mask(image, output)