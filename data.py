from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

IMAGE_SHAPE = (256, 256)

def show(*stuff):
    num_things = len(stuff)
    for i in range(num_things):
        plt.subplot(1, num_things, i+1)
        plt.imshow(stuff[i].squeeze())
    plt.show()

def load_image(path: str) -> np.array:
    image_array = Image.open(path)
    image_array = np.asarray(image_array.resize(IMAGE_SHAPE)) / 255.0
    return image_array

def load_mask(path: str) -> np.array:
    image_array = Image.open(path)
    image_array = np.asarray(image_array.resize(IMAGE_SHAPE)) / 255.0
    return image_array

def load_images(start_idx: int, end_idx: int, verbose: bool = False) -> tuple:
    if verbose:
        print("loading {} images...".format(end_idx-start_idx))
    image_names = os.listdir('Images/Train')

    image_path = 'Images/Train'
    mask_path = 'Text_Region_Mask/Train'

    images, masks = np.zeros((1,*IMAGE_SHAPE, 3)), np.zeros((1,*IMAGE_SHAPE))
    for im in image_names[start_idx:end_idx]:
        if verbose:
            print("loading {}".format(im))
        image = load_image(os.path.join(image_path, im))
        mask = load_mask(os.path.join(mask_path, im[:-3]+"png"))
        
        images = np.concatenate([images, np.expand_dims(image, axis=0)], axis=0)
        masks = np.concatenate([masks, np.expand_dims(mask, axis=0)], axis=0)
    
    return images[1:], masks[1:]

        