import os
import json
import coco_text
from PIL import Image
import numpy as np
from skimage.draw import polygon
import matplotlib.pyplot as plt

__author__ = 'cryanwashere'
__version__ = '1.0'

IMAGE_SHAPE = (256,256)

def open_labels(label_path):
    
    cocotext = coco_text.COCO_Text(label_path)
    cocotext.info()
    print(cocotext.loadAnns(cocotext.getAnnIds(imgIds=[217925])))

def mask_from_points( mask: np.array, mask_points: list, original_shape: tuple ) -> np.array:
    #verts = list(zip(*[iter(mask_points)] * 2)) + [(0, 0)]
    #print(verts)
    mask_points = np.array(mask_points).reshape(len(mask_points)//2,2)
    mask_points /= original_shape
    mask_points *= mask.shape
    mask_points = mask_points.astype(int) - 1
    _x = mask_points[:,0]
    _y = mask_points[:,1]
    #print(_x, _y)
    rr, cc = polygon(_y, _x)
    mask[rr,cc] = 1.0
    return mask

def mask_from_anns(anns, image_shape, original_shape):
    mask = np.zeros(image_shape, dtype=np.float32)
    for ann in anns:
        #print(ann['utf8_string'])
        mask = mask_from_points(mask, ann['mask'], original_shape)
    return mask

def show_image_and_mask(image_array: np.array, mask: np.array):
    image_array = image_array.squeeze()
    mask = mask.squeeze()
    plt.subplot(1,2,1)
    plt.imshow(image_array)
    plt.subplot(1,2,2)
    plt.imshow(mask)
    plt.show()

def show_image_mask_and_output(image_array: np.array, mask:np.array, output: np.array):
    image_array = image_array.squeeze()
    mask = mask.squeeze()
    output = output.squeeze()
    plt.subplot(1,3,1)
    plt.imshow(image_array)
    plt.subplot(1,3,2)
    plt.imshow(mask)
    plt.subplot(1,3,3)
    plt.imshow(output)
    plt.show()

def load_image(path) -> np.array:
    image_array = Image.open(path)
    image_array = np.asarray(image_array.resize(IMAGE_SHAPE)) / 255.0
    return image_array


def load_image_and_mask(path, ct) -> tuple:
    

    image_array = Image.open(path)
    org_shape = image_array.size
    image_array = np.asarray(image_array.resize(IMAGE_SHAPE)) / 255.0

    image_id = int(path[-16:-4])
    anns = ct.loadAnns(ct.getAnnIds(imgIds=[image_id]))
    #print(anns)
    mask = mask_from_anns(anns, IMAGE_SHAPE, org_shape)
    
    return image_array, mask

def load_random_image_chunk(dir_path: str, num_imgs: int, ct: coco_text.COCO_Text) -> tuple:
    all_imgs = np.array(os.listdir(dir_path))
    img_idx = np.random.randint(len(all_imgs), size=num_imgs)
    img_subset = all_imgs[img_idx]

    img_batch, msk_batch = np.zeros((1,*IMAGE_SHAPE, 3)), np.zeros((1,*IMAGE_SHAPE))
    for img_name in img_subset:
        path = os.path.join(dir_path, img_name)
        img, msk = load_image_and_mask(path, ct)
        img_batch = np.concatenate([img_batch, np.expand_dims(img, axis=0)], axis=0)
        msk_batch = np.concatenate([msk_batch, np.expand_dims(msk, axis=0)], axis=0)
    
    return img_batch[1:], msk_batch[1:]



cocotext = coco_text.COCO_Text('cocotext.v2.json')
cocotext.info()

#load_image_and_mask('/Users/cameronryan/Desktop/cocotext/sample_image_set/COCO_train2014_000000318483.jpg', cocotext)
#load_image_and_mask('sample_image_set/COCO_train2014_000000560978.jpg', cocotext)