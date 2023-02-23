import shutil
import os
import sys

main_path = '/Volumes/easystore/cocotext/train2014/train2014'
images = os.listdir(main_path)
print("got image directory list")

start_idx = sys.argv[1]
end_idx = sys.argv[2]
print("grabbing images from {} to {}".format(start_idx, end_idx))

image_subset = images[start_idx:end_idx]
for img_name in image_subset:
    print("loading {}".format(img_name))
    shutil.copyfile(os.path.join(main_path, img_name), os.path.join('sample_image_set', img_name))