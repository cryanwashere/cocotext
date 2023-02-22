import coco_text
import sys
import os
import requests

cocotext = coco_text.COCO_Text('cocotext.v2.json')
img_list = list(cocotext.imgs.items())

save_dir = sys.argv[1]
start_idx = int(sys.argv[2])
end_idx = int(sys.argv[3])


if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

img_subset = img_list[start_idx:end_idx]
for img in img_subset:
    url = "http://images.cocodataset.org/train2014/"+ img[1]['file_name']
    print("downloading {}".format(url))
    im_data = requests.get(url).content
    with open(os.path.join(save_dir, img[1]['file_name']), 'wb') as f:
        f.write(im_data)

