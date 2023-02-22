import requests

ann_data = requests.get('https://github.com/bgshih/cocotext/releases/download/dl/cocotext.v2.zip').content
with open('cocotext.v2.zip','wb') as f:
    f.write(ann_data)