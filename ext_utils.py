# -*- coding: utf-8 -*-

import glob
import re
import os

def is_img(fp):
    return any(fp.endswith(ext) for ext in [".png", ".jpg", ".jpeg"])

if __name__ == '__main__':
    img_dir = r'/xxx/pytorch-sepconv/Imgclassified/道'
    frames = [os.path.join(img_dir, x) for x in os.listdir(img_dir)]
    frames = [x for x in frames if is_img(x)]
    frames.sort(key=lambda x: int(re.findall('-\d+-', x)[0][1:-1]))
    print(frames)
    with open(os.path.join(img_dir, 'img_list.txt'), 'w') as f:
        for img_path in frames:
            f.write(img_path + '\n')