import os
import cv2
import numpy as np

def calc_IE(pred_img_path, gt_img_path):
    pred_img = cv2.imread(pred_img_path)
    gt_img = cv2.imread(gt_img_path)
    res = (pred_img - gt_img)**2
    print(res.shape)
    res = np.sum(res, axis=-1)
    print(res.shape)
    res = np.sqrt(res)
    res = np.mean(res)
    return res

if __name__ == '__main__':
    pred_dir = 'middlebury-other-pred'
    gt_dir = 'other-gt-interp'
    scenarios = os.listdir(pred_dir)
    ies = []
    for scenario in scenarios:
        print(scenario)
        pred_img_path = os.path.join(pred_dir, scenario, '1.png')
        gt_img_path = os.path.join(gt_dir, scenario, 'frame10i11.png')
        ies.append(calc_IE(pred_img_path, gt_img_path))
    print(ies)
    print(np.mean(np.array(ies)))