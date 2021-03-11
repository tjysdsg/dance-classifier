from train_idance.preprocess_pose import get_vid
from train_idance.common import CATEGORIES
import cv2
import os
import json


def main():
    data_dir = os.path.join('idance')
    out_dir = 'idance-pose'

    fps = {}
    for cat_dir in os.scandir(data_dir):  # categories
        if cat_dir.is_dir():
            category = cat_dir.name
            if category in CATEGORIES:
                for v in os.scandir(cat_dir.path):
                    vid = get_vid(v.name)
                    cam = cv2.VideoCapture(v.path)
                    fps[vid] = cam.get(cv2.CAP_PROP_FPS)

    json.dump(fps, open(os.path.join(out_dir, f'fps.json'), 'w'))


if __name__ == '__main__':
    main()
