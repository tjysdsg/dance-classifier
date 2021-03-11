import os
import numpy as np
from train_letsdance.common import KEYPOINTS
from multiprocessing import Process
import json


def get_vid(category: str, vid: str, person_idx: int):
    return f'{category}_{vid}_{person_idx}'


def preprocess_category(category: str, cat_dir: str, cat_out_dir: str):
    all_data = {}  # vid per person -> list of frames -> keypoints
    xs = []
    ys = []
    for f in os.scandir(cat_dir):  # video frames inside one category
        filename: str = f.name
        split = filename.split('_')
        fid = int(os.path.splitext(split[-1])[0])

        data = json.load(open(f.path, 'r'))
        for pid, person in enumerate(data):
            assert "person" in person[0]
            person = person[1:]
            points = {}
            for pt in person:
                pt_name = pt[0]
                if pt_name in KEYPOINTS:
                    points[pt_name] = pt[1]

            coords = []
            for kp in KEYPOINTS:  # ensure order by doing this
                coords.append(points[kp])
            coords = np.asarray(coords).T  # [[x1,x2,...], [y1,y2,...]]

            # find min max of X Y
            xs.append(coords[0])
            ys.append(coords[1])

            coords = coords.tolist()

            vid = get_vid(category, '_'.join(split[:-1]), pid)
            if vid not in all_data:
                all_data[vid] = {}
            all_data[vid][fid] = coords

    json.dump(all_data, open(os.path.join(cat_out_dir, f'{category}.json'), 'w'))

    xs = np.asarray(xs)
    ys = np.asarray(ys)
    # print(f'{category}: mean_x={np.mean(xs)}, std_x={np.std(xs)}')
    # print(f'{category}: mean_y={np.mean(ys)}, std_y={np.std(ys)}')
    return xs, ys


def main():
    out_dir = 'letsdance-densepose'
    data_dir = os.path.join('letsdance', 'densepose')

    procs = []
    x_min = y_min = np.inf
    x_max = y_max = -np.inf
    for cat_dir in os.scandir(data_dir):  # categories
        if cat_dir.is_dir():
            category = cat_dir.name

            # procs.append(Process(target=preprocess_category, args=(category, cat_dir.path, out_dir)))
            xs, ys = preprocess_category(category, cat_dir.path, out_dir)

            if len(xs) > 0 and len(ys) > 0:
                x_min = min(x_min, np.min(xs))
                x_max = max(x_max, np.max(xs))
                y_min = min(y_min, np.min(ys))
                y_max = max(y_max, np.max(ys))
    print(x_min, x_max, y_min, y_max)

    for p in procs:
        p.start()

    for p in procs:
        p.join()


if __name__ == '__main__':
    main()
