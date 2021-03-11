from pose_cnn import PoseCNN
import json
import torch
import os
import numpy as np
from utils import extract_consecutive_frames
from sys import argv

N_CONSECUTIVE_FRAMES = 16
ORIGINAL_FPS = 30  # data-samples/demo.mp4
FPS = 10


def prepross_json(json_dir: str) -> np.ndarray:
    ret = []

    fids = []
    frame_path = []
    for f in os.scandir(json_dir):
        fid = int(os.path.splitext(f.name)[0].split('_')[1])
        fids.append(fid)
        frame_path.append(f.path)

    idx_sort = np.argsort(fids)
    fids = np.asarray(fids)[idx_sort]
    frame_path = np.asarray(frame_path)[idx_sort]

    prev_coords = None
    prev_fid = -1
    for i, fid in enumerate(fids):
        path = frame_path[i]
        data = json.load(open(path, 'r'))
        data = data['people']

        # handle missing frames, use coordinates from the previous frame
        for j in range(prev_fid + 1, fid):
            print(f"WARNING: frame {j} of {path} missing")
            if prev_coords is not None:
                ret.append(prev_coords)

        # if a frame doesn't contain a person, use coordinates from the previous frame
        if len(data) == 0:
            print(f"WARNING: {path} doesn't contain any person")
            if prev_coords is not None:
                ret.append(prev_coords)
            continue

        data = data[0]['pose_keypoints_2d']
        xs = data[::3]
        ys = data[1::3]
        cs = data[2::3]

        ret.append([xs, ys, cs])
        prev_coords = [xs, ys, cs]
        prev_fid = fid

    return np.asarray(ret, dtype=np.float32)


def get_frames(data_dir: str) -> np.ndarray:
    frames = prepross_json(data_dir)

    stride = int(ORIGINAL_FPS / FPS)
    x = extract_consecutive_frames(frames, N_CONSECUTIVE_FRAMES, stride)
    x = np.moveaxis(x, 2, 1)  # x, y are separated channels now

    # only for printing
    fids = np.arange(0, len(frames), dtype=np.int)
    fids = extract_consecutive_frames(fids, N_CONSECUTIVE_FRAMES, stride)
    print(f'Frame indices {fids}')

    return np.asarray(x, dtype=np.float32)


def main():
    model_path = argv[1]
    image_dir = argv[2]
    x = get_frames(image_dir)
    x = torch.from_numpy(x)
    model = PoseCNN.load_from_checkpoint(model_path)
    model.eval()
    y_pred = model(x)
    pred = torch.argmax(y_pred, dim=-1)
    print(pred)


if __name__ == '__main__':
    main()
