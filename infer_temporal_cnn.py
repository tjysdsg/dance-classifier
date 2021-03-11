from temporal_cnn import TemporalCNN
import torch
from skimage.io import imread
import os
import numpy as np
from sys import argv

from utils import extract_consecutive_frames

N_CONSECUTIVE_FRAMES = 16


def get_frames(data_dir: str) -> np.ndarray:
    fids: list = []
    paths = []
    for file in os.scandir(data_dir):
        filename = file.name
        fid = int(os.path.splitext(filename)[0])
        fids.append(fid)
        paths.append(file.path)

    idx_sorted = np.argsort(fids)
    paths = np.asarray(paths)[idx_sorted]
    paths = extract_consecutive_frames(paths, N_CONSECUTIVE_FRAMES, 1)

    # only for printing
    fids: np.ndarray = np.asarray(fids)[idx_sorted]
    fids = extract_consecutive_frames(fids, N_CONSECUTIVE_FRAMES, 1)
    print(f'Frame indices: {fids}')

    x = np.asarray(
        [
            [
                imread(f, as_gray=True) for f in row
            ]
            for row in paths
        ], dtype=np.float32
    )

    return x


def main():
    model_path = argv[1]
    image_dir = argv[2]
    x = get_frames(image_dir)
    x = torch.from_numpy(x)
    model = TemporalCNN.load_from_checkpoint(model_path)
    model.eval()
    y_pred = model(x)
    pred = torch.argmax(y_pred, dim=-1)
    print(pred)


if __name__ == '__main__':
    main()
