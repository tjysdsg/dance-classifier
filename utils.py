import cv2
from typing import Iterable
import os


def write_video_frames(video_path: str, output_dir: str, fps=10, img_size=(224, 224)):
    vidcap = cv2.VideoCapture(video_path)
    step = 1000 / fps

    success, image = vidcap.read()
    count = -1
    while success:
        count += 1
        output_path = os.path.join(output_dir, f"{count}.jpg")
        if os.path.exists(output_path):
            print(f'Skipping {output_path}')
            continue

        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * step))

        image = cv2.resize(image, img_size)
        cv2.imwrite(output_path, image)

        success, image = vidcap.read()


# FIXME: frames is not used if the number of them is smaller than n_consecutive_frames
def extract_consecutive_frames(seq: Iterable, n_consecutive_frames: int, stride: int):
    ret = []
    window_size = n_consecutive_frames * stride
    max_offset = len(seq) // window_size - 1
    max_offset = int(max_offset * window_size)
    for offset in range(0, max_offset + 1, window_size):
        for s in range(offset, offset + stride):
            ret.append(seq[s: s + window_size: stride])

    return ret
