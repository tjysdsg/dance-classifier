import os
import skimage
import numpy as np
from skimage.io import imread, imsave
import skimage.transform
from multiprocessing import Process
from train_letsdance.common import CATEGORIES


def load_single_frame(path: str, img_size=(224, 224)):
    img = imread(path)
    # img = Normalize()(img)
    img = skimage.transform.resize(img, img_size)
    img = np.asarray(img, dtype=np.float32)
    return img


def preprocess_category(category: str, cat_dir: str, cat_out_dir: str):
    if category not in CATEGORIES:
        return

    for f in os.scandir(cat_dir):  # video frames inside one category
        filename: str = f.name
        out_filename = f'{os.path.splitext(filename)[0]}.jpg'
        output_path = os.path.join(cat_out_dir, out_filename)

        if os.path.exists(output_path):
            print(f"Skipping {output_path}")
            continue

        img = load_single_frame(f.path)
        img = skimage.img_as_ubyte(img)

        imsave(output_path, img)
        # np.save(output_path, img, allow_pickle=False)
    print(f'Category {category} DONE')


def main():
    out_dir = 'letsdance-preprocessed'
    data_dir = 'letsdance/rgb'

    procs = []
    for cat_dir in os.scandir(data_dir):  # categories
        if cat_dir.is_dir():
            category = cat_dir.name
            cat_out_dir = os.path.join(out_dir, category)
            os.makedirs(cat_out_dir, exist_ok=True)

            procs.append(Process(target=preprocess_category, args=(category, cat_dir.path, cat_out_dir)))

    for p in procs:
        p.start()

    for p in procs:
        p.join()


if __name__ == '__main__':
    main()
