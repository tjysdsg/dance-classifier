import os
from preprocessor import DancePreprocessor
from train_idance.common import CATEGORIES
from utils import write_video_frames


# FIXME: currently only using one person per frame
class IDanceRGBPreprocessor(DancePreprocessor):
    def preprocess_category(self, category: str):
        cat_dir = os.path.join(self.data_dir, category)
        cat_outdir = os.path.join(self.out_dir, category)
        for v in os.scandir(cat_dir):
            filename = v.name
            filename = filename.split('.')[0]
            video_path = v.path

            output_dir = os.path.join(cat_outdir, filename)
            os.makedirs(output_dir, exist_ok=True)

            write_video_frames(video_path, output_dir, img_size=(512, 512))


def main():
    data_dir = os.path.join('idance')
    out_dir = 'idance-rgb'

    preprocessor = IDanceRGBPreprocessor(data_dir, out_dir, CATEGORIES)
    preprocessor.run()


if __name__ == '__main__':
    main()
