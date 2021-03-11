import os
import numpy as np
from preprocessor import DancePreprocessor
from train_idance.common import CATEGORIES, get_vid
import json

WITH_CONFIDENCE = False


# FIXME: currently only using one person per frame
class IDancePosePreprocessor(DancePreprocessor):
    def preprocess_category(self, category: str):
        cat_dir = os.path.join(self.data_dir, category)
        all_data = {}  # vid -> frames -> keypoints
        for v in os.scandir(cat_dir):
            vid = get_vid(v.name)
            vid_data = {}

            fids = []
            frame_path = []
            for f in os.scandir(v.path):
                path = f.path
                name = f.name
                fid = int(name.split('_')[2])
                fids.append(fid)
                frame_path.append(path)

            idx_sort = np.argsort(fids)
            fids = np.asarray(fids)[idx_sort].tolist()
            frame_path = np.asarray(frame_path)[idx_sort].tolist()

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
                        vid_data[j] = prev_coords

                # if a frame doesn't contain a person, use coordinates from the previous frame
                if len(data) == 0:
                    print(f"WARNING: {path} doesn't contain any person")
                    if prev_coords is not None:
                        vid_data[fid] = prev_coords
                    continue

                data = data[0]['pose_keypoints_2d']
                xs = data[::3]
                ys = data[1::3]
                cs = data[2::3]

                if WITH_CONFIDENCE:
                    vid_data[fid] = [xs, ys, cs]
                    prev_coords = [xs, ys, cs]
                else:
                    vid_data[fid] = [xs, ys]
                    prev_coords = [xs, ys]

                prev_fid = fid

            all_data[vid] = vid_data

        json.dump(all_data, open(os.path.join(self.out_dir, f'{category}.json'), 'w'))


def main():
    data_dir = os.path.join('idance-pose')
    out_dir = 'idance-pose'

    preprocessor = IDancePosePreprocessor(data_dir, out_dir, CATEGORIES)
    preprocessor.run()


if __name__ == '__main__':
    main()
