import numpy as np
from torch.utils.data import Dataset
from skimage.io import imread
import skimage
from random import Random
import skimage.transform


# FIXME: possible duplicated samples in a batch
class TemporalDataset(Dataset):
    def __init__(
            self, vids: list, vid2fid: dict, vid2label: dict, fid2path: dict,
            n_consecutive_frames: int, category2id: dict, stride=1, gray=False,
            transform=None, img_size=(224, 224), preprocessed=False,
    ):
        self.vids = vids
        np.random.shuffle(self.vids)
        self.vid2fid = vid2fid
        self.fid2path = fid2path
        self.vid2label = vid2label
        self.transform = transform
        self.n_consecutive_frames = n_consecutive_frames
        self.img_size = img_size
        self.category2id = category2id
        self.rand = Random()
        self.preprocessed = preprocessed
        self.gray = gray
        self.stride = stride

    def __len__(self):
        return len(self.vids)

    def load_single_frame(self, path: str):
        from matplotlib.colors import Normalize
        img = imread(path, as_gray=self.gray)
        if not self.preprocessed:
            img = Normalize()(img)
            img = skimage.transform.resize(img, self.img_size)
            if self.transform is not None:
                img = self.transform(img)
            img = np.asarray(img, dtype=np.float32)
        else:
            img = skimage.img_as_float32(img)

        # from matplotlib import pyplot as plt
        # plt.imshow(img, cmap='gray')
        # plt.axis('off')
        # plt.show()
        return img

    def __getitem__(self, idx):
        vid = self.vids[idx]
        fids = self.vid2fid[vid]

        # randomly choose n_consecutive frames in the video
        # ASSUMING a video contains more than `n_consecutive_frames` frames
        if len(fids) < self.n_consecutive_frames * self.stride:
            print(fids)
            raise RuntimeError(
                f"Video {vid} in category {self.vid2label[vid]} "
                f"contains too few frames: {len(fids) // self.stride}"
            )
        start = self.rand.randint(0, len(fids) - self.n_consecutive_frames * self.stride)

        frames = fids[start: start + self.stride * self.n_consecutive_frames: self.stride]
        files = np.asarray([self.fid2path[fid] for fid in frames])
        x = np.asarray([self.load_single_frame(f) for f in files])

        label = self.vid2label[vid]

        # print(f'label {label} files {", ".join(files)}')

        y = self.category2id[label]

        # below for ensuring there's no mismatched label
        """
        for f in frames:
            path = self.fid2path[f]
            if label not in path:
                raise RuntimeError(f"Label {label}, vid {vid}, fid {f}, path {path}")
        """

        return x, y
