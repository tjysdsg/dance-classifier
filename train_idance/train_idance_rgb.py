from torch.utils.data import DataLoader, Dataset
from skimage.io import imread
import skimage
import json
import skimage.transform
import numpy as np
import os
from pytorch_lightning import Trainer
from temporal_datamodule import TemporalDataModule
from temporal_cnn import TemporalCNN
from train_idance.common import CATEGORY_TO_ID, CATEGORIES
from utils import extract_consecutive_frames

n_frames = 16
data_dir = 'idance-rgb'


class _Dataset(Dataset):
    def __init__(self, x, y, img_size=(224, 224)):
        self.files = x
        self.y = y.astype(np.int64)
        self.img_size = img_size

    def load_single_frame(self, path: str):
        img = imread(path, as_gray=True)
        img = skimage.transform.resize(img, self.img_size)
        img = np.asarray(img, dtype=np.float32)
        return img

    def __len__(self):
        return self.files.shape[0]

    def __getitem__(self, idx):
        files = self.files[idx]
        x = np.asarray([self.load_single_frame(f) for f in files], dtype=np.float32)
        return x, self.y[idx]


class _DataModule(TemporalDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, test_size=0.2, val_size=0.1,
                 n_consecutive_frames=6, fps=10, *args, **kwargs, ):
        self.vid2fps = json.load(open(os.path.join('idance-pose', 'fps.json')))
        self.fps = fps

        self.vid2label = {}
        self.vids = []  # video ids
        self.cat2vid = {}
        self.fid2path = {}
        self.vid2fid = {}
        super().__init__(data_dir=data_dir, n_consecutive_frames=n_consecutive_frames, batch_size=batch_size,
                         test_size=test_size, val_size=val_size, *args, **kwargs, )

    def get_unique_fid(self, vid, fid):
        return f'{vid}_{fid}'

    def read_data_dir(self):
        for cat_dir in os.scandir(self.data_dir):  # categories
            if cat_dir.is_dir():
                category = cat_dir.name

                vids = []
                for v in os.scandir(cat_dir.path):  # videos inside one category
                    filename: str = v.name
                    vid = os.path.splitext(filename)[0]

                    frames = []
                    for f in os.scandir(v.path):
                        fid = int(os.path.splitext(f.name)[0])
                        frames.append(fid)  # fids in `frames` are converted to unique strings later
                        self.fid2path[self.get_unique_fid(vid, fid)] = f.path

                    self.vid2fid[vid] = frames
                    self.vids.append(vid)
                    self.vid2label[vid] = category
                    vids.append(vid)

                self.cat2vid[category] = vids

        for vid in self.vids:  # sort frames by time, and add unique prefixes to fid
            self.vid2fid[vid] = sorted(self.vid2fid[vid])
            self.vid2fid[vid] = [self.get_unique_fid(vid, f) for f in self.vid2fid[vid]]

    def get_frames(self, video_idx: int) -> tuple:
        vid = self.vids[video_idx]
        fids = self.vid2fid[vid]  # fids is already sorted

        stride = int(self.vid2fps[vid] / self.fps)
        if stride < 1:
            raise RuntimeError(f"FPS of video {vid} is too low: {self.vid2fps[vid]}")

        # ASSUMING a video contains more than `n_consecutive_frames` frames
        if len(fids) < self.n_consecutive_frames * stride:
            print(fids)
            raise RuntimeError(
                f"Video {vid} in category {self.vid2label[vid]} "
                f"contains too few frames: {len(fids)}"
            )

        fid2d = extract_consecutive_frames(fids, self.n_consecutive_frames, stride)
        x = np.asarray(
            [
                [
                    self.fid2path[f] for f in row
                ]
                for row in fid2d
            ]
        )

        label = CATEGORY_TO_ID[self.vid2label[vid]]
        y = np.full(x.shape[0], label)
        return x, y

    def init_dataset(self, x, y):
        return _Dataset(x, y)


def train():
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        patience=10,
        verbose=True,
        mode='max'
    )
    checkpoint_callback = ModelCheckpoint(
        save_top_k=True,
        verbose=True,
        monitor='val_acc',
        mode='max'
    )

    model = TemporalCNN(num_classes=len(CATEGORIES), n_consecutive_frames=n_frames, lr=0.001)
    trainer = Trainer(
        gpus=1,
        max_epochs=1, callbacks=[checkpoint_callback, early_stop_callback],
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=_DataModule(data_dir, n_consecutive_frames=n_frames))
    trainer.test()


def test(model_path):
    trainer = Trainer(gpus=1)

    model = TemporalCNN.load_from_checkpoint(model_path)

    dm = _DataModule(data_dir, n_consecutive_frames=n_frames)
    trainer.test(model, datamodule=dm)


def main():
    from sys import argv
    action = argv[1]

    if action == 'train':
        train()
    elif action == 'test':
        test(argv[2])
    else:
        raise RuntimeError(f"Unknown action {action}")


if __name__ == '__main__':
    main()
