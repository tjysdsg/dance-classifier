from pose_cnn import PoseCNN
from sys import argv
from utils import extract_consecutive_frames
from lstm import LSTM_RNN
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from pytorch_lightning import Trainer
from train_idance.common import CATEGORY_TO_ID, CATEGORIES, KEYPOINTS
from temporal_datamodule import TemporalDataModule
import json
import pickle

USE_RNN = False
n_frames = 16


class _Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class _DataModule(TemporalDataModule):
    def __init__(
            self,
            data_dir: str,
            n_consecutive_frames: int,
            batch_size: int = 32, test_size=0.2, val_size=0.1,
            fps=10, *args, **kwargs,
    ):
        self.fps = fps
        self.vid2fps = json.load(open(os.path.join(data_dir, 'fps.json')))
        self.vid2label = {}
        self.cat2vid = {}
        self.vid2frames = {}

        super().__init__(data_dir=data_dir, n_consecutive_frames=n_consecutive_frames, batch_size=batch_size,
                         test_size=test_size, val_size=val_size, *args, **kwargs)

    def init_dataset(self, x, y):
        return _Dataset(x, y)

    def read_data_dir(self):
        for f in os.scandir(self.data_dir):
            if not f.is_file():
                continue
            filename = f.name
            file = os.path.join(self.data_dir, filename)

            cat = filename.split('.')[0]
            if cat not in CATEGORIES:
                continue

            data = json.load(open(file, 'r'))
            self.cat2vid[cat] = []
            for vid, frames in data.items():
                if len(frames) >= self.n_consecutive_frames:
                    self.vids.append(vid)
                    self.vid2label[vid] = cat
                    self.cat2vid[cat].append(vid)
                    self.vid2frames[vid] = frames

    def get_frames(self, video_idx: int):
        vid = self.vids[video_idx]
        frames = self.vid2frames[vid]

        stride = int(self.vid2fps[vid] / self.fps)
        if stride < 1:
            raise RuntimeError(f"FPS of video {vid} is too low: {self.vid2fps[vid]}")

        # ASSUMING a video contains more than `n_consecutive_frames` frames
        if len(frames) < self.n_consecutive_frames * stride:
            print(frames)
            raise RuntimeError(
                f"Video {vid} in category {self.vid2label[vid]} "
                f"contains too few frames: {len(frames)}"
            )

        fids = sorted([int(k) for k in frames.keys()])
        label = CATEGORY_TO_ID[self.vid2label[vid]]

        fid2d = extract_consecutive_frames(fids, self.n_consecutive_frames, stride)
        x = np.asarray([
            [
                np.asarray(frames[str(f)])[:, KEYPOINTS] for f in row
            ] for row in fid2d
        ])  # coordinates each frame
        # delta = np.diff(x, axis=0, prepend=np.expand_dims(x[0], 0))
        # x = np.concatenate([x, delta], axis=-1)

        if USE_RNN:
            # concatenate xs and ys
            shape = x.shape
            x = x.reshape((shape[0], shape[1], -1))
        else:
            x = np.moveaxis(x, 2, 1)  # x, y are separated channels now

        y = np.full(x.shape[0], label)
        return x, y


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

    model = PoseCNN(
        num_classes=len(CATEGORY_TO_ID), n_keypoints=len(KEYPOINTS), n_consecutive_frames=n_frames,
        n_channels=2
    )
    trainer = Trainer(
        gpus=1, max_epochs=200, callbacks=[checkpoint_callback, early_stop_callback],
        num_sanity_val_steps=0,
    )
    dm = _DataModule('idance-pose', n_consecutive_frames=n_frames)

    trainer.fit(model, datamodule=dm)

    pickle.dump(model, open('idance-posecnn-tmp.ckpt', 'wb'))

    trainer.test()


def test(model_path):
    trainer = Trainer(gpus=1)

    model = PoseCNN.load_from_checkpoint(model_path)

    dm = _DataModule('idance-pose', n_consecutive_frames=n_frames)
    trainer.test(model, datamodule=dm)


def main():
    action = argv[1]

    if action == 'train':
        train()
    elif action == 'test':
        test(argv[2])
    else:
        raise RuntimeError(f"Unknown action {action}")


if __name__ == '__main__':
    main()
