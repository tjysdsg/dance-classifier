from pose_cnn import PoseCNN
import os
from temporal_datamodule import TemporalDataModule
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import Trainer
import json
from train_letsdance.common import CATEGORY_TO_ID, KEYPOINTS
import numpy as np
from utils import extract_consecutive_frames

ORIGINAL_FPS = 30
n_frames = 16
data_dir = 'letsdance-densepose'


class _Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class _DataModule(TemporalDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, test_size=0.2, val_size=0.1,
                 n_consecutive_frames=6, fps=30, *args, **kwargs, ):
        self.vid2label = {}
        self.vids = []
        self.cat2vid = {}
        self.vid2frames = {}
        self.fps = fps

        super().__init__(data_dir=data_dir, n_consecutive_frames=n_consecutive_frames, batch_size=batch_size,
                         test_size=test_size, val_size=val_size, *args, **kwargs, )

    def read_data_dir(self):
        for f in os.scandir(self.data_dir):
            if not f.is_file():
                continue
            filename = f.name
            file = os.path.join(self.data_dir, filename)

            cat = filename.split('.')[0]
            data = json.load(open(file, 'r'))
            for vid, frames in data.items():
                if len(frames) >= self.n_consecutive_frames:
                    if cat not in self.cat2vid:
                        self.cat2vid[cat] = [vid]
                    else:
                        self.cat2vid[cat].append(vid)
                    self.vids.append(vid)
                    self.vid2label[vid] = cat
                    self.vid2frames[vid] = frames

    def get_frames(self, video_idx: int):
        vid = self.vids[video_idx]
        frames = self.vid2frames[vid]

        stride = int(ORIGINAL_FPS / self.fps)

        # ASSUMING a video contains more than `n_consecutive_frames` frames
        if len(frames) < self.n_consecutive_frames * stride:
            print(frames)
            raise RuntimeError(
                f"Video {vid} in category {self.vid2label[vid]} "
                f"contains too few frames: {len(frames)}"
            )

        fids = sorted([int(k) for k in frames.keys()])
        fid2d = extract_consecutive_frames(fids, self.n_consecutive_frames, stride)
        x = np.asarray([
            [
                np.asarray(frames[str(f)]) for f in row
            ] for row in fid2d
        ])  # coordinates each frame
        x = np.moveaxis(x, 2, 1)  # x, y are separated channels now

        label = CATEGORY_TO_ID[self.vid2label[vid]]
        y = np.full(x.shape[0], label)
        return x.astype(np.float32), y.astype(np.int64)

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

    model = PoseCNN(
        num_classes=len(CATEGORY_TO_ID), n_keypoints=len(KEYPOINTS), n_consecutive_frames=n_frames,
        n_channels=2,
    )
    trainer = Trainer(
        gpus=1, max_epochs=200,
        callbacks=[checkpoint_callback, early_stop_callback],
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=_DataModule(data_dir, n_consecutive_frames=n_frames))
    trainer.test()


def test(model_path):
    trainer = Trainer(gpus=1)

    model = PoseCNN.load_from_checkpoint(model_path)

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
