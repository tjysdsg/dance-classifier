from abc import abstractmethod
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset


class TemporalDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            n_consecutive_frames: int,
            batch_size: int = 32, test_size=0.2, val_size=0.1,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.n_consecutive_frames = n_consecutive_frames

        self.vids = []
        self.x = []
        self.y = []
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_val = None
        self.y_val = None
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None

        # actual data reading
        self.read_data_dir()
        self.combine_frames()
        self.train_test_split()
        self.init_dataloaders()

        # print number of samples in each categories in each subset
        for subset in ['train', 'test', 'val']:
            labels = getattr(self, f'y_{subset}')

            if labels is None:
                print(f'No {subset} set')
                continue

            unique_labels, counts = np.unique(labels, return_counts=True)
            for i, label in enumerate(unique_labels):
                print(f'Number of {label} in {subset} set is: {counts[i]}')

    def combine_frames(self):
        for i in range(len(self.vids)):
            x, y = self.get_frames(i)
            self.x.append(x)
            self.y.append(y)

        self.x = np.vstack(self.x)
        self.y = np.hstack(self.y)
        print("X size:", self.x.shape)
        print("y size:", self.y.shape)

    def train_test_split(self):
        from sklearn.model_selection import train_test_split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y,
            test_size=self.test_size,
            random_state=1024
        )
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.x_train, self.y_train,
            test_size=self.val_size,
            random_state=1024
        )

    def init_dataloaders(self):
        self.train_loader = DataLoader(
            self.init_dataset(self.x_train, self.y_train), batch_size=self.batch_size, num_workers=8
        )
        self.test_loader = DataLoader(
            self.init_dataset(self.x_test, self.y_test), batch_size=self.batch_size, num_workers=8
        )
        self.val_loader = DataLoader(
            self.init_dataset(self.x_val, self.y_val), batch_size=self.batch_size, num_workers=8
        )

    @abstractmethod
    def init_dataset(self, x, y) -> Dataset:
        pass

    @abstractmethod
    def read_data_dir(self):
        pass

    @abstractmethod
    def get_frames(self, video_idx: int) -> tuple:
        pass

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.test_loader

    def test_dataloader(self):
        return self.val_loader
