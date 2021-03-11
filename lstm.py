from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from torch import nn
from torch.functional import F
import pytorch_lightning as pl


class LSTM_RNN(pl.LightningModule):
    def __init__(self, n_keypoints: int, n_consecutive_frames: int, num_classes: int):
        super().__init__()
        self.lr = 0.0001
        self.train_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()

        hidden_size = 256
        self.lstm = nn.LSTM(input_size=n_keypoints, hidden_size=hidden_size, num_layers=2, batch_first=True,
                            dropout=0.2)
        self.fc1 = nn.Linear(hidden_size * n_consecutive_frames, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x, _ = self.lstm(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, -1)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), self.lr, momentum=0.9)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, verbose=True)
        return dict(optimizer=optimizer, lr_scheduler=scheduler, monitor='val_acc')

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        train_acc_batch = self.train_acc(y_pred, y)
        self.log('train_acc_batch', train_acc_batch, prog_bar=True, on_step=True)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        self.test_acc(y_pred, y)
        self.log('test_acc', self.test_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        val_loss = F.cross_entropy(y_pred, y)
        self.val_acc(y_pred, y)
        self.log('val_acc', self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        return val_loss
