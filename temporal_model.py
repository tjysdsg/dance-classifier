from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from pytorch_lightning.metrics import ConfusionMatrix
from abc import abstractmethod
from torch.functional import F
import pytorch_lightning as pl


class TemporalModel(pl.LightningModule):
    def __init__(self, lr: float, num_classes: int, *args, **kwargs):
        super().__init__()
        self.lr = lr
        self.num_classes = num_classes
        self.train_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.confusion = ConfusionMatrix(self.num_classes)

        self.init_layers(*args, **kwargs)
        self.save_hyperparameters()

    @abstractmethod
    def init_layers(self, *args, **kwargs):
        pass

    @abstractmethod
    def forward(self, x):
        pass

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

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        val_loss = F.cross_entropy(y_pred, y)
        self.val_acc(y_pred, y)
        self.log('val_acc', self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        self.test_acc(y_pred, y)
        self.log('test_acc', self.test_acc, prog_bar=True, on_step=False, on_epoch=True)

        pred = torch.argmax(y_pred, -1)
        return dict(loss=loss, pred=pred, y=y)

    def test_epoch_end(self, outputs):
        pred = torch.cat([o['pred'] for o in outputs])
        y = torch.cat([o['y'] for o in outputs])

        print('Confusion matrix', self.confusion(pred, y))
