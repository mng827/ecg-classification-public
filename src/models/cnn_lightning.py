import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning.pytorch as pl
from sklearn.metrics import confusion_matrix
from cnn import Conv1dModel


class LightningConv1dModel(pl.LightningModule):
    """ A PyTorch Lightning module encapsulating training
        and testing details for a model
    """
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.num_classes = cfg.model.num_classes

        self.model = Conv1dModel(cfg.model.input_channels,
                                 cfg.model.num_conv_blocks,
                                 cfg.model.hidden_features,
                                 cfg.model.kernel_size,
                                 cfg.model.num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(),
                                     lr=cfg.train.lr,
                                     weight_decay=cfg.train.weight_decay)

        self.train_confusion_matrix_meter = np.zeros((self.num_classes, self.num_classes))
        self.val_confusion_matrix_meter = np.zeros((self.num_classes, self.num_classes))
        self.test_confusion_matrix_meter = np.zeros((self.num_classes, self.num_classes))


    def training_step(self, batch, batch_idx):
        signal, label = batch
        pred = self.model(signal)

        loss = self.criterion(pred, label)
        _, pred_label = torch.max(pred.data, 1)

        label_np = label.cpu().numpy()
        pred_label_np = pred_label.cpu().numpy()
        self.train_confusion_matrix_meter += confusion_matrix(label_np, pred_label_np,
                                                              labels=np.arange(self.num_classes))

        self.log("train_loss", loss, on_epoch=True)

        return loss


    def on_train_epoch_end(self):
        average_recall = self.__calculate_avg_recall(self.train_confusion_matrix_meter)
        self.log("train_avg_recall", average_recall)

        self.train_confusion_matrix_meter = np.zeros((self.num_classes, self.num_classes))
        return


    def configure_optimizers(self):
        return self.optimizer


    def validation_step(self, batch, batch_idx):
        signal, label = batch
        pred = self.model(signal)

        loss = self.criterion(pred, label)
        _, pred_label = torch.max(pred.data, 1)

        label_np = label.cpu().numpy()
        pred_label_np = pred_label.cpu().numpy()
        self.val_confusion_matrix_meter += confusion_matrix(label_np, pred_label_np,
                                                            labels=np.arange(self.num_classes))

        self.log("val_loss", loss, on_epoch=True)
        return


    def on_validation_epoch_end(self):
        average_recall = self.__calculate_avg_recall(self.val_confusion_matrix_meter)
        self.log("val_avg_recall", average_recall)

        self.val_confusion_matrix_meter = np.zeros((self.num_classes, self.num_classes))
        return


    def test_step(self, batch, batch_idx):
        signal, label = batch
        pred = self.model(signal)
        _, pred_label = torch.max(pred.data, 1)

        label_np = label.cpu().numpy()
        pred_label_np = pred_label.cpu().numpy()
        self.test_confusion_matrix_meter += confusion_matrix(label_np, pred_label_np,
                                                            labels=np.arange(self.num_classes))
        return


    def on_test_epoch_end(self):
        average_recall = self.__calculate_avg_recall(self.test_confusion_matrix_meter)
        self.log("test_avg_recall", average_recall)

        self.test_confusion_matrix_meter = np.zeros((self.num_classes, self.num_classes))
        return


    def __calculate_avg_recall(self, confusion_matrix):
        recall_denominator = np.sum(confusion_matrix, axis=1)
        recall_numerator = np.diagonal(confusion_matrix)
        return np.mean(recall_numerator / recall_denominator)
