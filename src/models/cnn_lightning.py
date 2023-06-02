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
        recall_per_class = self.__calculate_recall_per_class(self.train_confusion_matrix_meter)
        precision_per_class = self.__calculate_precision_per_class(self.train_confusion_matrix_meter)
        f1score_per_class = self.__calculate_f1score_per_class(self.train_confusion_matrix_meter)
        accuracy = self.__calculate_accuracy(self.train_confusion_matrix_meter)

        self.log("train_avg_recall", np.nanmean(recall_per_class))
        self.log("train_avg_precision", np.nanmean(precision_per_class))
        self.log("train_avg_f1score", np.nanmean(f1score_per_class))
        self.log("train_accuracy", accuracy)
        self.logger.experiment.log_text(self.logger.run_id,
                                        str(self.train_confusion_matrix_meter.tolist()),
                                        "train_confusion_matrix.txt")

        self.train_confusion_matrix_meter = np.zeros((self.num_classes, self.num_classes))
        return


    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', cooldown=5),
                "monitor": "val_loss",
            }
        }


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
        recall_per_class = self.__calculate_recall_per_class(self.val_confusion_matrix_meter)
        precision_per_class = self.__calculate_precision_per_class(self.val_confusion_matrix_meter)
        f1score_per_class = self.__calculate_f1score_per_class(self.val_confusion_matrix_meter)
        accuracy = self.__calculate_accuracy(self.val_confusion_matrix_meter)

        self.log("val_avg_recall", np.nanmean(recall_per_class))
        self.log("val_avg_precision", np.nanmean(precision_per_class))
        self.log("val_avg_f1score", np.nanmean(f1score_per_class))
        self.log("val_accuracy", accuracy)
        self.logger.experiment.log_text(self.logger.run_id,
                                        str(self.val_confusion_matrix_meter.tolist()),
                                        "val_confusion_matrix.txt")

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
        recall_per_class = self.__calculate_recall_per_class(self.test_confusion_matrix_meter)
        precision_per_class = self.__calculate_precision_per_class(self.test_confusion_matrix_meter)
        f1score_per_class = self.__calculate_f1score_per_class(self.test_confusion_matrix_meter)
        accuracy = self.__calculate_accuracy(self.test_confusion_matrix_meter)
        for i in range(self.num_classes):
            self.log(f"test_recall_class{i}", recall_per_class[i])
            self.log(f"test_precision_class{i}", precision_per_class[i])
            self.log(f"test_f1score_class{i}", f1score_per_class[i])

        self.log(f"test_avg_recall", np.nanmean(recall_per_class))
        self.log(f"test_avg_precision", np.nanmean(precision_per_class))
        self.log(f"test_avg_f1score", np.nanmean(f1score_per_class))
        self.log(f"test_accuracy", accuracy)
        self.logger.experiment.log_text(self.logger.run_id,
                                        str(self.test_confusion_matrix_meter.tolist()),
                                        "test_confusion_matrix.txt")

        self.test_confusion_matrix_meter = np.zeros((self.num_classes, self.num_classes))
        return


    def __calculate_recall_per_class(self, confusion_matrix):
        recall_denominator = np.sum(confusion_matrix, axis=1)
        recall_numerator = np.diagonal(confusion_matrix)
        return recall_numerator / recall_denominator


    def __calculate_precision_per_class(self, confusion_matrix):
        precision_denominator = np.sum(confusion_matrix, axis=0)
        precision_numerator = np.diagonal(confusion_matrix)
        return precision_numerator / precision_denominator


    def __calculate_f1score_per_class(self, confusion_matrix):
        positive_pred = np.sum(confusion_matrix, axis=0)
        positive_label = np.sum(confusion_matrix, axis=1)
        true_positive = np.diagonal(confusion_matrix)

        return 2.0 * true_positive / (positive_pred + positive_label)

    def __calculate_accuracy(self, confusion_matrix):
        correct = np.sum(np.diagonal(confusion_matrix))
        total = np.sum(confusion_matrix)
        return correct / total
