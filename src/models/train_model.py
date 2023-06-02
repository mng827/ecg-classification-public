import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from data.dataset import EcgDataset
from data.transforms import GaussianNoise, RandomMask, BaselineWander
import matplotlib.pyplot as plt
from cnn_lightning import LightningConv1dModel
import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import EarlyStopping
import mlflow
import hydra
import logging
import subprocess


logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    mlf_logger = MLFlowLogger(experiment_name="ecg_classification",
                              tracking_uri="./mlruns",
                              log_model=True)
    try:
        git_commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        git_diff = subprocess.check_output(['git', 'diff', 'HEAD']).decode('ascii').strip()

        mlf_logger.experiment.log_param("git_version", git_commit)
        mlf_logger.experiment.log_param("git_changes", git_diff)
    except:
        logger.error("Error getting git info")

    train_df = pd.read_csv(cfg.data.train_csv, header=None)
    test_df = pd.read_csv(cfg.data.test_csv, header=None)

    train_df, val_df = train_test_split(
        train_df, test_size=0.2, stratify=train_df.iloc[:,-1]
    )

    train_transforms = [
        GaussianNoise(magnitude_range=(0, 0.02), prob=0.5),
        BaselineWander(magnitude_range=(0, 0.05), prob=0.5)
    ]

    train_dataset = EcgDataset(train_df, transform=train_transforms)
    val_dataset = EcgDataset(val_df, transform=None)
    test_dataset = EcgDataset(test_df, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False)

    sample_signal, sample_label = next(iter(train_loader))
    print(sample_signal.shape, sample_label.shape)
    classes = ["Normal", "Supraventricular", "Ventricular", "Fusion", "Unknown"]
    for i in range(sample_signal.shape[0]):
        plt.clf()
        plt.plot(sample_signal[i,0])
        plt.title(classes[sample_label[i]])
        plt.savefig(f"{i:03d}.png")

    lightning_module = LightningConv1dModel(cfg)
    lightning_module.save_hyperparameters(cfg)

    early_stopping = EarlyStopping('val_avg_recall', min_delta=0.01, patience=20, mode='max')

    trainer = pl.Trainer(limit_train_batches=100,
                         max_epochs=cfg.train.num_epochs,
                         logger=mlf_logger,
                         deterministic=True,
                         callbacks=[early_stopping])

    trainer.fit(model=lightning_module,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    trainer.test(model=lightning_module, dataloaders=test_loader)


if __name__ == "__main__":
    main()
