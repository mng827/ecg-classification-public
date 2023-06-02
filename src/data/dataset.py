import numpy as np
import torch


class EcgDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df=df
        self.transform=transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        signal = self.df.iloc[idx, :-1].astype(float)

        if self.transform:
            for trans in self.transform:
                signal = trans(signal)

        signal = np.expand_dims(signal, axis=0)

        label = self.df.iloc[idx, -1].astype(int)

        signal = torch.tensor(signal).float()
        label = torch.tensor(label).long()

        return signal, label
