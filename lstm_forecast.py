# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 10:33:22 2021

@author: abdul
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch import optim

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


df = pd.read_csv('coinbase_BTCUSD, 1D.csv',skiprows=0)
df.time = pd.to_datetime(df.time, unit='s')
df['week_day'] = df.time.dt.weekday
df['month'] = df.time.dt.month
df.drop('Shapes',axis=1,inplace=True)
df = df[14:]
features = ['open','high','low','close','RSI','Volume','week_day','month']
#features = df.columns[1:]

train_size = int(df.shape[0] * .8)
train_df, test_df = df.iloc[:train_size][features], df.iloc[train_size:][features].reset_index(drop=True)
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(train_df)

train_df = pd.DataFrame(
    scaler.transform(train_df),
    index = train_df.index,
    columns = train_df.columns
    )

test_df = pd.DataFrame(
    scaler.transform(test_df),
    index = test_df.index,
    columns = test_df.columns
    )

def create_sequences(input_data, target_column, sequence_length):
    
    sequences = []
    data_size = len(input_data)
    
    for i in range(data_size - sequence_length):
        
        sequence = input_data[i:i+sequence_length]
        
        label_position = i + sequence_length
        label = input_data.iloc[label_position][target_column]
        
        sequences.append((sequence, label))
    return sequences

seq_length = 7
target = "close"

train_sequences = create_sequences(train_df, target, seq_length)
test_sequences = create_sequences(test_df, target, seq_length)

class CryptoDataset(Dataset):
    
    def __init__(self, sequences):
        self.sequences = sequences
        
    def __len__(self):
        return (len(self.sequences))
    
    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]
        return dict(
            sequence = torch.Tensor(sequence.to_numpy()),
            label = torch.tensor(label).float()
            )
    

class CryptoDataModule(pl.LightningDataModule):
    def __init__(self, train_sequences, test_sequences, batch_size = 8):
        super().__init__()
        self.train_sequences = train_sequences
        self.test_sequences = test_sequences
        self.batch_size = batch_size
        
    def setup(self):
        self.train_dataset = CryptoDataset(train_sequences)
        self.test_dataset = CryptoDataset(test_sequences)
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0)
    
    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0)
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0)
        
        
epochs = 356
batch_size = 32
data_module = CryptoDataModule(train_sequences, test_sequences, batch_size)
data_module.setup()

class PricePredictionModel(nn.Module):
    
    def __init__(self, n_features, n_hidden = 256, n_layers = 2):
        super().__init__()
        self.n_hidden = n_hidden
        
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            batch_first=True,
            num_layers=n_layers,
            dropout=0.2,
            bidirectional=True
            )
        
        self.linear = nn.Linear(n_hidden, 1)
        
    def forward(self, x):
        
        self.lstm.flatten_parameters()
        
        _, (hidden, _) = self.lstm(x)
        out = hidden[-1]
        
        return self.linear(out)
    
class CryptoPricePredictor(pl.LightningModule):
    
    def __init__(self, n_features, n_hidden = 128, n_layers = 2, lr = 1e-4):
        super().__init__()
        self.model = PricePredictionModel(n_features, n_hidden, n_layers)
        self.criterion = nn.MSELoss()
        self.lr = lr
    def forward(self, x, labels = None):
        output = self.model(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels.unsqueeze(dim=1))
        return loss, output
    
    def training_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        loss, outputs = self(sequences, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        loss, outputs = self(sequences, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        loss, outputs = self(sequences, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr = self.lr)
    
model = CryptoPricePredictor(n_features=train_df.shape[1])

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    verbose=True,
    monitor = "val_loss",
    mode = "min")

logger = TensorBoardLogger("lightning_logs", name="btc-price-day")

early_stopping_callback = EarlyStopping(monitor="val_loss", patience = 2)

trainer = pl.Trainer(
    logger = logger,
    checkpoint_callback = checkpoint_callback,
    callbacks = [early_stopping_callback],
    max_epochs = epochs,
    gpus=1,
    enable_progress_bar  = True
    
    )

trainer.fit(model, data_module)

#trained_model = CryptoPricePredictor.load_from_checkpoint('C:/Users/abdul/Python Scripts/time_series/lightning_logs/btc-price-day/version_1/checkpoints/epoch=140-step=4088.ckpt', 
 #                                                         n_features = train_df.shape[1])
trained_model = model
test_dataset = CryptoDataset(test_sequences)

predictions = []
labels = []
trained_model.freeze()

for item in test_dataset:
    sequence = item["sequence"]
    label = item["label"]
    
    _, output = trained_model(sequence.unsqueeze(dim=0))
    predictions.append(output.item())
    labels.append(label.item())
    
descaler = MinMaxScaler()
descaler.min_, descaler.scale_ = scaler.min_[3], scaler.scale_[3]
    
def descale(descaler, values):
    values_2d = np.array(values)[:, np.newaxis]
    return descaler.inverse_transform(values_2d).flatten()

predictions_descaled = descale(descaler, predictions)
labels_descaled = descale(descaler, labels)

import matplotlib
dates = matplotlib.dates.date2num(df.iloc[train_size:-seq_length,0].tolist())

plt.figure(figsize=(24,8))
plt.plot_date(dates, predictions_descaled, "-", label = "predictions",linewidth=2)
plt.plot_date(dates, labels_descaled, "-", label = "real")
plt.xticks(rotation=45)
plt.legend()
plt.title("btc-close price")
plt.show()