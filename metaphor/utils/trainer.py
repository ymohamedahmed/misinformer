
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import numpy as np


class ClassifierTrainer:
    def __init__(self, lr: float, patience: int, weight_decay:float, num_epochs:int, device:str, loss:nn.Module):
        self.lr = lr
        self.patience = patience
        self.wd = weight_decay
        self.epochs = num_epochs
        self.device = device
        self.loss = loss

    def fit(self, model: nn.Module, train: DataLoader, validation: DataLoader):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wd)
        train_acc, train_loss = [], []
        val_acc, val_loss = [], []

        for epoch in range(self.epochs):
            N, mean_loss, mean_acc = 0, 0, 0
            for x, y in train:
                optimizer.zero_grad()
                ind = x[0].to(self.device)
                emb = x[1].to(self.device)
                y = y.to(self.device)
                logits = model.forward(emb, ind)
                loss = self.loss(logits, y)
                loss.backward()
                mean_loss += loss
                mean_acc += ClassifierTrainer._acc(logits, y)
                optimizer.step()
                N += 1
            train_acc.append(mean_acc/N)
            train_loss.append(mean_loss/N)

            # evaluate on validation set
            vacc, vloss = self._evaluate_validation(model, validation)
            val_acc.append(vacc)
            val_loss.append(vloss)
            if epoch - np.argmin(np.array(val_loss)) > self.patience:
                break
        return {'train_accuracy': train_acc, 'train_loss': train_loss,
                'validation_accuracy': val_acc, 'validation_loss': val_loss}

    @staticmethod
    def _acc(logits: torch.FloatTensor, y: torch.IntTensor):
        preds = torch.argmax(logits, dim=1)
        return torch.mean(1.*(preds == y)).item()

    def _evaluate_validation(self, model: nn.Module, validation: DataLoader):
        N, mean_acc, mean_loss = 0, 0, 0
        for x, y in validation:
            ind = x[0].to(self.device)
            emb = x[1].to(self.device)
            y = y.to(self.device)
            logits = model.forward(emb, ind)
            acc = ClassifierTrainer._acc(logits, y)
            loss = self.loss(logits, y)
            mean_acc += acc
            mean_loss += loss
            N += 1
        return mean_acc/N, mean_loss/N
