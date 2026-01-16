# utils.py
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


def load_data(
    train_path="train.csv",
    test_path="test.csv",
    label_col="label",
    id_col="id",
    seed=42,
):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_cols = [c for c in train_df.columns if c not in (label_col, id_col)]

    X = train_df[feature_cols].to_numpy()
    y = train_df[label_col].to_numpy()

    X_test = test_df[feature_cols].to_numpy()
    test_ids = test_df[id_col].to_numpy() if id_col in test_df.columns else None

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, y_train, y_val, X_test, test_ids


def make_loaders(X_train, X_val, y_train, y_val, X_test, batch_size=64):
    train_loader = DataLoader(
        TabularDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TabularDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TabularDataset(X_test),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader


def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy(logits, y)
        n += 1

    return total_loss / n, total_acc / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        logits = model(X)
        loss = criterion(logits, y)

        total_loss += loss.item()
        total_acc += accuracy(logits, y)
        n += 1

    return total_loss / n, total_acc / n


def fit(model, train_loader, val_loader, epochs=30, lr=1e-3, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for _ in range(epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = eval_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

    return model, history


@torch.no_grad()
def predict(model, test_loader, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    preds = []
    for X in test_loader:
        X = X.to(device)
        logits = model(X)
        preds.append(logits.argmax(dim=1).cpu().numpy())

    return np.concatenate(preds)


def make_submission(
    preds,
    sample_path="sample_submission.csv",
    out_path="submission.csv",
    label_col="label",
):
    sub = pd.read_csv(sample_path)
    sub[label_col] = preds
    sub.to_csv(out_path, index=False)

def load_full_train_and_test(
    train_path="train.csv",
    test_path="test.csv",
    label_col="label",
    id_col="id",
):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_cols = [c for c in train_df.columns if c not in (label_col, id_col)]

    X_full = train_df[feature_cols].to_numpy()
    y_full = train_df[label_col].to_numpy()
    X_test = test_df[feature_cols].to_numpy()

    scaler = StandardScaler()
    X_full = scaler.fit_transform(X_full)
    X_test = scaler.transform(X_test)

    return X_full, y_full, X_test
