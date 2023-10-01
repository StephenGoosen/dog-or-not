import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import numpy as np


def accuracy_fn(y_true, y_pred):
    y_pred = torch.round(torch.sigmoid(y_pred))
    correct = torch.sum(torch.eq(y_pred, y_true))
    total_samples = len(y_true)
    acc = (correct / total_samples) * 100
    return acc

def calculate_precision(outputs, labels):
    predicted_labels = torch.round(torch.sigmoid(outputs))
    return precision_score(labels.cpu(), predicted_labels.cpu())

def calculate_recall(outputs, labels):
    predicted_labels = torch.round(torch.sigmoid(outputs))
    return recall_score(labels.cpu(), predicted_labels.cpu())

def calculate_f1_score(outputs, labels):
    predicted_labels = torch.round(torch.sigmoid(outputs))
    return f1_score(labels.cpu(), predicted_labels.cpu())

def calculate_confusion_matrix(outputs, labels):
    predicted_labels = torch.round(torch.sigmoid(outputs))
    return confusion_matrix(labels.cpu(), predicted_labels.cpu())

def train_step(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, 
               accuracy_fn=accuracy_fn, device=torch.device("cuda")):

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.unsqueeze(1).float().to(device)
        model.train()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%")
    return train_loss, train_acc

def val_step(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, 
             accuracy_fn=accuracy_fn, device=torch.device("cuda")):

    val_loss, val_acc = 0, 0

    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.unsqueeze(1).float().to(device)
            val_pred = model(X)
            val_loss += loss_fn(val_pred, y)
            val_acc += accuracy_fn(y_true=y, y_pred=val_pred)
    
        val_loss /= len(data_loader)
        val_acc /= len(data_loader)
    print(f"Test loss: {val_loss:.5f} | Test acc: {val_acc:.2f}%\n")
    return val_loss, val_acc