import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from model_l1 import CNNLSTM
from data_preparation_l1 import prepare_l1_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

positive_samples_dir = "./data/pos_combined"
negative_samples_dir = "./data/negative_combined"
X_train, X_test, y_train, y_test = prepare_l1_data(positive_samples_dir, negative_samples_dir)

def preprocess_samples(samples):
    processed = []
    for l1_data in samples:
        l1 = (l1_data - np.mean(l1_data)) / (np.std(l1_data) + 1e-8)
        processed.append(l1[:, np.newaxis])
    return torch.tensor(np.array(processed), dtype=torch.float32)

X_train = preprocess_samples(X_train)
X_test = preprocess_samples(X_test)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

hidden_size = 128
model = CNNLSTM(hidden_size=hidden_size, input_channels=1).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def train_model(model, train_loader, test_loader, num_epochs=100):
    best_auc = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(probs)

        acc = accuracy_score(y_true, [1 if p > 0.5 else 0 for p in y_pred])
        auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0

        print(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}, Accuracy: {acc:.4f}, AUC: {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "best_model_l1.pth")

train_model(model, train_loader, test_loader, num_epochs=100)

