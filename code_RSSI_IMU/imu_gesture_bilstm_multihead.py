import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import random

# ==== fixed seed ====
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ==== 数据增强函数 ====
def apply_augmentation(data):
    data = data.copy()
    # 添加高斯噪声
    noise = np.random.normal(0, 0.01, size=data.shape)
    data += noise

    # 缩放扰动
    scale = np.random.uniform(0.9, 1.1)
    data *= scale

    # 掉轴：以一定概率清除某一轴
    if np.random.rand() < 0.3:  # 30%概率执行
        axis_to_zero = np.random.choice(data.shape[1])  # 0~5轴
        data[:, axis_to_zero] = 0.0

    return data


# ==== Dataset Definition ====
class IMUGestureDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].permute(1, 0), self.y[idx]  # shape: (seq_len, features)

# ==== BiLSTM + Multi-Head Attention Model ====
class BiLSTMWithMultiHeadAttention(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2, num_classes=4, num_heads=4):
        super(BiLSTMWithMultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim*2, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch, seq_len, hidden_dim*2)
        attn_output, _ = self.attn(lstm_out, lstm_out, lstm_out)
        out = attn_output.mean(dim=1)  # average over time steps
        out = self.dropout(out)
        out = self.fc(out)
        return out

# ==== Load and preprocess data ====
df = pd.read_csv("gesture_data.csv", header=None)
X = df.iloc[:, :-1].values.reshape(-1, 6, 20)
y = df.iloc[:, -1].values.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

X_train_aug = np.array([apply_augmentation(x.T).T for x in X_train])
X_train_tensor = torch.tensor(X_train_aug, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = IMUGestureDataset(X_train_tensor, y_train_tensor)
test_dataset = IMUGestureDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ==== Training and Evaluation ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BiLSTMWithMultiHeadAttention().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train_losses = []
test_losses = []

def train_and_evaluate_model():
    for epoch in range(20):
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_test_loss += loss.item()
        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}")

    print("Training complete!")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Testing Loss Curve (BiLSTM + Multi-Head Attention)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def evaluate_classification_metrics(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n Accuracy: {acc * 100:.2f}%\n")

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Purples')
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.show()

    print(" Classification Report (per class):")
    print(classification_report(all_labels, all_preds, digits=4))

# ==== Run ====
train_and_evaluate_model()
evaluate_classification_metrics(model, test_loader, device)

# Save model parameters after training is completed
torch.save(model.state_dict(), "imu_model.pt")