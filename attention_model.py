import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class AttentionAggregator(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=1)
        )
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        attn = self.attention(x)    
        x = x * attn                 
        x = torch.relu(self.fc1(x))  
        x = self.norm1(x)
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))  
        x = self.norm2(x)
        x = self.dropout(x)
        out = self.out(x)            
        return out, attn

class AttentionTrainer:
    def __init__(self, sent_cols, lr=0.001, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.scaler = StandardScaler()
        self.sent_cols = sent_cols
        self.model = AttentionAggregator(input_dim=len(sent_cols))
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.schedular = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10)

        self.losses = []
        self.accuracies = []

    def prepare_data(self, df, target_col):
        features = self.sent_cols 

        df['Target_smooth'] = df[target_col].rolling(3).mean().fillna(0)

        X = df[features].fillna(0).values
        y = df['Target_smooth'].apply(lambda x: 1 if x > 0 else 0).values

        X = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

        input_dim = X_train.shape[1]
        self.model = AttentionAggregator(input_dim=input_dim, hidden_dim=128, dropout=0.1)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10)

        self.features = features

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs, _ = self.model(self.X_train)
            loss = self.criterion(outputs, self.y_train)
            loss.backward()
            self.optimizer.step()

            self.losses.append(loss.item())


            with torch.no_grad():
                preds_binary = (torch.sigmoid(outputs) >= 0.5).float()
                acc = (preds_binary == self.y_train).float().mean().item()
                self.accuracies.append(acc)

            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}]  Loss: {loss.item():.4f}")

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            preds, attn = self.model(self.X_test)
            probs = torch.sigmoid(preds)
            preds_binary = (probs >= 0.5).float()
            accuracy = (preds_binary == self.y_test).float().mean().item()
            print(f"Validation Accuracy: {accuracy:.4f}")
        return probs.numpy(), attn.numpy()

    def add_attention_scores(self, df):
        X = df[self.sent_cols].fillna(0).values
        X = self.scaler.transform(X)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            scores, attn = self.model(X_tensor)
            df['attention_score'] = torch.sigmoid(scores).numpy()
        return df, attn
    
    def save_model(self, path="attention_model_weights.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"Model weights saved at: {path}")

    def save_attention_csv(self, df, path="df_with_attention_scores.csv"):
        df.to_csv(path, index=False)
        print(f"CSV saved at: {path}")

    def plot_metrics(self):
        plt.figure(figsize=(12,5))

        plt.subplot(1,2,1)
        plt.plot(range(1, len(self.losses)+1), self.losses, label="Training Loss", color='blue')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(range(1, len(self.accuracies)+1), self.accuracies, label="Training Accuracy", color='green')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy Curve")
        plt.legend()

        plt.tight_layout()
        plt.show()
