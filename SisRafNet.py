import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.fft import fft

class OFDMSisRafDataset(Dataset):
    def __init__(self, csv_path, n_subcarriers=80):
        self.df = pd.read_csv(csv_path)
        self.n_sub = n_subcarriers

        if 'Unnamed: 0' in self.df.columns:
            self.df = self.df.drop(columns=['Unnamed: 0'])

        self.sample_ids = self.df['sample_id'].unique()

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]

        df_s = self.df[self.df['sample_id'] == sample_id] \
                    .sort_values('time_index')

        tx = df_s['tx_real'].values + 1j * df_s['tx_imag'].values
        rx = df_s['rx_real'].values + 1j * df_s['rx_imag'].values

        # ✅ NumPy FFT (CORRECT)
        X = np.fft.fft(tx, n=self.n_sub)
        Y = np.fft.fft(rx, n=self.n_sub)

        H = Y / (X + 1e-12)

        x = np.stack([Y.real, Y.imag], axis=1)
        y = np.stack([H.real, H.imag], axis=1)

        return torch.tensor(x, dtype=torch.float32), \
               torch.tensor(y, dtype=torch.float32)

class SisRafNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.proj = nn.Linear(2 * hidden_dim, hidden_dim)

        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        # x: (B, 80, 2)
        x = self.embedding(x)

        out, _ = self.gru(x)  # (B, 80, 2D)
        out = self.proj(out)  # (B, 80, D)

        h_hat = self.head(out)  # (B, 80, 2)
        return h_hat

def train_sisrafnet():
    csv_path = r"C:\Users\Asus\AY2025-26_FYP\FYP25-26\ofdm_signal_data\ofdm_signals_1000samples_20251210_214733.csv"

    dataset = OFDMSisRafDataset(csv_path)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SisRafNet(
        input_dim=2,
        hidden_dim=64,
        num_layers=2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    epochs = 40

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            y_hat = model(x)

            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}]  Loss: {total_loss / len(train_loader):.6f}")

    print("✅ Training finished")
    return model

if __name__ == "__main__":
    model = train_sisrafnet()
