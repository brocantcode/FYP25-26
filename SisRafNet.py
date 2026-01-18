import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

__all__ = ["OFDMSisRafDataset", "SisRafNet", "train_sisrafnet"]

class OFDMSisRafDataset(Dataset):
    """OFDM frequency-domain dataset for SisRafNet training.

    Loads TX/RX complex time-domain signals from CSV, performs an `n_subcarriers`-point
    FFT (without CP removal), and returns normalized channel features: `H_gt` and
    LS channel estimate `H_ls` as four real-valued channels `[Re(H_gt), Im(H_gt), Re(H_ls), Im(H_ls)]`,
    and the residual channel target `(H_gt - H_ls)` as two channels. This ensures
    inputs and targets share the same scale.

    Note: This dataset shape (`n_subcarriers=80`) matches previously saved models.
    """

    def __init__(self, csv_path: str, n_subcarriers: int = 64, perfect_ls: bool = False) -> None:
        self.df = pd.read_csv(csv_path)
        self.n_sub = n_subcarriers
        self.perfect_ls = perfect_ls

        if 'Unnamed: 0' in self.df.columns:
            self.df = self.df.drop(columns=['Unnamed: 0'])

        self.sample_ids = self.df['sample_id'].unique()

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_id = self.sample_ids[idx]

        df_s = self.df[self.df['sample_id'] == sample_id].sort_values('time_index')

        tx = df_s['tx_real'].values + 1j * df_s['tx_imag'].values
        rx = df_s['rx_real'].values + 1j * df_s['rx_imag'].values

        # Frequency-domain (NumPy FFT)
        X = np.fft.fft(tx, n=self.n_sub)
        Y = np.fft.fft(rx, n=self.n_sub)

        # Ground-truth channel
        H_gt = Y / (X + 1e-12)

        # LS channel estimate using the SAME pilot setup as evaluation (K=64)
        pilot_value = 3 + 3j
        if self.n_sub == 64:
            pilot_carriers = np.array([0, 8, 16, 24, 32, 40, 48, 56, 63], dtype=int)
        else:
            # Fallback to evenly-spaced 9 pilots when K != 64
            pilot_carriers = np.linspace(0, self.n_sub - 1, num=9, dtype=int)
        H_ls = ls_channel_estimation(Y[np.newaxis, :], pilot_carriers, pilot_value, self.n_sub)

        # Perfect LS experiment: force H_ls = H_gt and residual target = 0
        if self.perfect_ls:
            H_ls = H_gt

        # Residual learning target: H_gt - H_ls
        # Network input uses channel-scale features: (Re(H_gt), Im(H_gt), Re(H_ls), Im(H_ls))
        x = np.stack([np.real(H_gt), np.imag(H_gt), np.real(H_ls), np.imag(H_ls)], axis=1)  # (n_sub, 4)
        if self.perfect_ls:
            y = np.zeros((self.n_sub, 2), dtype=np.float32)
        else:
            y = np.stack([np.real(H_gt - H_ls), np.imag(H_gt - H_ls)], axis=1)  # (n_sub, 2)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def ls_channel_estimation(Y: np.ndarray, pilot_carriers: np.ndarray, pilot_value: complex, K: int) -> np.ndarray:
    """Pilot-only LS with linear interpolation across subcarriers.

    Args:
        Y: shape (N_sym, K)
        pilot_carriers: indices of pilot tones
        pilot_value: known pilot value (complex)
        K: number of subcarriers
    Returns:
        H_est averaged over symbols, shape (K,)
    """

    Y = np.asarray(Y)
    assert Y.ndim == 2, f"Y must be 2-D, got {Y.shape}"

    N_sym = Y.shape[0]
    H_est = np.zeros((N_sym, K), dtype=np.complex64)

    for n in range(N_sym):
        Y_n = Y[n]

        H_pilots = Y_n[pilot_carriers] / pilot_value

        H_real = np.interp(
            np.arange(K),
            pilot_carriers,
            H_pilots.real
        )
        H_imag = np.interp(
            np.arange(K),
            pilot_carriers,
            H_pilots.imag
        )

        H_est[n] = H_real + 1j * H_imag

    # Average over OFDM symbols
    return np.mean(H_est, axis=0)

class SisRafNet(nn.Module):
    """Bidirectional GRU-based channel estimator operating along frequency.

    Architecture is kept identical to the saved model layout:
    - Embedding: Linear -> LayerNorm -> ReLU producing `hidden_dim`
    - GRU: 2-layer bidirectional over the frequency axis
    - Projection: 2*hidden_dim -> hidden_dim
    - Head: hidden_dim -> 2 (Re, Im)
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, num_layers: int = 2) -> None:
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        out, _ = self.gru(x)
        out = self.proj(out)
        return self.head(out)

def nmse_loss(y_hat: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalized MSE over batch.

    Computes NMSE per sample as ||y_hat - y||^2 / (||y||^2 + eps) and returns the batch mean.

    Expected shapes: (B, K, 2) where the last dim is [Re, Im].
    """
    err = y_hat - y
    num = torch.sum(err * err, dim=(1, 2))
    den = torch.sum(y * y, dim=(1, 2)) + eps
    nmse = num / den
    return torch.mean(nmse)

def channel_nmse_loss(delta_H_pred: torch.Tensor, H_ls: torch.Tensor, H_gt: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Channel-aware NMSE: compares reconstructed channel H_hat to H_gt.

    Inputs are real-imag two-channel tensors of shape (B, K, 2).
    H_hat = H_ls + delta_H_pred
    Loss = mean_b( ||H_hat - H_gt||^2 / (||H_gt||^2 + eps) )
    """
    H_hat = H_ls + delta_H_pred
    err = H_hat - H_gt
    num = torch.sum(err * err, dim=(1, 2))
    den = torch.sum(H_gt * H_gt, dim=(1, 2)) + eps
    return torch.mean(num / den)

def train_sisrafnet() -> SisRafNet:
    """Train SisRafNet on the OFDM dataset with simple early stopping.

    Paths are set to match the current workspace. Training configuration is kept
    identical to preserve compatibility with previously saved weights.
    """

    csv_path = r"C:\Users\Asus\AY2025-26_FYP\FYP25-26\ofdm_signal_data\ofdm_signals_1000samples_20251210_214733.csv"

    # Model save path
    save_dir = r"C:\Users\Asus\AY2025-26_FYP\FYP25-26\saved_models"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "SisRafNet_freq_domain.pth")

    dataset = OFDMSisRafDataset(csv_path, n_subcarriers=64)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SisRafNet(input_dim=4, hidden_dim=64, num_layers=2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Residual NMSE for delta_H; channel NMSE combines H_ls + delta_H vs H_gt

    # Allow overriding epochs via CLI arg or env var for quick runs
    if len(sys.argv) > 1:
        try:
            epochs = int(sys.argv[1])
        except Exception:
            epochs = int(os.environ.get("EPOCHS", "100"))
    else:
        epochs = int(os.environ.get("EPOCHS", "100"))
    # Fallback comment: default max epochs
    # epochs = 100
    patience = 10              # early stop patience
    best_loss = float("inf")
    patience_counter = 0

    print("Training SisRafNet with early stopping...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            y_hat = model(x)

            # Two-stage loss
            delta_H_pred = y_hat            # (B, K, 2)
            delta_H_gt = y                  # (B, K, 2)
            H_gt = x[..., 0:2]              # (B, K, 2)
            H_ls = x[..., 2:4]              # (B, K, 2)

            loss_channel = channel_nmse_loss(delta_H_pred, H_ls, H_gt)
            loss_resid = nmse_loss(delta_H_pred, delta_H_gt)
            loss = loss_channel + 0.1 * loss_resid
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Early stopping tracking
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"Epoch {epoch+1}: new best loss = {best_loss:.6f} (model saved)")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}: no improvement ({patience_counter}/{patience})")

        print(f"Epoch [{epoch+1}/{epochs}]  NMSE: {avg_loss:.6f}")

        # (Sanity prints removed)

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Training finished.")
    print(f"Best model saved at: {model_path}")

    return model

if __name__ == "__main__":
    train_sisrafnet()
