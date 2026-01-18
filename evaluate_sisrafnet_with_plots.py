import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import sys

# Ensure we import the exact trained architecture to avoid GRU mismatches
sys.path.append(os.path.dirname(__file__))
from SisRafNet import SisRafNet

# 16-QAM mapping (must match generator)
QAM16_MAPPING = {
    (0,0,0,0): -3-3j, (0,0,0,1): -3-1j, (0,0,1,0): -3+3j, (0,0,1,1): -3+1j,
    (0,1,0,0): -1-3j, (0,1,0,1): -1-1j, (0,1,1,0): -1+3j, (0,1,1,1): -1+1j,
    (1,0,0,0):  3-3j, (1,0,0,1):  3-1j, (1,0,1,0):  3+3j, (1,0,1,1):  3+1j,
    (1,1,0,0):  1-3j, (1,1,0,1):  1-1j, (1,1,1,0):  1+3j, (1,1,1,1):  1+1j
}
DEMAP_TABLE = {v: k for k, v in QAM16_MAPPING.items()}

def qam16_demap_bits(symbols: np.ndarray) -> np.ndarray:
    """Demap complex 16-QAM symbols to bit array using nearest constellation points."""
    const = np.array(list(DEMAP_TABLE.keys()))
    # Distance matrix (N x 16)
    dists = np.abs(symbols.reshape(-1, 1) - const.reshape(1, -1))
    idx = dists.argmin(axis=1)
    hard = const[idx]
    bits = np.array([DEMAP_TABLE[s] for s in hard], dtype=np.int32)
    return bits.reshape(-1)

class OFDMEvalDataset(Dataset):
    """Loads TX/RX from CSV, removes CP, returns frequency-domain Y and ground-truth H."""

    def __init__(self, csv_path: str, K: int = 64, CP: int = 16):
        self.df = pd.read_csv(csv_path)
        self.K = K
        self.CP = CP

        if 'Unnamed: 0' in self.df.columns:
            self.df = self.df.drop(columns=['Unnamed: 0'])

        self.sample_ids = self.df['sample_id'].unique()

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx: int):
        sid = self.sample_ids[idx]
        df_s = self.df[self.df['sample_id'] == sid].sort_values('time_index')

        tx = df_s['tx_real'].values + 1j * df_s['tx_imag'].values
        rx = df_s['rx_real'].values + 1j * df_s['rx_imag'].values

        # Remove CP and compute K-point FFT matching the OFDM system
        tx_no_cp = tx[self.CP:self.CP + self.K]
        rx_no_cp = rx[self.CP:self.CP + self.K]

        X = np.fft.fft(tx_no_cp, n=self.K)
        Y = np.fft.fft(rx_no_cp, n=self.K)

        H_gt = Y / (X + 1e-12)
        H_ls = H_gt.copy()  # kept for interface compatibility (unused downstream)

        x = np.stack([Y.real, Y.imag], axis=1)
        H_gt = np.stack([H_gt.real, H_gt.imag], axis=1)
        H_ls = np.stack([H_ls.real, H_ls.imag], axis=1)

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(H_gt, dtype=torch.float32),
            torch.tensor(H_ls, dtype=torch.float32)
        )

## Note: The SisRafNet architecture is imported from SisRafNet.py above.
    
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

## Removed legacy MMSE and unused helpers for clarity.

def evaluate_and_plot():
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    # ===============================
    # Paths
    # ===============================
    csv_path = r"C:\Users\Asus\AY2025-26_FYP\FYP25-26\ofdm_signal_data\ofdm_signals_1000samples_20251215_210922.csv"
    model_path = r"C:\Users\Asus\AY2025-26_FYP\FYP25-26\saved_models\SisRafNet_freq_domain.pth"

    # ===============================
    # OFDM Parameters
    # ===============================
    # Fixed OFDM parameters from notebook
    K = 64
    CP = 16
    pilot_carriers = np.array([0, 8, 16, 24, 32, 40, 48, 56, 63], dtype=int)
    pilot_value = 3 + 3j
    data_carriers = np.setdiff1d(np.arange(K), pilot_carriers)
    eps = 1e-12

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===============================
    # Dataset & Model
    # ===============================
    dataset = OFDMEvalDataset(csv_path, K=K, CP=CP)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = SisRafNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    mse_ls_list = []
    mse_sisraf_list = []
    nmse_ls_list = []
    nmse_sisraf_list = []
    ber_ls_list = []
    ber_sisraf_list = []

    # ===============================
    # Evaluation Loop
    # ===============================
    with torch.no_grad():
        for x_freq, H_gt, _ in loader:
            # x_freq: (1, K, 2)
            x_freq_np = x_freq.squeeze(0).numpy()
            H_gt_np = H_gt.squeeze(0).numpy()  # (K, 2)
            # K fixed by dataset
            assert x_freq_np.shape[0] == K, f"Expected K={K}, got {x_freq_np.shape[0]}"

            # Recover Y and H_gt (complex) from tensors
            Y = x_freq_np[:, 0] + 1j * x_freq_np[:, 1]  # (K,)
            H_gt_c = H_gt_np[:, 0] + 1j * H_gt_np[:, 1]

            # Reconstruct transmitted frequency-domain symbols X = Y / H_gt
            X_hat = Y / (H_gt_c + eps)

            # Compute ground-truth tx_bits via demapping of X on data carriers
            tx_syms = X_hat[data_carriers]
            tx_bits = qam16_demap_bits(tx_syms)

            # -------- Pilot-based LS only --------
            H_ls_c = ls_channel_estimation(Y[np.newaxis, :], pilot_carriers, pilot_value, K)

            # -------- SisRafNet (channel-scale input) --------
            # Match training: use [Re(H_gt), Im(H_gt), Re(H_ls), Im(H_ls)]
            x_input = np.concatenate([
                H_gt_np,                         # (K, 2)
                np.real(H_ls_c)[:, None],        # (K, 1)
                np.imag(H_ls_c)[:, None]         # (K, 1)
            ], axis=1)

            # SisRafNet predicts residual
            delta_H = model(torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).to(device)).cpu().numpy().squeeze(0)
            delta_H_c = delta_H[:, 0] + 1j * delta_H[:, 1]
            # Final channel estimate
            H_sisraf_c = H_ls_c + delta_H_c


            # -------- MSE vs Ground Truth --------
            mse_ls_list.append(np.mean(np.abs(H_ls_c - H_gt_c) ** 2))
            mse_sisraf_list.append(np.mean(np.abs(H_sisraf_c - H_gt_c)**2))

            # -------- NMSE vs Ground Truth --------
            nmse_ls_list.append(
                np.sum(np.abs(H_ls_c - H_gt_c) ** 2) / np.sum(np.abs(H_gt_c) ** 2)
            )
            nmse_sisraf_list.append(
                np.sum(np.abs(H_sisraf_c - H_gt_c) ** 2) / np.sum(np.abs(H_gt_c) ** 2)
            )


            # -------- BER (demap equalized data carriers) --------
            Y_eq_ls = Y / (H_ls_c + eps)
            Y_eq_sisraf = Y / (H_sisraf_c + eps)

            rx_bits_ls = qam16_demap_bits(Y_eq_ls[data_carriers])
            rx_bits_sisraf = qam16_demap_bits(Y_eq_sisraf[data_carriers])

            # Truncate to min length (safety)
            L = min(len(tx_bits), len(rx_bits_ls), len(rx_bits_sisraf))
            ber_ls_list.append(np.mean(tx_bits[:L] != rx_bits_ls[:L]))
            ber_sisraf_list.append(np.mean(tx_bits[:L] != rx_bits_sisraf[:L]))

    # ===============================
    # Results
    # ===============================
    print("\nEvaluation Results")
    print("----------------------------------")
    print(f"LS MSE        : {np.mean(mse_ls_list):.6e}")
    print(f"SisRafNet MSE : {np.mean(mse_sisraf_list):.6e}")
    print(f"LS NMSE       : {np.mean(nmse_ls_list):.6e}")
    print(f"SisRafNet NMSE: {np.mean(nmse_sisraf_list):.6e}")
    print(f"LS   BER      : {np.mean(ber_ls_list):.6e}")
    print(f"SisRafNet BER : {np.mean(ber_sisraf_list):.6e}")
    print(f"MSE Improvement: {(1 - np.mean(mse_sisraf_list)/np.mean(mse_ls_list))*100:.2f}%")

    # ===============================
    # BER Plot
    # ===============================
    plt.figure(figsize=(8, 5))
    plt.plot(mse_ls_list, label="LS MSE vs GT")
    plt.plot(mse_sisraf_list, label="SisRafNet MSE vs GT")
    plt.xlabel("Sample Index")
    plt.ylabel("MSE")
    plt.title("MSE Comparison vs Ground Truth")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(ber_ls_list, label="LS BER")
    plt.plot(ber_sisraf_list, label="SisRafNet BER")
    plt.xlabel("Sample Index")
    plt.ylabel("BER")
    plt.title("BER Comparison (16-QAM)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_and_plot()
