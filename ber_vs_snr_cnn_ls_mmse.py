from xml.parsers.expat import model
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ======================================================
# 1. PATHS & SYSTEM PARAMETERS
# ======================================================
DATA_PATH = r"C:\Users\Asus\AY2025-26_FYP\FYP25-26\ofdm_signal_data\ofdm_signals_1000samples_20251215_210922.csv"
MODEL_PATH =r"C:\Users\Asus\AY2025-26_FYP\FYP25-26\saved_models\cnn_channel_estimator_20251211_222519.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_FFT = 64
pilot_idx = np.array([0, 8, 16, 24, 32, 40, 48, 56, 63])
all_idx = np.arange(N_FFT)
data_idx = np.setdiff1d(all_idx, pilot_idx)
CP_LEN = 16

SNR_dB_RANGE = np.arange(0, 31, 5)   # 0,5,10,...,30 dB

# ======================================================
# 2. 16-QAM MAPPING (USER-PROVIDED)
# ======================================================
mapping_table = {
    (0,0,0,0): -3-3j, (0,0,0,1): -3-1j, (0,0,1,0): -3+3j, (0,0,1,1): -3+1j,
    (0,1,0,0): -1-3j, (0,1,0,1): -1-1j, (0,1,1,0): -1+3j, (0,1,1,1): -1+1j,
    (1,0,0,0):  3-3j, (1,0,0,1):  3-1j, (1,0,1,0):  3+3j, (1,0,1,1):  3+1j,
    (1,1,0,0):  1-3j, (1,1,0,1):  1-1j, (1,1,1,0):  1+3j, (1,1,1,1):  1+1j
}

demap_table = {v: k for k, v in mapping_table.items()}
const_points = np.array(list(demap_table.keys()))
bit_labels = list(demap_table.values())

# ======================================================
# 3. 16-QAM DEMAPPER
# ======================================================
def qam16_demap(symbols):
    bits = []
    for s in symbols:
        idx = np.argmin(np.abs(s - const_points))
        bits.extend(bit_labels[idx])
    return np.array(bits, dtype=int)

# ======================================================
# 4. CHANNEL ESTIMATORS
# ======================================================
def ls_estimator_pilot(rx_freq, tx_freq, pilot_idx):
    return rx_freq[pilot_idx] / tx_freq[pilot_idx]

def interpolate_channel(H_pilot, pilot_idx, N_FFT):
    all_idx = np.arange(N_FFT)
    H_real = np.interp(all_idx, pilot_idx, H_pilot.real)
    H_imag = np.interp(all_idx, pilot_idx, H_pilot.imag)
    return H_real + 1j * H_imag

def mmse_estimator_pilot(rx_freq, tx_freq, pilot_idx, noise_var):
    H_ls_pilot = rx_freq[pilot_idx] / tx_freq[pilot_idx]
    snr_pilot = np.abs(tx_freq[pilot_idx])**2 / noise_var
    H_mmse_pilot = H_ls_pilot * (snr_pilot / (snr_pilot + 1))
    return H_mmse_pilot


"""
state_dict = torch.load(MODEL_PATH, map_location="cpu")
for k in state_dict.keys():
print(k)
"""

# ======================================================
# 5. CNN CHANNEL ESTIMATOR
# ======================================================
class CNN_ChannelEstimator(nn.Module):
    def __init__(self, input_dim=2, hidden_dims=[128, 256, 512, 512, 256, 128, 64],
                 output_dim=2, dropout_rate=0.2):
        super().__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.input_bn = nn.BatchNorm1d(hidden_dims[0])

        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.residual_layers = nn.ModuleList()

        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dims[i+1]))

            if hidden_dims[i] == hidden_dims[i+1]:
                self.residual_layers.append(None)
            else:
                self.residual_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))

        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.elu = nn.ELU()

        self.dropout = nn.Dropout(dropout_rate)
        self.dropout_heavy = nn.Dropout(dropout_rate * 1.5)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.input_bn(x)
        x = self.elu(x)
        x = self.dropout(x)

        for i, (layer, bn, residual) in enumerate(
            zip(self.hidden_layers, self.batch_norms, self.residual_layers)
        ):
            identity = x
            x = layer(x)
            x = bn(x)

            if residual is None:
                x = x + identity
            else:
                x = x + residual(identity)

            if i < len(self.hidden_layers) // 2:
                x = self.elu(x)
                x = self.dropout(x)
            else:
                x = self.leaky_relu(x)
                x = self.dropout_heavy(x)

        return self.output_layer(x)
    
def load_cnn():
    model = CNN_ChannelEstimator().to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def dnn_channel_estimator(model, rx_freq):
    """
    rx_freq: numpy array of complex RX symbols, shape (N,)
    returns: complex channel estimate, shape (N,)
    """
    x = np.stack([rx_freq.real, rx_freq.imag], axis=1)  # (N, 2)
    x = torch.tensor(x, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        h_hat = model(x).cpu().numpy()  # (N, 2)

    return h_hat[:, 0] + 1j * h_hat[:, 1]

# ======================================================
# 6. BER FUNCTION
# ======================================================
def ber(tx_bits, rx_bits):
    return np.mean(tx_bits != rx_bits)

# ======================================================
# 7. MAIN BER vs SNR EVALUATION
# ======================================================
def main():
    df = pd.read_csv(DATA_PATH)
    model = load_cnn()

    ber_ls_all = []
    ber_mmse_all = []
    ber_cnn_all = []

    print("🚀 Running BER vs SNR evaluation...")

    for snr_db in SNR_dB_RANGE:
        print(f"🔹 SNR = {snr_db} dB")

        ber_ls, ber_mmse, ber_cnn = [], [], []

        for sid in df.sample_id.unique():
            sample = df[df.sample_id == sid]

            tx_time = sample.tx_real.values + 1j * sample.tx_imag.values
            rx_time_clean = sample.rx_real.values + 1j * sample.rx_imag.values

            # AWGN
            signal_power = np.mean(np.abs(rx_time_clean)**2)
            noise_var = signal_power * 10**(-snr_db / 10)
            noise = np.sqrt(noise_var / 2) * (
                np.random.randn(len(rx_time_clean)) +
                1j * np.random.randn(len(rx_time_clean))
            )
            rx_time = rx_time_clean + noise

            # FFT
            tx_freq = np.fft.fft(tx_time[CP_LEN:CP_LEN + N_FFT])
            rx_freq = np.fft.fft(rx_time[CP_LEN:CP_LEN + N_FFT])

            tx_bits_data = qam16_demap(tx_freq[data_idx])

            # ============================
            # LS (pilot-based)
            # ============================
            H_ls_pilot = ls_estimator_pilot(rx_freq, tx_freq, pilot_idx)
            H_ls = interpolate_channel(H_ls_pilot, pilot_idx, N_FFT)
            eq_ls = rx_freq / H_ls

            rx_bits_ls = qam16_demap(eq_ls[data_idx])
            ber_ls.append(ber(tx_bits_data, rx_bits_ls))


            # ============================
            # MMSE (pilot-based)
            # ============================
            H_mmse_pilot = mmse_estimator_pilot(
                rx_freq, tx_freq, pilot_idx, noise_var
            )
            H_mmse = interpolate_channel(H_mmse_pilot, pilot_idx, N_FFT)
            eq_mmse = rx_freq / H_mmse

            rx_bits_mmse = qam16_demap(eq_mmse[data_idx])
            ber_mmse.append(ber(tx_bits_data, rx_bits_mmse))


            # ============================
            # DNN (pilot-based)
            # ============================
            H_dnn_pilot = dnn_channel_estimator(model, rx_freq[pilot_idx])
            H_dnn = interpolate_channel(H_dnn_pilot, pilot_idx, N_FFT)
            eq_cnn = rx_freq / H_dnn

            rx_bits_cnn = qam16_demap(eq_cnn[data_idx])
            ber_cnn.append(ber(tx_bits_data, rx_bits_cnn))


        ber_ls_all.append(np.mean(ber_ls))
        ber_mmse_all.append(np.mean(ber_mmse))
        ber_cnn_all.append(np.mean(ber_cnn))


    # ==================================================
    # 8. PLOT RESULTS
    # ==================================================
    plt.figure(figsize=(8,6))
    plt.semilogy(SNR_dB_RANGE, ber_ls_all, marker='o', label='LS')
    #plt.semilogy(SNR_dB_RANGE, ber_mmse_all, marker='s', label='MMSE')
    plt.semilogy(SNR_dB_RANGE, ber_cnn_all, marker='^', label='DNN')

    print("LS BER:", ber_ls_all)
    print("MMSE BER:", ber_mmse_all)
    print("DNN BER:", ber_cnn_all)

    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("BER vs SNR Comparison (16-QAM OFDM)")
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
