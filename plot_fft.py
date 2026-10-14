import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

healthy = sio.loadmat("97.mat")["X097_DE_time"].flatten()
faulty  = sio.loadmat("105.mat")["X105_DE_time"].flatten()

FS = 12000  # assumed sample rate in Hz — we verify this below

def compute_fft(signal, fs):
    n = len(signal)
    window = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(signal * window)) / n
    freqs = np.fft.rfftfreq(n, d=1/fs)
    return freqs, spectrum

f_h, s_h = compute_fft(healthy, FS)
f_f, s_f = compute_fft(faulty, FS)

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=True)

axes[0].plot(f_h, s_h, linewidth=0.7)
axes[0].set_title("Healthy — FFT (assuming 12 kHz)")
axes[0].set_ylabel("Magnitude")

axes[1].plot(f_f, s_f, linewidth=0.7, color="tab:red")
axes[1].set_title("Inner race fault — FFT (assuming 12 kHz)")
axes[1].set_ylabel("Magnitude")
axes[1].set_xlabel("Frequency (Hz)")

axes[1].set_xlim(0, 1000)  # zoom into the region that matters

plt.tight_layout()
plt.savefig("fft_comparison.png", dpi=150)
plt.show()

# Sample-rate sanity check: shaft speed is ~1797 rpm = ~29.95 Hz.
# Report the strongest low-frequency peak in the healthy spectrum.
mask = (f_h > 5) & (f_h < 100)
peak_freq = f_h[mask][np.argmax(s_h[mask])]
print(f"Strongest healthy peak between 5-100 Hz: {peak_freq:.2f} Hz")
print("Shaft frequency should be ~29.95 Hz. If the peak is ~7.5 Hz, the file is actually 48 kHz.")