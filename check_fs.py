import scipy.io as sio
import numpy as np

healthy = sio.loadmat("97.mat")["X097_DE_time"].flatten()
rpm = float(sio.loadmat("97.mat")["X097RPM"].squeeze())
print(f"RPM stored in file: {rpm}  ->  shaft freq = {rpm/60:.2f} Hz")

FS = 12000
n = len(healthy)
spectrum = np.abs(np.fft.rfft(healthy * np.hanning(n))) / n
freqs = np.fft.rfftfreq(n, d=1/FS)

def peak_near(target, half_width=1.0):
    mask = (freqs > target - half_width) & (freqs < target + half_width)
    i = np.argmax(spectrum[mask])
    return freqs[mask][i], spectrum[mask][i]

f30, m30 = peak_near(rpm/60)        # where shaft freq sits if 12 kHz is right
f75, m75 = peak_near(rpm/60/4)      # where it would sit if file is really 48 kHz

print(f"Peak near {rpm/60:.2f} Hz (12 kHz hypothesis): {f30:.2f} Hz, magnitude {m30:.6f}")
print(f"Peak near {rpm/60/4:.2f} Hz (48 kHz hypothesis): {f75:.2f} Hz, magnitude {m75:.6f}")
print(f"Ratio (12kHz peak / 48kHz peak): {m30/m75:.1f}x")