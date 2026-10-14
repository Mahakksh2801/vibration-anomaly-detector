import scipy.io as sio
import matplotlib.pyplot as plt

healthy = sio.loadmat("97.mat")["X097_DE_time"].flatten()
faulty  = sio.loadmat("105.mat")["X105_DE_time"].flatten()

N = 4000  # number of samples to display

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharey=True)

axes[0].plot(healthy[:N], linewidth=0.7)
axes[0].set_title("Healthy bearing — 97.mat (drive-end accelerometer)")
axes[0].set_ylabel("Amplitude")

axes[1].plot(faulty[:N], linewidth=0.7, color="tab:red")
axes[1].set_title("Inner race fault 0.007\" — 105.mat (drive-end accelerometer)")
axes[1].set_ylabel("Amplitude")
axes[1].set_xlabel("Sample index")

plt.tight_layout()
plt.savefig("time_domain_comparison.png", dpi=150)
plt.show()