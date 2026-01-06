import numpy as np
import torch 
import pickle

dic = None
with open("/home/ravindu/fyp/neural_timeseries_diffusion/outputs/2026-01-05/22-35-17/samples.pkl", "rb") as f:
    dic = pickle.load(f)

sig = dic["samples"][0]
cond = dic["cond"][0]

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"

# convert to numpy
if isinstance(sig, torch.Tensor):
    arr = sig.detach().cpu().numpy()
else:
    arr = np.array(sig)

# normalize shape to (channels, samples)
if arr.ndim == 1 and arr.size == 12 * 800:
    arr = arr.reshape(12, 800)
elif arr.ndim == 2 and arr.shape == (800, 12):
    arr = arr.T
elif arr.ndim == 2 and arr.shape[0] != 12 and arr.shape[1] == 12:
    arr = arr.T

n_ch, n_samp = arr.shape

fig, axs = plt.subplots(n_ch, 1, figsize=(12, 2 * n_ch), sharex=True)
for i in range(n_ch):
    axs[i].plot(np.arange(n_samp), arr[i])
    axs[i].set_ylabel(f"Ch {i+1}", rotation=0, labelpad=20)
axs[-1].set_xlabel("Sample")
plt.title(str(cond))
plt.tight_layout()
plt.savefig("test_EEG.png")

