import scipy.io as sio

for f in ["97.mat", "105.mat"]:
    data = sio.loadmat(f)
    keys = [k for k in data if not k.startswith("__")]
    print(f, "->", keys)
    for k in keys:
        print("  ", k, data[k].shape if hasattr(data[k], "shape") else data[k])