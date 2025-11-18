import os, shutil, numpy as np
base='data/poetry_whitman'
train = np.memmap(f'{base}/train.bin', dtype=np.uint16, mode='r')
sizes = [50_000, 100_000, 300_000, min(1_000_000, len(train))]
for N in sizes:
    out=f'{base}_{N}'; os.makedirs(out, exist_ok=True)
    shutil.copy(f'{base}/val.bin',  f'{out}/val.bin')
    shutil.copy(f'{base}/meta.pkl', f'{out}/meta.pkl')
    np.array(train[:N], dtype=np.uint16).tofile(f'{out}/train.bin')
    print("wrote", out)