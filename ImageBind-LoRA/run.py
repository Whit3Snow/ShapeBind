import numpy as np
from tqdm import tqdm

a = np.load('../../volume/embeddings/embeddings-edit.npz')
b = np.load('../../volume/embeddings/embeddings-edit-test.npz')

e = {}
for file in tqdm(a.files):
    e[file] = a[file]
for file in tqdm(b.files):
    e[file] = b[file]

np.savez('../../volume/embeddings/embeddings-full.npz', **e)
