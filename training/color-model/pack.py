import csv, numpy as np, sys
from PIL import Image
from multiprocessing import Pool
S = 64
rows = list(csv.reader(open('crops/meta.csv')))
def load(i):
    im = Image.open(f'crops/{rows[i][0]}.jpg').convert('RGB').resize((S, S), Image.BOX)
    return np.asarray(im, dtype=np.uint8)
with Pool(4) as p: arr = p.map(load, range(len(rows)), chunksize=2000)
np.save('img64.npy', np.stack(arr)); print(len(arr))
