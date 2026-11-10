from pathlib import Path
import sys, os, hashlib, csv
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
os.environ['RIS_MNIST_RAW']=str(ROOT/'data/MNIST/raw')
import core as c
import fl_core as base
from power_solver import solve
c.solve=solve
D=38282; H=256; NOISE=.01
METHODS=['function','fairness','rate','power','ao_2bit']
def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def save_csv(path,rows):
    keys=sorted(set().union(*(r.keys() for r in rows)))
    temp=Path(str(path)+'.tmp')
    with temp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    temp.replace(path)
def layout(tau,p):
    capacity=tau-p-H
    if capacity<=0:raise ValueError('No payload capacity')
    slices=[(a,min(a+capacity,D)) for a in range(0,D,capacity)]
    return slices,D+len(slices)*(p+H)
def configure():
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.mkldnn.deterministic=True
    np.seterr(over='raise',invalid='raise')
def load_books(seed):
    books=dict(np.load(ROOT/'books'/str(seed)/'books.npz'))
    books['union']=np.concatenate([books[k] for k in METHODS[:4]])
    return books
