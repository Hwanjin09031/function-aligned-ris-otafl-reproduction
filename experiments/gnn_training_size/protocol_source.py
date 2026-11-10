import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,time,json,hashlib
import numpy as np
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'source'))
import training as tr
from power_solver import solve
c=tr.c;assert c.solve is solve
KINDS=('function','fairness','rate','power')
def digest(*arrays):
 h=hashlib.sha256()
 for a in arrays:h.update(np.ascontiguousarray(a).tobytes())
 return h.hexdigest()
def samples(seed,n=1000,rho=.05,stream=1,K=8,N=4):
 env=c.legacy.create_clustered_environment(np.random.default_rng(seed),K,N,64,4,kappa_G=10.)
 return c.cc.generate_clustered_samples(np.random.default_rng(seed+10000*stream),env,n,K,N,64,kappa_hr=10**.3,csi_error_ratio=rho)
def dump(p,x):p.write_text(json.dumps(x,indent=2))
