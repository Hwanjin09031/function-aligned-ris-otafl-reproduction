"""Controlled conditional-CSI parameter sweeps."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,time,json,hashlib,argparse
import numpy as np
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R))
import core as c
from power_solver import solve
c.solve=solve
NOISE=.01;K=8;L=16;T=20;PASSES=3
METHODS=('function','fairness','rate','power')
def digest(*arrays):
    h=hashlib.sha256()
    for a in arrays:h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()
def data(seed,stream,n,M,N):
    env=c.legacy.create_clustered_environment(np.random.default_rng(seed),K,N,M,4,kappa_G=10.)
    samples=c.cc.generate_clustered_samples(np.random.default_rng(seed+stream*10000),env,n,K,N,M,kappa_hr=10**.3,csi_error_ratio=.05)
    return c.moments(samples)
def utility(g,kind):
    if kind=='fairness':return np.min(np.sum(abs(g)**2,axis=-1),axis=-1)*K*K
    return c.metric(g,NOISE,kind)
def score_matrix(book,hd,C,cv,kind):
    B,K,N=hd.shape;out=np.empty((B,len(book)))
    for start in range(0,len(book),8):
        sub=book[start:start+8];ell=len(sub)
        g=hd[:,None]+np.einsum('bknm,lm->blkn',C,sub)
        if kind=='function':
            cov=np.broadcast_to(cv[:,None],(B,ell,K,N,N)).reshape(-1,K,N,N)
            out[:,start:start+ell]=-solve(g.reshape(-1,K,N),cov,NOISE)[-1].reshape(B,ell)
        else:out[:,start:start+ell]=utility(g.reshape(-1,K,N),kind).reshape(B,ell)
    return out
def greedy(scores,L=16):
    chosen=[];best=np.full(len(scores),-np.inf)
    for _ in range(L):
        gain=np.maximum(best[:,None],scores).mean(0);gain[chosen]=-np.inf
        j=int(gain.argmax());chosen.append(j);best=np.maximum(best,scores[:,j])
    return np.array(chosen)
def make_pool(hd,C,seed,alphabet):
    # Same first-250 training-only construction for every training-set size.
    rng=np.random.default_rng(seed+200);ids=rng.choice(250,64,replace=False)
    B,K,N=hd.shape;M=C.shape[-1];Q=len(alphabet)
    initial=rng.choice(alphabet,(64,M));chunks=[]
    for kind in ('rate','power','fairness'):
        theta=initial.copy()
        for _ in range(PASSES):
            g=c.effective(hd[ids],C[ids],theta)
            for m in range(M):
                without=g-C[ids,:,:,m]*theta[:,m,None,None]
                cand=without[:,None]+C[ids,None,:,:,m]*alphabet[None,:,None,None]
                scores=utility(cand.reshape(-1,K,N),kind).reshape(64,Q)
                ix=scores.argmax(1);theta[:,m]=alphabet[ix];g=cand[np.arange(64),ix]
        chunks.append(theta)
    random=rng.choice(alphabet,(32,M));chunks.append(random)
    pool=np.concatenate(chunks)
    _,ids=np.unique(np.mod(np.rint(np.angle(pool)*Q/(2*np.pi)).astype(int),Q),axis=0,return_index=True)
    return pool[np.sort(ids)],random[:L]
def refine(initial,hd,C,cv,kind,alphabet):
    start=time.perf_counter();B,K,N=hd.shape;L,M=initial.shape;Q=len(alphabet)
    book=initial.copy();scores=score_matrix(book,hd,C,cv,kind);checkpoints=[book.copy()]
    history=[{'pass':0,'mean_best_utility':float(scores.max(1).mean()),'moves':0}]
    cov=np.broadcast_to(cv[:,None],(B,Q,K,N,N)).reshape(-1,K,N,N)
    for sweep in range(PASSES):
        groups=scores.argmax(1);counts=np.bincount(groups,minlength=L)
        g=c.effective(hd,C,book[groups]);moves=0
        for m in (range(M) if sweep%2==0 else range(M-1,-1,-1)):
            current=np.mod(np.rint(np.angle(book[:,m])*Q/(2*np.pi)).astype(int),Q)
            without=g-C[:,:,:,m]*book[groups,m,None,None]
            candidates=without[:,None]+C[:,None,:,:,m]*alphabet[None,:,None,None]
            if kind=='function':state_scores=-solve(candidates.reshape(-1,K,N),cov,NOISE)[-1].reshape(B,Q)
            else:state_scores=utility(candidates.reshape(-1,K,N),kind).reshape(B,Q)
            totals=np.stack([np.bincount(groups,weights=state_scores[:,a],minlength=L) for a in range(Q)],axis=1)
            pick=totals.argmax(1);gain=totals[np.arange(L),pick]-totals[np.arange(L),current]
            keep=(gain<=1e-12*np.maximum(counts,1))|(counts==0);pick[keep]=current[keep]
            assert np.min(totals[np.arange(L),pick]-totals[np.arange(L),current])>=-1e-12
            moves+=int(np.count_nonzero(pick!=current));book[:,m]=alphabet[pick]
            g=candidates[np.arange(B),pick[groups]]
        scores=score_matrix(book,hd,C,cv,kind);value=float(scores.max(1).mean())
        assert value>=history[-1]['mean_best_utility']-1e-11
        history.append({'pass':sweep+1,'mean_best_utility':value,'moves':moves})
        checkpoints.append(book.copy())
    return checkpoints,history,time.perf_counter()-start
