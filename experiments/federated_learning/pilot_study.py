"""Empirical pilot/LS companion. NOT an exact Bayesian conditional-CSI model.

UE pilots are orthogonal over K=8 uses per RIS state. Despread observations
have complex-noise variance noise/(K*pilot_power*repetitions). All decisions
read observed LS channels only; original conditional means are not available.
"""
from common import *
from scipy.linalg import hadamard
import json,time
K=8;N=4;M=64;P=65
def cn(rng,shape,q):return np.sqrt(q/2)*(rng.normal(size=shape)+1j*rng.normal(size=shape))
def factors(theta,W):
    t=np.concatenate([np.ones((*theta.shape[:-1],1)),theta],axis=-1)
    # W = ((A^H A)^-1)^T: effective errors are t^T error_coefficients.
    return np.einsum('...p,pq,...q->...',t.conj(),W,t).real
def covariance(theta,W,q):
    f=factors(theta,W)
    return np.broadcast_to((q*f)[...,None,None,None]*np.eye(N),(*f.shape,K,N,N)).copy()
def component_bank(hd,C,book,W,q,w):
    B=len(hd);L=len(book);g=hd[:,None]+np.einsum('bknm,lm->blkn',C,book)
    cv=np.broadcast_to(covariance(book,W,q)[None],(B,L,K,N,N)).reshape(-1,K,N,N)
    b,v,j=solve(g.reshape(-1,K,N),cv,NOISE,np.repeat(np.broadcast_to(w,(B,K)),L,axis=0))
    ix=j.reshape(B,L).argmin(1);rr=np.arange(B)
    return book[ix],b.reshape(B,L,K)[rr,ix],v.reshape(B,L,N)[rr,ix],j.reshape(B,L)[rr,ix]
def component_ao(hd,C,book,W,q,w,steps=10):
    theta,b,v,j=component_bank(hd,C,book,W,q,w);initial=j.copy()
    for _ in range(steps):
        tn=theta.copy();u=np.einsum('bknm,bn->bkm',C,v.conj())*b[:,:,None]
        residual=b*np.einsum('bkn,bn->bk',hd,v.conj())+np.einsum('bkm,bm->bk',u,tn)-w
        gamma=q*np.sum(abs(v)**2,axis=1)*np.sum(abs(b)**2,axis=1)
        for m in range(M):
            without=residual-u[:,:,m]*tn[:,m,None]
            # Exact coordinate minimization of the LS-error-covariance surrogate.
            cross=W[m+1,0]+np.einsum('j,bj->b',W[m+1,1:],tn)-W[m+1,m+1]*tn[:,m]
            z=-np.sum(u[:,:,m].conj()*without,axis=1)-gamma*cross
            new=c.legacy.quantize_unit_modulus(np.exp(1j*np.angle(z)),2)
            tn[:,m]=new;residual=without+u[:,:,m]*new[:,None]
        cv=covariance(tn,W,q)
        fixed=c.obj(c.effective(hd,C,tn),b,v,cv,NOISE,w)
        assert np.max(fixed-j)<1e-9
        bn,vn,jn=solve(c.effective(hd,C,tn),cv,NOISE,w,v0=v)
        accept=jn<=j+1e-12
        theta[accept]=tn[accept];b[accept]=bn[accept];v[accept]=vn[accept];j[accept]=jn[accept]
    assert np.max(j-initial)<1e-9
    return theta,b,v,j
