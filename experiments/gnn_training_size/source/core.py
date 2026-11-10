"""Conditional-CSI radio kernels. All vectors use E|CN(0,q)|^2=q."""
import numpy as np
import conditional_csi as cc
import radio_core as legacy

def moments(samples):
 return (np.array([s.h_d_hat for s in samples]),np.array([s.C_hat for s in samples]),
         np.array([s.csi.user_covariance(np.ones(s.C_hat.shape[-1])) for s in samples]))
def effective(hd,C,theta):return hd+np.einsum('bknm,bm->bkn',C,theta)
def obj(g,b,v,cov,noise,w):
 a=np.einsum('bkn,bn->bk',g,v.conj())
 proj=np.einsum('bn,bknm,bm->bk',v.conj(),cov,v).real
 return np.sum(abs(b*a-w)**2+abs(b)**2*proj,axis=1)+noise*np.sum(abs(v)**2,axis=1)
from power_solver import solve
def bank(hd,C,book,cov,noise,w=None,criterion='function'):
 B,K,N=hd.shape;L=len(book)
 w=np.broadcast_to(np.ones(K)/K if w is None else w,(B,K))
 g=hd[:,None]+np.einsum('bknm,lm->blkn',C,book)
 if criterion=='function':
  cv=np.broadcast_to(cov[:,None],(B,L,K,N,N)).reshape(-1,K,N,N)
  bb,vv,jj=solve(g.reshape(-1,K,N),cv,noise,np.repeat(w,L,axis=0))
  ix=jj.reshape(B,L).argmin(axis=1);rr=np.arange(B)
  return book[ix],bb.reshape(B,L,K)[rr,ix],vv.reshape(B,L,N)[rr,ix],jj.reshape(B,L)[rr,ix]
 score=metric(g.reshape(-1,K,N),noise,criterion).reshape(B,L)
 ix=score.argmax(axis=1);theta=book[ix];b,v,j=solve(g[np.arange(B),ix],cov,noise,w)
 return theta,b,v,j
def metric(g,noise,kind):
 if kind=='power':return np.sum(abs(g)**2,axis=(1,2))
 A=np.eye(g.shape[-1])[None]+np.einsum('bkn,bkm->bnm',g,g.conj())/noise
 return np.linalg.slogdet(A)[1]/np.log(2)
def phase_sweep(hd,C,theta,b,v,w,bits=None):
 theta=theta.copy();u=np.einsum('bknm,bn->bkm',C,v.conj())*b[:,:,None]
 residual=b* np.einsum('bkn,bn->bk',hd,v.conj())+np.einsum('bkm,bm->bk',u,theta)-w
 for m in range(theta.shape[1]):
  without=residual-u[:,:,m]*theta[:,m,None]
  z=-np.sum(u[:,:,m].conj()*without,axis=1)
  new=np.exp(1j*np.angle(z))
  if bits is not None:new=legacy.quantize_unit_modulus(new,bits)
  theta[:,m]=new;residual=without+u[:,:,m]*new[:,None]
 return theta
def ao(hd,C,cov,noise,initial,w=None,iterations=10,bits=None):
 B,K,N=hd.shape;w=np.broadcast_to(np.ones(K)/K if w is None else w,(B,K))
 theta=initial.copy();b,v,j=solve(effective(hd,C,theta),cov,noise,w)
 initial_j=j.copy();history=[j.mean()]
 for _ in range(iterations):
  tn=phase_sweep(hd,C,theta,b,v,w,bits)
  bn,vn,jn=solve(effective(hd,C,tn),cov,noise,w,v0=v)
  accept=jn<=j+1e-12
  theta[accept]=tn[accept];b[accept]=bn[accept];v[accept]=vn[accept];j[accept]=jn[accept]
  history.append(j.mean())
 assert np.all(j<=initial_j+1e-10)
 return theta,b,v,j,np.array(history)
def normalize(delta,w):
 # Float32 mean/scale are the explicitly transmitted control fields.
 means=delta.mean(axis=1).astype(np.float32).astype(float)
 std=np.nextafter(np.sqrt(np.mean((delta-means[:,None])**2,axis=1)).astype(np.float32),np.float32(np.inf)).astype(float)
 std=np.maximum(std,1e-12);s0=std.max();u=(delta-means[:,None])/std[:,None]
 alpha=w*std/s0;offset=float(w@means)
 assert np.max(abs(offset+s0*(alpha@u)-w@delta))<1e-10
 return u,alpha,offset,s0
